// VulkanParticleField_test — the PHASE 0 GATE for plans/particle-field.md,
// plus the PHASE 1 draw/identity assertions.
//
// ParticleField's entire claim is that a field of N particles costs the CPU
// O(1) per frame: exactly one MeshEntry, one EntrySpan, one MaterialDesc /
// GeometryDesc / DrawInfo slot, whatever N is. That claim is not a design
// preference — it is the reason the type exists instead of an InstancedMesh —
// so it gets a test that would fail loudly if the entity were ever expanded
// per particle, BEFORE any representation is built on top of it.
//
// Three things are asserted, in ascending order of what they buy:
//
//   1. A scene containing a ParticleField renders headless with ZERO
//      validation errors. Phase 0 draws no particles, so the only thing this
//      can catch is the plumbing itself — buffers written in the wrong window
//      (VUID-03047), a freed handle a sibling frame still names — which is
//      exactly the defect class this backend keeps producing.
//   2. submit() round-trips: positions go in, frames render, liveCount is
//      plumbed, and a re-submit every frame for many frames neither crashes
//      nor leaks (the ring's slot rotation is exercised by construction, since
//      the ring is one slot deeper than the frames in flight).
//   3. THE GATE. With THREEPP_CPU_PHASE_PROFILE=1, the three phases that are
//      O(entries) by construction —
//          scene.3_leanMatrixRefresh   (per-span world-matrix refresh)
//          scene.7_tlasRefitFill       (per-entry TLAS instance descriptors)
//          frame.D_buildIndirect       (per-entry DrawInfo + draw commands)
//      — must be FLAT as the field's capacity goes 10k → 300k. If any of them
//      scales with capacity, the entity is not one entry and the whole design
//      is wrong; the plan says stop and fix before anything else, and this is
//      what says so out loud. Measured interleaved (A B A B), because that is
//      the repo's rule for any A/B claim.
//   4. PHASE 1. A MeshRepr field DRAWS: its live particles paint pixels as one
//      indirect draw whose instanceCount was written on the device, the ids
//      AOV carries a distinct particle index per particle in .w (plan §2.3 —
//      which is also the only proof that gl_InstanceIndex is the particle
//      index and not a draw index), a parked prefix paints strictly fewer
//      pixels, and dead slots (w < 0) paint none at all.
//
// The profiler registry is read directly rather than parsed off stderr: it is
// a static in an inline function, so the test binary and the library share one
// instance. THREEPP_CPU_PHASE_WINDOW is set absurdly high so the registry never
// auto-prints and auto-clears mid-measurement; the test clears it by hand.
//
// Plain exit-code program (not Catch2), matching the other tests here. Exits 42
// (→ CTest "Skipped") when no Vulkan/RT GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

// PRIVATE header, reachable because the test's include path carries src/. The
// gate needs the per-phase numbers programmatically; FrameTimings only carries
// the coarse cpu* totals, in which a per-particle regression would be invisible.
#include "threepp/renderers/vulkan/VulkanCpuPhaseProf.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // Big enough that the phase-2 dust captures are worth LOOKING at (they are
    // part of that phase's checkpoint), small enough that the phase-0 gate's
    // 240 measured frames still run in seconds.
    constexpr int kW = 640, kH = 400;
    constexpr int kSkipCode = 42;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    void setEnv(const char* k, const char* v) {
#ifdef _WIN32
        _putenv_s(k, v);
#else
        setenv(k, v, /*overwrite*/ 1);
#endif
    }

    // A deterministic slab of particles. Values matter only in that they are
    // finite and distinct — nothing shades them in phase 0 — but a fixed seed
    // keeps the memcpy sizes and the frame timings comparable run to run.
    std::vector<ParticlePos> makePositions(std::uint32_t n, unsigned seed) {
        std::vector<ParticlePos> out(n);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        for (std::uint32_t i = 0; i < n; ++i) {
            out[i] = {u(rng) * 2.f, 1.f + u(rng), u(rng) * 2.f, 1.f};
        }
        return out;
    }

    // ── CPU phase profiler access ───────────────────────────────────────────
    using threepp::vulkan::cpuprof::Registry;

    void profReset() {
        auto& r = Registry::get();
        r.acc.clear();
        r.frames = 0;
    }

    double profMsPerFrame(const char* key) {
        auto& r = Registry::get();
        if (r.frames == 0) return 0.0;
        const auto it = r.acc.find(key);
        if (it == r.acc.end()) return 0.0;
        return it->second / double(r.frames);
    }

    struct PhaseTriple {
        double lean = 0, tlas = 0, indirect = 0;
        void add(const PhaseTriple& o) {
            lean += o.lean;
            tlas += o.tlas;
            indirect += o.indirect;
        }
    };

    PhaseTriple readPhases() {
        PhaseTriple p;
        p.lean     = profMsPerFrame("scene.3_leanMatrixRefresh");
        p.tlas     = profMsPerFrame("scene.7_tlasRefitFill");
        p.indirect = profMsPerFrame("frame.D_buildIndirect");
        return p;
    }

    std::uint32_t entryTotal(const VulkanRenderer& r) {
        const auto s = r.autoLodStats();
        std::uint32_t n = 0;
        for (std::uint32_t v : s.entriesPerLevel) n += v;
        return n;
    }

    // ── Cross-process determinism harness (phase 2, checkpoint c) ───────────
    //
    // "The same scene rendered twice is byte-identical" CANNOT be tested by
    // comparing two consecutive frames of one run: this renderer's GI, ReSTIR
    // and froxel EMA are all stochastic per FRAME INDEX (seeded from
    // pc.frame), so frame N and frame N+1 legitimately differ by ~1.6% of
    // bytes on a completely static scene. What the r32ui fixed-point claim
    // actually asserts is RUN-TO-RUN reproducibility — render frame N of the
    // same scene in two separate processes and get the same bits — which is
    // also precisely the property a sensor golden needs, and precisely the one
    // float atomics would destroy (their sum depends on the order the GPU
    // happened to retire them in, which is a per-run coin flip).
    //
    // So the test re-executes ITSELF as a child, which renders a fixed frame
    // count of one named scene and dumps the raw framebuffer. The parent runs
    // the pairs and compares bytes.
    constexpr int kDetFrames = 40;// fixed: the child's whole timeline

    // Scene variants the child can render. Every one of them uses the SAME
    // geometry, lights and camera — only the ParticleField differs — so a
    // difference between two of them is attributable to the field alone.
    // `Fire` is F0's determinism checkpoint: the same dust field with emission
    // turned on. Emission brings a per-PIXEL hash into the march (the banding
    // dither), so it is exactly the term that would break run-to-run
    // reproducibility if it were ever keyed on a frame index or a wall clock.
    // `Emit` is F2's determinism checkpoint: the SAME scene with a
    // device-emitted field instead of a submitted one. A stateless closed form
    // hashed from (seed, slot) and evaluated at a caller-supplied t has nothing
    // in it that could differ between two processes — no RNG, no wall clock, no
    // integration history — and this is what says so out loud.
    enum class DetMode { NoField, DensityOff, Parked, Dust, Fire, Emit };

    std::vector<ParticlePos> makeDustSlab(std::uint32_t n, const Vector3& c, const Vector3& h) {
        std::vector<ParticlePos> out(n);
        std::mt19937 rng(20260810u);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        for (std::uint32_t i = 0; i < n; ++i) {
            out[i] = {c.x + u(rng) * h.x, c.y + u(rng) * h.y, c.z + u(rng) * h.z, 1.f};
        }
        return out;
    }

    // The child. Returns an exit code; writes width*height*3 raw bytes.
    int runDetChild(DetMode mode, const std::string& outPath) {
        std::unique_ptr<Canvas> canvasPtr;
        std::unique_ptr<VulkanRenderer> rendererPtr;
        try {
            canvasPtr = std::make_unique<Canvas>(
                    Canvas::Parameters().title("pf-det").size(kW, kH).vsync(false).headless(true));
            rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
        } catch (const std::exception&) { return kSkipCode; }
        Canvas& canvas = *canvasPtr;
        VulkanRenderer& renderer = *rendererPtr;
        renderer.setDlss(false);
        renderer.setFsr(false);
        renderer.setRenderScale(1.0f);
        // PINNED: auto-exposure adapts on wall-clock dt, which is the one
        // genuinely non-reproducible input in the frame (repo perf/capture rule).
        renderer.setAutoExposure(false);
        renderer.setClearColor(Color(0.05f, 0.06f, 0.08f));

        Scene scene;
        auto light = DirectionalLight::create(0xffffff, 2.5f);
        light->position.set(4, 8, 6);
        scene.add(light);
        scene.add(AmbientLight::create(0x9fb4d8, 1.4f));
        auto ground = Mesh::create(PlaneGeometry::create(20.f, 20.f), MeshStandardMaterial::create());
        ground->rotateX(-math::PI / 2.f);
        scene.add(ground);
        auto backdropMat = MeshStandardMaterial::create();
        backdropMat->color = Color(0.85f, 0.82f, 0.75f);
        auto backdrop = Mesh::create(BoxGeometry::create(5.f, 2.4f, 0.2f), backdropMat);
        backdrop->position.set(0.f, 1.2f, -2.0f);
        scene.add(backdrop);
        auto box = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f), MeshStandardMaterial::create());
        box->position.set(0.f, 0.5f, 0.f);
        scene.add(box);

        const Vector3 dCenter(0.f, 1.3f, 1.8f);
        const Vector3 dHalf(3.0f, 1.5f, 1.4f);
        constexpr std::uint32_t kDust = 100'000;
        std::shared_ptr<ParticleField> field;
        std::vector<ParticlePos> dpos;
        if (mode == DetMode::Emit) {
            // A device-emitted, MESH-drawn field: the pixels come from
            // particle_emit.comp's closed form and from nothing else. The clock
            // is the frame index, never a wall clock, so the child's whole
            // timeline is a function of kDetFrames alone.
            ParticleField::Config cfg;
            cfg.capacity      = 50'000;
            cfg.ownership     = ParticleField::Ownership::Renderer;
            cfg.wSemantic     = ParticleField::WSemantic::Radius;
            cfg.uniformRadius = 0.03f;
            field = ParticleField::create(cfg);
            auto m = MeshStandardMaterial::create();
            m->color = Color(0.92f, 0.94f, 0.98f);
            field->setMeshRepr(BoxGeometry::create(0.06f, 0.06f, 0.06f), m);
            ParticleField::EmitterParams ep;
            ep.spawnCenter.set(0.f, 3.2f, 1.2f);
            ep.spawnHalfExtent.set(3.0f, 0.1f, 1.6f);
            ep.velocity.set(0.f, -1.3f, 0.f);
            ep.wind.set(0.3f, 0.f, 0.f);
            ep.driftAmplitude = 0.25f;
            ep.driftFrequency = 0.2f;
            ep.driftScale     = 4.f;
            ep.lifetime   = 3.0f;
            ep.dutyCycle  = 0.92f;
            ep.size       = 0.03f;
            ep.sizeJitter = 0.4f;
            ep.seed       = 424242u;
            field->setEmitter(ep);
            scene.add(field);
        } else if (mode != DetMode::NoField) {
            ParticleField::Config cfg;
            cfg.capacity = kDust;
            field = ParticleField::create(cfg);
            dpos = makeDustSlab(kDust, dCenter, dHalf);
            field->submit(dpos.data(), kDust);
            if (mode != DetMode::DensityOff) {
                field->setDensityRepr(dCenter, dHalf, 0.02f, 64);
            }
            if (mode == DetMode::Fire) {
                auto& dr = field->densityRepr();
                dr.emissiveIntensity = 40.f;
                dr.tempBottomK       = 1900.f;
                dr.tempTopK          = 900.f;
                dr.tempFalloff       = 1.6f;
            }
            if (mode == DetMode::Parked) field->setLiveCount(0);
            scene.add(field);
        }

        auto camera = PerspectiveCamera::create(55.f, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0.f, 2.2f, 6.f);
        camera->lookAt(Vector3(0.f, 0.8f, 0.f));

        for (int i = 0; i < kDetFrames; ++i) {
            // Fixed dt, driven by the frame index: the emitter's clock is the
            // caller's, and a capture must be a function of nothing else.
            if (mode == DetMode::Emit)
                field->setEmitterTime(float(i) * (1.f / 60.f), 1.f / 60.f);
            canvas.animateOnce([&] { renderer.render(scene, *camera); });
        }
        const auto px = renderer.readRGBPixels();
        FILE* f = std::fopen(outPath.c_str(), "wb");
        if (!f) return 2;
        std::fwrite(px.data(), 1, px.size(), f);
        std::fclose(f);
        return 0;
    }

    std::vector<unsigned char> readFileBytes(const std::string& p) {
        std::vector<unsigned char> out;
        FILE* f = std::fopen(p.c_str(), "rb");
        if (!f) return out;
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n > 0) {
            out.resize(std::size_t(n));
            if (std::fread(out.data(), 1, out.size(), f) != out.size()) out.clear();
        }
        std::fclose(f);
        return out;
    }

    // Spawn the child. The extra outer quote pair is the Windows `cmd /c`
    // convention (it strips the first and last quote of the whole line).
    int spawnDetChild(const std::string& exe, const char* mode, const std::string& out) {
        setEnv("THREEPP_PF_DET_MODE", mode);
        setEnv("THREEPP_PF_DET_OUT", out.c_str());
#ifdef _WIN32
        const std::string cmd = "\"\"" + exe + "\"\"";
#else
        const std::string cmd = "\"" + exe + "\"";
#endif
        const int rc = std::system(cmd.c_str());
        setEnv("THREEPP_PF_DET_MODE", "");
        setEnv("THREEPP_PF_DET_OUT", "");
        return rc;
    }

}// namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // Child mode: render ONE named scene for a fixed number of frames and dump
    // the raw framebuffer. Nothing else — no profiler, no assertions.
    if (const char* m = std::getenv("THREEPP_PF_DET_MODE")) {
        if (m[0] != '\0') {
            const char* out = std::getenv("THREEPP_PF_DET_OUT");
            if (!out) return 2;
            const std::string ms(m);
            const DetMode mode = ms == "nofield"    ? DetMode::NoField
                               : ms == "densityoff" ? DetMode::DensityOff
                               : ms == "parked"     ? DetMode::Parked
                               : ms == "fire"       ? DetMode::Fire
                               : ms == "emit"       ? DetMode::Emit
                                                    : DetMode::Dust;
            return runDetChild(mode, out);
        }
    }
    const std::string selfExe = argc > 0 ? std::string(argv[0]) : std::string();
    // Where the demonstration PNGs and the child processes' raw captures go.
    // Unset = no captures written and the cross-process comparisons fall back
    // to the working directory.
    const char* capDirForDet = std::getenv("THREEPP_PF_CAPTURE_DIR");

    // Before the Canvas: the profiler registry latches its flags on first use,
    // and the validation layer must be requested before the instance exists.
    setEnv("THREEPP_CPU_PHASE_PROFILE", "1");
    setEnv("THREEPP_CPU_PHASE_WINDOW", "100000000");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanParticleField_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    check(Registry::get().on, "CPU phase profiler is enabled");

    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setRenderScale(1.0f);
    // PINNED. Auto-exposure adapts on wall-clock dt, so it both (a) makes a
    // capture depend on how long the previous phase happened to take and
    // (b) actively CANCELS the thing phase 2 measures — dust brightens the
    // frame, the exposure walks back down, and the extinction the medium
    // applied disappears from the numbers. Repo perf/capture rule.
    renderer.setAutoExposure(false);
    renderer.setClearColor(Color(0.05f, 0.06f, 0.08f));

    Scene scene;
    auto light = DirectionalLight::create(0xffffff, 2.5f);
    light->position.set(4, 8, 6);
    scene.add(light);
    auto ground = Mesh::create(PlaneGeometry::create(20.f, 20.f), MeshStandardMaterial::create());
    ground->rotateX(-math::PI / 2.f);
    scene.add(ground);
    auto box = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f), MeshStandardMaterial::create());
    box->position.set(0.f, 0.5f, 0.f);
    scene.add(box);

    auto camera = PerspectiveCamera::create(55.f, float(kW) / float(kH), 0.1f, 100.f);
    camera->position.set(0.f, 2.2f, 6.f);
    camera->lookAt(Vector3(0.f, 0.8f, 0.f));

    const auto frame = [&] { canvas.animateOnce([&] { renderer.render(scene, *camera); }); };

    // ── Baseline: the scene without any field ───────────────────────────────
    for (int i = 0; i < 8; ++i) frame();
    const std::uint32_t baseEntries = entryTotal(renderer);
    std::printf("[info] baseline entries (no field): %u\n", baseEntries);

    const std::uint32_t errorsBeforeFields = vulkan::validationErrorCount();

    // ── Construction contract ───────────────────────────────────────────────
    {
        bool threw = false;
        try {
            ParticleField::Config bad;
            bad.capacity = 0;
            (void) ParticleField::create(bad);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "create() rejects capacity == 0");

        threw = false;
        try {
            ParticleField::Config bad;
            bad.capacity = 16;
            bad.ownership = ParticleField::Ownership::Interop;
            (void) ParticleField::create(bad);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "create() rejects Ownership::Interop in this phase");
    }
    check(sizeof(ParticlePos) == 16, "ParticlePos is 16 B");

    // ── (a)/(b): a field in the scene, submitted every frame ────────────────
    constexpr std::uint32_t kSmall = 10'000;
    constexpr std::uint32_t kLarge = 300'000;

    ParticleField::Config cfgSmall;
    cfgSmall.capacity      = kSmall;
    cfgSmall.uniformRadius = 0.01f;
    auto small = ParticleField::create(cfgSmall);

    ParticleField::Config cfgLarge;
    cfgLarge.capacity      = kLarge;
    cfgLarge.uniformRadius = 0.01f;
    auto large = ParticleField::create(cfgLarge);

    check(small->capacity() == kSmall, "capacity() reports the configured capacity");
    check(small->liveCount() == 0, "a fresh field is parked (liveCount == 0)");

    const auto posSmall = makePositions(kSmall, 1234u);
    const auto posLarge = makePositions(kLarge, 4321u);

    small->submit(posSmall.data(), kSmall);
    check(small->liveCount() == kSmall, "submit() sets liveCount");
    small->setLiveCount(kSmall / 2);
    check(small->liveCount() == kSmall / 2, "setLiveCount() parks a prefix");
    small->submit(posSmall.data(), kSmall + 5);
    check(small->liveCount() == kSmall, "submit() clamps n to capacity");

    scene.add(small);
    for (int i = 0; i < 8; ++i) {
        small->submit(posSmall.data(), kSmall);
        frame();
    }
    const std::uint32_t withSmall = entryTotal(renderer);
    std::printf("[info] entries with a %u-particle field: %u\n", kSmall, withSmall);
    check(baseEntries == 0 || withSmall == baseEntries + 1,
          "a ParticleField adds EXACTLY ONE entry");

    // The 300k field, in the same scene, must add exactly one more.
    scene.add(large);
    for (int i = 0; i < 8; ++i) {
        large->submit(posLarge.data(), kLarge);
        frame();
    }
    const std::uint32_t withBoth = entryTotal(renderer);
    std::printf("[info] entries with a %u-particle field too: %u\n", kLarge, withBoth);
    check(baseEntries == 0 || withBoth == baseEntries + 2,
          "a 300k-particle field also adds EXACTLY ONE entry");
    scene.remove(*large);
    for (int i = 0; i < 4; ++i) frame();

    // Park-don't-remove (plan §1.4). visible = false hides the field from
    // traverseVisible, so it builds no entries — but the field object is alive,
    // so its device ring must survive and be reusable on re-show. This is the
    // path a use-after-free in the sweep would live on.
    small->visible = false;
    for (int i = 0; i < 6; ++i) frame();
    check(baseEntries == 0 || entryTotal(renderer) == baseEntries,
          "a parked (visible=false) field builds no entry");
    small->visible = true;
    for (int i = 0; i < 6; ++i) {
        small->submit(posSmall.data(), kSmall);
        frame();
    }
    check(baseEntries == 0 || entryTotal(renderer) == baseEntries + 1,
          "an un-parked field comes back as one entry");

    // ── (c) THE GATE ────────────────────────────────────────────────────────
    // Both fields are moved AND re-submitted every measured frame, so all three
    // phases are actually doing work rather than hitting their static-frame
    // early-outs: a static scene would make the gate pass by measuring nothing.
    constexpr int kMeasureFrames = 60;
    constexpr int kPairs = 2;

    const auto measure = [&](const std::shared_ptr<ParticleField>& field,
                             const std::vector<ParticlePos>& src, std::uint32_t n) {
        scene.add(field);
        for (int i = 0; i < 12; ++i) {// warm: structural rebuild, then settle
            field->position.y = 1.f + 0.001f * float(i);
            field->submit(src.data(), n);
            frame();
        }
        profReset();
        for (int i = 0; i < kMeasureFrames; ++i) {
            field->position.y = 1.f + 0.001f * float(i % 7);
            field->submit(src.data(), n);
            frame();
        }
        const PhaseTriple p = readPhases();
        scene.remove(*field);
        for (int i = 0; i < 4; ++i) frame();
        return p;
    };

    PhaseTriple a{}, b{};
    for (int pair = 0; pair < kPairs; ++pair) {
        a.add(measure(small, posSmall, kSmall));
        b.add(measure(large, posLarge, kLarge));
    }
    const double inv = 1.0 / double(kPairs);
    a.lean *= inv; a.tlas *= inv; a.indirect *= inv;
    b.lean *= inv; b.tlas *= inv; b.indirect *= inv;

    std::printf("\n[gate] ms/frame, mean of %d interleaved pairs of %d frames\n",
                kPairs, kMeasureFrames);
    std::printf("  %-28s %10s %10s %8s\n", "phase", "10k", "300k", "ratio");
    const auto row = [&](const char* name, double lo, double hi) {
        const double ratio = lo > 0.0 ? hi / lo : 0.0;
        std::printf("  %-28s %10.5f %10.5f %8.2f\n", name, lo, hi, ratio);
    };
    row("scene.3_leanMatrixRefresh", a.lean, b.lean);
    row("scene.7_tlasRefitFill", a.tlas, b.tlas);
    row("frame.D_buildIndirect", a.indirect, b.indirect);

    // Flatness bound. Generous on purpose: these phases are microseconds at
    // this entry count, so a strict ratio would fail on timer noise rather than
    // on scaling. What it must catch is the real failure mode — an entity
    // expanded per particle, which would put 300k entries through all three and
    // land them in the MILLISECONDS. The absolute floor is what keeps a 3x
    // ratio between two sub-20us numbers from being an assertion at all.
    constexpr double kNoiseFloorMs = 0.020;
    constexpr double kMaxRatio     = 3.0;
    const auto flat = [&](const char* name, double lo, double hi) {
        const bool ok = hi <= kNoiseFloorMs || hi <= kMaxRatio * lo;
        check(ok, std::string(name) + " is flat from 10k to 300k capacity");
    };
    flat("scene.3_leanMatrixRefresh", a.lean, b.lean);
    flat("scene.7_tlasRefitFill", a.tlas, b.tlas);
    flat("frame.D_buildIndirect", a.indirect, b.indirect);

    // ── PHASE 1: MeshRepr actually draws, and each particle is identifiable ──
    //
    // The phase-0 gate above proves a field is one entry. This proves the one
    // entry PAINTS: a MeshRepr field renders its live particles as one indirect
    // draw whose instanceCount came off the device, the ids AOV carries a
    // per-particle index in .w (plan §2.3), and the dead-slot predicate (w < 0)
    // costs no pixels.
    //
    // Everything else is hidden so "not sky" means "a particle", which is what
    // makes the pixel counts assertions rather than eyeballing.
    {
        ground->visible = false;
        box->visible = false;

        constexpr std::uint32_t kGrid = 4;
        constexpr std::uint32_t kN    = kGrid * kGrid;

        ParticleField::Config cfg;
        cfg.capacity      = kN;
        cfg.uniformRadius = 0.125f;
        cfg.orientations  = true;
        auto drawn = ParticleField::create(cfg);
        // Proxy authored AT the particle size: under WSemantic::InvMass the w
        // channel is inverse mass and the proxy draws at scale 1.
        auto proxyMat = MeshStandardMaterial::create();
        proxyMat->color = Color(0.9f, 0.2f, 0.15f);
        drawn->setMeshRepr(BoxGeometry::create(0.25f, 0.25f, 0.25f), proxyMat);
        check(drawn->material() == proxyMat,
              "setMeshRepr publishes the material as the field's Mesh material "
              "(the G-buffer shades particles through the entry's MaterialDesc)");

        // Identity orientations — the pixels then depend only on position, so a
        // wrong dead-slot or instanceCount shows up as a count, not as a pose.
        std::vector<float> quats(std::size_t(kN) * 4u, 0.f);
        for (std::uint32_t i = 0; i < kN; ++i) quats[std::size_t(i) * 4u + 3u] = 1.f;
        drawn->setOrientations(quats.data(), kN);

        std::vector<ParticlePos> grid(kN);
        for (std::uint32_t i = 0; i < kN; ++i) {
            const float gx = float(i % kGrid) - 1.5f;
            const float gy = float(i / kGrid) - 1.5f;
            grid[i] = {gx * 0.6f, 1.2f + gy * 0.42f, 0.f, 1.f};
        }

        scene.add(drawn);
        drawn->submit(grid.data(), kN);
        for (int i = 0; i < 8; ++i) { drawn->submit(grid.data(), kN); frame(); }

        // Per-particle identity out of the ids AOV: .x is the entry index + 1
        // (0 = sky), .w is the particle index.
        const auto readIds = [&](std::vector<std::uint16_t>& px, int& w, int& h) {
            std::vector<std::uint8_t> raw;
            int bw = 0, bh = 0, bpp = 0;
            if (!renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, bw, bh, bpp))
                return false;
            w = bw; h = bh;
            px.resize(std::size_t(bw) * std::size_t(bh) * 4u);
            std::memcpy(px.data(), raw.data(), px.size() * sizeof(std::uint16_t));
            return true;
        };

        std::vector<std::uint16_t> ids;
        int iw = 0, ih = 0;
        check(readIds(ids, iw, ih), "ids AOV readback succeeded");

        std::vector<int> perParticle(kN, 0);
        std::size_t covered = 0, foreign = 0;
        std::uint16_t entryId = 0;
        for (std::size_t p = 0; p * 4u + 3u < ids.size(); ++p) {
            const std::uint16_t x = ids[p * 4u + 0u];
            if (x == 0) continue;// sky
            ++covered;
            if (entryId == 0) entryId = x;
            const std::uint16_t pid = ids[p * 4u + 3u];
            if (x != entryId || pid >= kN) ++foreign;
            else ++perParticle[pid];
        }
        std::printf("[info] %ux%u ids AOV: %zu particle pixels, entry id %u\n",
                    unsigned(iw), unsigned(ih), covered, unsigned(entryId));
        check(covered > 64, "a MeshRepr field paints pixels");
        check(foreign == 0, "every painted pixel is this field's entry with a particle index < N");

        std::uint32_t distinct = 0;
        for (int c : perParticle) if (c > 0) ++distinct;
        std::printf("[info] distinct particle indices visible in outIds.w: %u of %u\n",
                    distinct, kN);
        // Every grid cell is in frame and unoccluded, so all of them must have
        // their OWN index. A single shared value would mean gl_InstanceIndex is
        // not the particle index — the one thing this pipeline exists to do.
        check(distinct == kN, "outIds.w carries a distinct index per particle");

        // Dead slots. Same live count, but every w < 0: the proxy must collapse
        // to zero area, exactly like GrainField's zeroed 3x3.
        std::vector<ParticlePos> dead = grid;
        for (auto& p : dead) p.w = -1.f;
        for (int i = 0; i < 6; ++i) { drawn->submit(dead.data(), kN); frame(); }
        check(readIds(ids, iw, ih), "ids AOV readback succeeded (dead slots)");
        std::size_t deadCovered = 0;
        for (std::size_t p = 0; p * 4u + 3u < ids.size(); ++p)
            if (ids[p * 4u + 0u] != 0) ++deadCovered;
        std::printf("[info] particle pixels with every slot dead (w < 0): %zu\n", deadCovered);
        check(deadCovered == 0, "dead slots (w < 0) paint no pixels");

        // liveCount is what the device-side instanceCount copy publishes: half
        // the field parked must paint strictly fewer pixels than all of it.
        for (int i = 0; i < 6; ++i) { drawn->submit(grid.data(), kN / 2); frame(); }
        check(readIds(ids, iw, ih), "ids AOV readback succeeded (half live)");
        std::size_t halfCovered = 0;
        for (std::size_t p = 0; p * 4u + 3u < ids.size(); ++p)
            if (ids[p * 4u + 0u] != 0) ++halfCovered;
        std::printf("[info] particle pixels at liveCount = %u: %zu (of %zu at %u)\n",
                    kN / 2, halfCovered, covered, kN);
        check(halfCovered > 0 && halfCovered < covered,
              "the draw's instanceCount tracks the device-side live count");

        // ── Motion vectors ──────────────────────────────────────────────────
        // The one part of the phase that is invisible to a still capture and
        // catastrophic when wrong: TAA and every denoiser reproject through
        // this attachment. The particle vertex stage composes the PREVIOUS
        // world transform inline from prevPositions[i] with the same
        // orientation — there is no expansion pass and no materialised prev
        // matrix — so this is what proves that composition, and it is also what
        // would catch prevPosAddr pointing at an uninitialised ring slot
        // (which reads as "everything moved from nowhere", i.e. a full-screen
        // smear, and as a still image looks fine).
        const auto readMotion = [&](std::vector<float>& out, int& w, int& h) {
            std::vector<std::uint8_t> raw;
            int bw = 0, bh = 0, bpp = 0;
            if (!renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::Motion, raw, bw, bh, bpp))
                return false;
            w = bw; h = bh;
            const std::size_t n = std::size_t(bw) * std::size_t(bh) * 4u;
            out.resize(n);
            const auto* hf = reinterpret_cast<const std::uint16_t*>(raw.data());
            for (std::size_t i = 0; i < n; ++i) {
                // float16 -> float32, by hand: the AOV's native format, and a
                // dependency-free decode is shorter than pulling one in.
                const std::uint16_t v = hf[i];
                const std::uint32_t sign = std::uint32_t(v >> 15) << 31;
                std::int32_t exp = (v >> 10) & 0x1F;
                std::uint32_t man = v & 0x3FF;
                std::uint32_t bits;
                if (exp == 0) {
                    bits = man == 0 ? sign : 0u;// subnormals -> 0, close enough here
                    if (man != 0) {
                        float f = float(man) * 5.9604645e-8f;
                        std::memcpy(&bits, &f, 4);
                        bits |= sign;
                    }
                } else if (exp == 31) {
                    bits = sign | 0x7F800000u | (man << 13);
                } else {
                    bits = sign | (std::uint32_t(exp - 15 + 127) << 23) | (man << 13);
                }
                std::memcpy(&out[i], &bits, 4);
            }
            return true;
        };

        // (i) Static field, static camera: motion must be ZERO at every
        // particle pixel. A garbage or self-referential prevPos shows up here.
        for (int i = 0; i < 8; ++i) { drawn->submit(grid.data(), kN); frame(); }
        std::vector<float> mo;
        int mw = 0, mh = 0;
        check(readMotion(mo, mw, mh), "motion AOV readback succeeded (static)");
        check(readIds(ids, iw, ih) && mw == iw && mh == ih,
              "motion and ids AOVs share the pixel grid");
        float maxStatic = 0.f;
        for (std::size_t p = 0; p * 4u + 3u < ids.size(); ++p) {
            if (ids[p * 4u + 0u] == 0) continue;
            maxStatic = std::max(maxStatic, std::max(std::fabs(mo[p * 4u + 0u]),
                                                     std::fabs(mo[p * 4u + 1u])));
        }
        std::printf("[info] max |motion| over particle pixels, nothing moving: %.6f NDC\n",
                    double(maxStatic));
        check(maxStatic < 1e-3f, "a static particle produces zero motion");

        // (ii) Move every particle by a known world delta and check the motion
        // vector against the delta PROJECTED with the same camera. Per particle,
        // keyed by the outIds.w index — so this also re-proves that .w and the
        // transform belong to the same particle.
        constexpr float kDx = 0.20f;
        std::vector<ParticlePos> moved = grid;
        for (int step = 1; step <= 3; ++step) {
            for (std::uint32_t i = 0; i < kN; ++i) moved[i].x = grid[i].x + kDx * float(step);
            drawn->submit(moved.data(), kN);
            frame();
        }
        check(readMotion(mo, mw, mh), "motion AOV readback succeeded (moving)");
        check(readIds(ids, iw, ih), "ids AOV readback succeeded (moving)");

        camera->updateMatrixWorld();
        std::vector<double> sx(kN, 0.0), sy(kN, 0.0);
        std::vector<int>    cnt(kN, 0);
        for (std::size_t p = 0; p * 4u + 3u < ids.size(); ++p) {
            if (ids[p * 4u + 0u] == 0) continue;
            const std::uint16_t pid = ids[p * 4u + 3u];
            if (pid >= kN) continue;
            sx[pid] += double(mo[p * 4u + 0u]);
            sy[pid] += double(mo[p * 4u + 1u]);
            ++cnt[pid];
        }
        int compared = 0, agree = 0;
        double worst = 0.0;
        for (std::uint32_t i = 0; i < kN; ++i) {
            if (cnt[i] < 4) continue;// too few pixels to average meaningfully
            Vector3 cur(moved[i].x, moved[i].y, moved[i].z);
            Vector3 prv(moved[i].x - kDx, moved[i].y, moved[i].z);
            cur.project(*camera);
            prv.project(*camera);
            const double ex = double(prv.x - cur.x), ey = double(prv.y - cur.y);
            const double mx = sx[i] / cnt[i], my = sy[i] / cnt[i];
            const double err = std::sqrt((mx - ex) * (mx - ex) + (my - ey) * (my - ey));
            const double mag = std::sqrt(ex * ex + ey * ey);
            worst = std::max(worst, err / std::max(mag, 1e-6));
            ++compared;
            // 12%: the AOV is float16 and the average is over the particle's
            // pixels, whose depths span the proxy cube, so the projected delta
            // is not one exact number.
            if (err <= 0.12 * mag) ++agree;
        }
        std::printf("[info] motion vs projected delta: %d/%d particles agree, worst %.1f%% off\n",
                    agree, compared, worst * 100.0);
        check(compared >= int(kN) / 2, "enough particles were measurable for the motion check");
        check(agree == compared,
              "every particle's motion vector matches its own projected displacement");

        scene.remove(*drawn);
        ground->visible = true;
        box->visible = true;
        for (int i = 0; i < 4; ++i) frame();
    }

    // ── F2: Ownership::Renderer — the DEVICE EMITTER ────────────────────────
    //
    // plans/particle-atmosphere.md F-C. The mode's claim is that a field of N
    // particles is written entirely on the device from a closed form, so the
    // host neither allocates, walks nor uploads a single position. Five things
    // are asserted, and the first is the API contract rather than a pixel:
    //
    //   (i)   THE MODE SPLIT IS ENFORCED. submit() on a Renderer field throws;
    //         setEmitter()/setEmitterTime() on a HostRing field throw. A silent
    //         no-op in either direction renders an empty field with no
    //         diagnostic anywhere, which is a debugging session instead of an
    //         exception at the call site.
    //   (ii)  A Renderer field DRAWS — its particles paint pixels through the
    //         same MeshRepr path a HostRing field uses, at one entry, with
    //         per-particle indices in outIds.w. Nothing was submitted: every
    //         position on screen was computed by particle_emit.comp.
    //   (iii) MOTION VECTORS, which are the whole reason the emitter writes
    //         f(t) and f(t - dt) in one dispatch. dt == 0 must give EXACTLY
    //         zero motion (the frozen-capture contract), dt > 0 must give
    //         motion whose SIGN says the previous position was above a falling
    //         particle, and doubling dt must double it. Numbers cannot see a
    //         smeared image, but they can see a reversed or dead velocity, and
    //         that is the class this catches.
    //   (iv)  Parking (setLiveCount(0)) stops the emitter and paints nothing.
    //   (v)   A HostRing field in the SAME scene is unaffected — the two modes
    //         share one pass and one draw path, so this is the regression test
    //         for the fork inside it.
    {
        ground->visible = false;
        box->visible    = false;
        // Settle the entry list BEFORE the baseline is read: autoLodStats
        // reports the last RENDERED frame, so hiding two meshes and reading the
        // count in the same breath compares two different scenes.
        for (int i = 0; i < 4; ++i) frame();

        // (i) The mode split.
        {
            ParticleField::Config rc;
            rc.capacity  = 64;
            rc.ownership = ParticleField::Ownership::Renderer;
            auto f = ParticleField::create(rc);
            check(f->liveCount() == 64,
                  "a Renderer field's live count is its CAPACITY at construction "
                  "(dead slots self-identify with w < 0)");
            bool threw = false;
            try { std::vector<ParticlePos> p(4); f->submit(p.data(), 4); }
            catch (const std::invalid_argument&) { threw = true; }
            check(threw, "submit() on an Ownership::Renderer field throws");

            threw = false;
            try {
                ParticleField::Config hc;
                hc.capacity = 64;
                auto hf = ParticleField::create(hc);
                hf->setEmitter(ParticleField::EmitterParams{});
            } catch (const std::invalid_argument&) { threw = true; }
            check(threw, "setEmitter() on a HostRing field throws");

            threw = false;
            try {
                ParticleField::Config hc;
                hc.capacity = 64;
                ParticleField::create(hc)->setEmitterTime(1.f, 0.016f);
            } catch (const std::invalid_argument&) { threw = true; }
            check(threw, "setEmitterTime() on a HostRing field throws");
        }

        // (ii) It draws. A slab of slots born across the top of the view and
        // falling fast enough that one frame's displacement is several pixels —
        // which is what makes the motion assertions below have something to
        // measure.
        constexpr std::uint32_t kEmit = 3'000;
        constexpr float kFall = 3.0f;// m/s, downward
        ParticleField::Config ecfg;
        ecfg.capacity      = kEmit;
        ecfg.ownership     = ParticleField::Ownership::Renderer;
        ecfg.wSemantic     = ParticleField::WSemantic::Radius;
        ecfg.uniformRadius = 0.05f;
        auto emit = ParticleField::create(ecfg);
        auto emitMat = MeshStandardMaterial::create();
        emitMat->color = Color(0.15f, 0.85f, 0.35f);
        emit->setMeshRepr(BoxGeometry::create(0.10f, 0.10f, 0.10f), emitMat);
        {
            ParticleField::EmitterParams ep;
            ep.spawnCenter.set(0.f, 2.6f, 0.f);
            ep.spawnHalfExtent.set(1.6f, 0.05f, 0.8f);
            ep.velocity.set(0.f, -kFall, 0.f);
            ep.lifetime  = 1.2f;// 3.6 m of fall — clears the frame
            ep.dutyCycle = 1.f; // continuous, so every slot is alive
            ep.size      = 0.05f;
            ep.seed      = 7331u;
            emit->setEmitter(ep);
        }
        emit->setEmitterTime(2.f, 1.f / 60.f);
        const std::uint32_t entriesBeforeEmit = entryTotal(renderer);
        scene.add(emit);
        for (int i = 0; i < 10; ++i) {
            emit->setEmitterTime(2.f + float(i) / 60.f, 1.f / 60.f);
            frame();
        }
        check(baseEntries == 0 || entryTotal(renderer) == entriesBeforeEmit + 1,
              "a Renderer-owned field is also EXACTLY ONE entry");

        const auto readIds2 = [&](std::vector<std::uint16_t>& px, int& w, int& h) {
            std::vector<std::uint8_t> raw;
            int bw = 0, bh = 0, bpp = 0;
            if (!renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, bw, bh, bpp))
                return false;
            w = bw; h = bh;
            px.resize(std::size_t(bw) * std::size_t(bh) * 4u);
            std::memcpy(px.data(), raw.data(), px.size() * sizeof(std::uint16_t));
            return true;
        };
        std::vector<std::uint16_t> eids;
        int ew = 0, eh = 0;
        check(readIds2(eids, ew, eh), "ids AOV readback succeeded (device emitter)");
        std::size_t emitPixels = 0, badId = 0;
        std::uint16_t emitEntry = 0;
        std::vector<std::uint8_t> seen(kEmit, 0);
        for (std::size_t p = 0; p * 4u + 3u < eids.size(); ++p) {
            if (eids[p * 4u + 0u] == 0) continue;
            ++emitPixels;
            if (emitEntry == 0) emitEntry = eids[p * 4u + 0u];
            const std::uint16_t pid = eids[p * 4u + 3u];
            // outIds.w is 16 bits, so a capacity above 65536 wraps by design —
            // kEmit is below that, so an index out of range is a real fault.
            if (eids[p * 4u + 0u] != emitEntry || pid >= kEmit) ++badId;
            else seen[pid] = 1;
        }
        std::uint32_t distinctEmit = 0;
        for (std::uint8_t s : seen) distinctEmit += s;
        std::printf("[info] device emitter: %zu pixels, %u distinct particle indices "
                    "(capacity %u), entry id %u\n",
                    emitPixels, distinctEmit, kEmit, unsigned(emitEntry));
        check(emitPixels > 500,
              "a Renderer-owned field paints pixels with NOTHING submitted from the host");
        check(badId == 0, "every emitted pixel is this field's entry with an index < capacity");
        // Not all of them: the slab is wider than the frustum and slots occlude
        // each other. A large fraction is what proves gl_InstanceIndex is the
        // particle index rather than one shared value.
        check(distinctEmit > kEmit / 8,
              "outIds.w carries many distinct particle indices for a device-emitted field");

        // (iii) Motion vectors out of the closed form.
        const auto readMotion2 = [&](std::vector<float>& out, int& w, int& h) {
            std::vector<std::uint8_t> raw;
            int bw = 0, bh = 0, bpp = 0;
            if (!renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::Motion, raw, bw, bh, bpp))
                return false;
            w = bw; h = bh;
            const std::size_t n = std::size_t(bw) * std::size_t(bh) * 4u;
            out.resize(n);
            const auto* hf = reinterpret_cast<const std::uint16_t*>(raw.data());
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint16_t v = hf[i];
                const std::uint32_t sign = std::uint32_t(v >> 15) << 31;
                const std::int32_t exp = (v >> 10) & 0x1F;
                const std::uint32_t man = v & 0x3FF;
                std::uint32_t bits;
                if (exp == 0) bits = sign;
                else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
                else bits = sign | (std::uint32_t(exp - 15 + 127) << 23) | (man << 13);
                std::memcpy(&out[i], &bits, 4);
            }
            return true;
        };
        // Mean motion over the field's own pixels, segmented by the ids AOV.
        const auto emitMotion = [&](float dt, double& meanY, double& meanMag) {
            for (int i = 0; i < 8; ++i) {
                emit->setEmitterTime(3.f, dt);
                frame();
            }
            std::vector<float> mo;
            std::vector<std::uint16_t> id2;
            int mw = 0, mh = 0, iw2 = 0, ih2 = 0;
            meanY = meanMag = 0.0;
            if (!readMotion2(mo, mw, mh) || !readIds2(id2, iw2, ih2)) return false;
            double sy = 0, sm = 0;
            std::size_t n = 0;
            for (std::size_t p = 0; p * 4u + 3u < id2.size(); ++p) {
                if (id2[p * 4u + 0u] == 0) continue;
                const double mx = mo[p * 4u + 0u], my = mo[p * 4u + 1u];
                sy += my;
                sm += std::sqrt(mx * mx + my * my);
                ++n;
            }
            if (n == 0) return false;
            meanY = sy / double(n);
            meanMag = sm / double(n);
            return true;
        };

        // Frozen: dt == 0 makes f(t) and f(t - dt) the SAME expression, so the
        // two buffers are bit-identical and every particle reprojects onto
        // itself. This is what lets a still capture converge under TAA, and it
        // is exactly zero rather than merely small.
        double y0 = 0, m0 = 0;
        check(emitMotion(0.f, y0, m0), "motion AOV readback succeeded (emitter frozen)");
        std::printf("[info] emitter dt = 0: mean |motion| over field pixels %.6f NDC\n", m0);
        check(m0 < 1e-4, "dt = 0 freezes the field: every particle has ZERO motion");

        double y1 = 0, m1 = 0, y2 = 0, m2 = 0;
        check(emitMotion(1.f / 60.f, y1, m1), "motion AOV readback succeeded (dt = 1/60)");
        check(emitMotion(2.f / 60.f, y2, m2), "motion AOV readback succeeded (dt = 2/60)");
        std::printf("[info] emitter dt = 1/60: mean motion .y %+.5f, |motion| %.5f NDC\n", y1, m1);
        std::printf("[info] emitter dt = 2/60: mean motion .y %+.5f, |motion| %.5f NDC\n", y2, m2);
        // SIGN. The particles fall, so their PREVIOUS position is ABOVE the
        // current one; the motion vector points current -> previous and the AOV
        // is GL-Y-up NDC, so .y must be positive. A negative mean here is
        // f(t + dt) instead of f(t - dt) — the reversed-velocity defect that no
        // image metric can see (plan F2).
        check(y1 > 0.0, "a falling particle's motion vector points UP (previous is above)");
        check(m1 > 5e-3, "dt = 1/60 produces a measurable motion vector");
        // LINEARITY. Twice the interval is twice the displacement, because the
        // trajectory is very nearly linear over 33 ms. This is what proves dt is
        // actually consumed rather than a constant hiding in the shader.
        const double ratio = m1 > 0.0 ? m2 / m1 : 0.0;
        std::printf("[info] |motion| ratio at 2x dt: %.3f (must be ~2)\n", ratio);
        check(ratio > 1.7 && ratio < 2.3,
              "the motion vector scales with the emitter's dt");

        // (iv) Park.
        emit->setLiveCount(0);
        for (int i = 0; i < 6; ++i) { emit->setEmitterTime(3.f, 1.f / 60.f); frame(); }
        check(readIds2(eids, ew, eh), "ids AOV readback succeeded (emitter parked)");
        std::size_t parkedPixels = 0;
        for (std::size_t p = 0; p * 4u + 3u < eids.size(); ++p)
            if (eids[p * 4u + 0u] != 0) ++parkedPixels;
        std::printf("[info] device emitter parked (liveCount 0): %zu pixels\n", parkedPixels);
        check(parkedPixels == 0, "a parked Renderer field paints nothing and skips its dispatch");
        emit->setLiveCount(kEmit);

        // (iv-b) THE EXACT DETERMINISM CLAIM, where it is exactly true.
        //
        // A framebuffer comparison cannot state this: two runs of the SAME
        // unmodified binary on this backend already differ by a percent or two
        // of bytes (the F0 amendment measures the floor), and a field of moving
        // particles keeps every temporal history in a permanent transient,
        // which amplifies it. So the emitter's purity is asserted on the
        // DENSITY VOLUME instead, which is integer fixed-point and therefore
        // bit-reproducible by construction — exactly the split F0 used for
        // emission.
        //
        // The test is a SEEK: evaluate at t, wander to a different t, come back
        // to t. A stateless closed form gives the same bytes; anything with
        // integration state, a frame counter or a wall clock in it cannot.
        {
            emit->setDensityRepr(Vector3(0.f, 1.6f, 0.f), Vector3(2.0f, 1.4f, 1.2f),
                                 0.6f, /*resolution*/ 32);
            const auto volAt = [&](float t, std::vector<std::uint32_t>& out) {
                for (int i = 0; i < 4; ++i) { emit->setEmitterTime(t, 1.f / 60.f); frame(); }
                std::uint32_t res = 0;
                return renderer.readParticleDensityVolume(*emit, out, res);
            };
            std::vector<std::uint32_t> vA, vElsewhere, vBack;
            const bool okSeek = volAt(4.25f, vA) && volAt(9.75f, vElsewhere) &&
                                volAt(4.25f, vBack);
            check(okSeek, "density volume readback succeeded for the device emitter");
            if (okSeek) {
                std::size_t occupied = 0, differ = 0, differElse = 0;
                for (std::size_t i = 0; i < vA.size(); ++i) {
                    if (vA[i] != 0) ++occupied;
                    if (vA[i] != vBack[i]) ++differ;
                    if (i < vElsewhere.size() && vA[i] != vElsewhere[i]) ++differElse;
                }
                std::printf("[info] emitter seek: %zu/%zu voxels occupied; t=4.25 revisited "
                            "differs in %zu voxels; a DIFFERENT t differs in %zu\n",
                            occupied, vA.size(), differ, differElse);
                check(occupied > 100, "the device-emitted field fills its density volume");
                check(differ == 0,
                      "the device emitter is a PURE FUNCTION of t: seeking away and back "
                      "reproduces the field BIT-IDENTICALLY");
                // The negative control: without it, a volume that was simply
                // never rewritten would pass the line above trivially.
                check(differElse > 0,
                      "a different t produces a different field (the seek test is not "
                      "measuring a stale volume)");
            }
            emit->densityRepr().enabled = false;
            for (int i = 0; i < 4; ++i) { emit->setEmitterTime(3.f, 1.f / 60.f); frame(); }
        }

        // (v) A HostRing field beside it. Disjoint in world space (and so on
        // screen), so each one's pixels are attributable, and the assertion is
        // that BOTH still paint: the two ownership modes share prepareFrame,
        // the counts copy, the indirect draw and the vertex stage, and this is
        // the regression test for the fork inside each of them.
        ParticleField::Config hcfg;
        hcfg.capacity      = 64;
        hcfg.uniformRadius = 0.12f;
        auto host = ParticleField::create(hcfg);
        auto hostMat = MeshStandardMaterial::create();
        hostMat->color = Color(0.90f, 0.25f, 0.15f);
        host->setMeshRepr(BoxGeometry::create(0.24f, 0.24f, 0.24f), hostMat);
        std::vector<ParticlePos> hostPos(64);
        for (std::uint32_t i = 0; i < 64; ++i)
            hostPos[i] = {-2.6f + 0.08f * float(i % 8), 0.5f + 0.30f * float(i / 8), 0.f, 1.f};
        scene.add(host);
        for (int i = 0; i < 8; ++i) {
            host->submit(hostPos.data(), 64);
            emit->setEmitterTime(3.f + float(i) / 60.f, 1.f / 60.f);
            frame();
        }
        check(readIds2(eids, ew, eh), "ids AOV readback succeeded (both modes in one scene)");
        std::size_t hostPixels = 0, devPixels = 0;
        const std::uint16_t hostEntry = 0;
        (void) hostEntry;
        for (std::size_t p = 0; p * 4u + 3u < eids.size(); ++p) {
            const std::uint16_t e = eids[p * 4u + 0u];
            if (e == 0) continue;
            if (e == emitEntry) ++devPixels;
            else ++hostPixels;
        }
        std::printf("[info] one scene, both modes: HostRing %zu px, Renderer %zu px\n",
                    hostPixels, devPixels);
        check(hostPixels > 100, "a HostRing field still draws beside a Renderer field");
        check(devPixels > 100, "a Renderer field still draws beside a HostRing field");

        scene.remove(*host);
        scene.remove(*emit);
        ground->visible = true;
        box->visible    = true;
        for (int i = 0; i < 4; ++i) frame();
    }

    // ── F2 checkpoint (e): the mode's ENTIRE per-frame CPU cost ─────────────
    //
    // The phase-0 gate above proved a HostRing field is O(1) in entry
    // bookkeeping — but it still pays one memcpy of capacity * 16 B per frame,
    // which shows up in frame.P_particleFields and grows with capacity. A
    // Renderer field pays NOTHING there: no host staging exists, the counts
    // block is written once per ring slot and never again, and the emitter's
    // whole state is a 128 B push constant. So this measures the phase directly
    // and demands it be FLAT from 10k to 300k — the thesis of the mode, stated
    // as a number rather than as a design intention.
    {
        constexpr int kMeasureFrames2 = 60;
        constexpr std::uint32_t kSmallEmit = 10'000, kLargeEmit = 300'000;

        const auto makeEmitter = [&](std::uint32_t cap) {
            ParticleField::Config c;
            c.capacity  = cap;
            c.ownership = ParticleField::Ownership::Renderer;
            c.wSemantic = ParticleField::WSemantic::Radius;
            c.uniformRadius = 0.02f;
            auto f = ParticleField::create(c);
            auto m = MeshStandardMaterial::create();
            f->setMeshRepr(BoxGeometry::create(0.04f, 0.04f, 0.04f), m);
            ParticleField::EmitterParams ep;
            ep.spawnCenter.set(0.f, 4.f, 0.f);
            ep.spawnHalfExtent.set(4.f, 0.1f, 4.f);
            ep.velocity.set(0.f, -1.2f, 0.f);
            ep.lifetime = 5.f;
            ep.size     = 0.02f;
            f->setEmitter(ep);
            return f;
        };
        auto emSmall = makeEmitter(kSmallEmit);
        auto emLarge = makeEmitter(kLargeEmit);

        struct EmitPhases { double lean = 0, tlas = 0, indirect = 0, fields = 0, emitMs = 0; };
        const auto measureEmit = [&](const std::shared_ptr<ParticleField>& f) {
            scene.add(f);
            for (int i = 0; i < 14; ++i) { f->setEmitterTime(float(i) / 60.f, 1.f / 60.f); frame(); }
            profReset();
            double gpu = 0.0;
            for (int i = 0; i < kMeasureFrames2; ++i) {
                f->setEmitterTime(1.f + float(i) / 60.f, 1.f / 60.f);
                frame();
                gpu += renderer.lastFrameTimings().particleEmitMs;
            }
            EmitPhases p;
            p.lean     = profMsPerFrame("scene.3_leanMatrixRefresh");
            p.tlas     = profMsPerFrame("scene.7_tlasRefitFill");
            p.indirect = profMsPerFrame("frame.D_buildIndirect");
            p.fields   = profMsPerFrame("frame.P_particleFields");
            p.emitMs   = gpu / double(kMeasureFrames2);
            scene.remove(*f);
            for (int i = 0; i < 4; ++i) frame();
            return p;
        };
        // Interleaved, repo A/B rule.
        EmitPhases lo{}, hi{};
        for (int pair = 0; pair < 2; ++pair) {
            const EmitPhases a = measureEmit(emSmall);
            const EmitPhases b = measureEmit(emLarge);
            lo.lean += a.lean; lo.tlas += a.tlas; lo.indirect += a.indirect;
            lo.fields += a.fields; lo.emitMs += a.emitMs;
            hi.lean += b.lean; hi.tlas += b.tlas; hi.indirect += b.indirect;
            hi.fields += b.fields; hi.emitMs += b.emitMs;
        }
        const double hinv = 0.5;
        lo.lean *= hinv; lo.tlas *= hinv; lo.indirect *= hinv; lo.fields *= hinv; lo.emitMs *= hinv;
        hi.lean *= hinv; hi.tlas *= hinv; hi.indirect *= hinv; hi.fields *= hinv; hi.emitMs *= hinv;

        std::printf("\n[gate] Ownership::Renderer CPU phases, ms/frame, 2 interleaved pairs "
                    "of %d frames\n", kMeasureFrames2);
        std::printf("  %-28s %10s %10s\n", "phase", "10k", "300k");
        const auto erow = [&](const char* n, double a, double b) {
            std::printf("  %-28s %10.5f %10.5f\n", n, a, b);
        };
        erow("scene.3_leanMatrixRefresh", lo.lean, hi.lean);
        erow("scene.7_tlasRefitFill", lo.tlas, hi.tlas);
        erow("frame.D_buildIndirect", lo.indirect, hi.indirect);
        erow("frame.P_particleFields", lo.fields, hi.fields);
        std::printf("  %-28s %10.5f %10.5f   (GPU)\n", "TP_ParticleEmit", lo.emitMs, hi.emitMs);

        // The same generous bound the phase-0 gate uses, and for the same
        // reason: these are microseconds, so an absolute floor is what keeps a
        // ratio between two sub-20us numbers from being an assertion at all.
        constexpr double kFloor2 = 0.020, kRatio2 = 3.0;
        const auto flat2 = [&](const char* n, double a, double b) {
            check(b <= kFloor2 || b <= kRatio2 * a,
                  std::string(n) + " is flat from 10k to 300k (Renderer ownership)");
        };
        flat2("scene.3_leanMatrixRefresh", lo.lean, hi.lean);
        flat2("scene.7_tlasRefitFill", lo.tlas, hi.tlas);
        flat2("frame.D_buildIndirect", lo.indirect, hi.indirect);
        flat2("frame.P_particleFields", lo.fields, hi.fields);
        // The strong form: whatever the capacity, the per-frame host work for a
        // device-emitted field is a push-constant fill. 20 us is 30x what it
        // measures and still a hundredth of what a 300k host memcpy costs.
        check(hi.fields < 0.020,
              "a 300k Renderer field costs ~zero CPU in frame.P_particleFields");
        check(hi.emitMs > 0.0, "the emit dispatch is bracketed (TP_ParticleEmit is nonzero)");
    }

    // ── PHASE 2: froxel density (dust) ──────────────────────────────────────
    //
    // The density representation's whole product is that a field of particles
    // NOBODY DRAWS still changes the image: σ_t rises inside a world box, the
    // froxel integrate pass folds it into the LUT's transmittance channel, and
    // the surface path reads that back to attenuate everything behind the
    // cloud and blend it toward the ambient haze. Three things are asserted:
    //
    //   (a) a 10^5-particle dust field CHANGES the frame, in the direction dust
    //       changes a frame — luminance contrast FALLS (haze washes the scene
    //       out) — and more σ changes it more. A pixel-count alone would pass
    //       on any bug that merely tinted the screen.
    //   (b) DETERMINISM. Consecutive frames of a settled static scene are
    //       byte-identical. This is the r32ui fixed-point claim: with float
    //       atomics the volume would differ every frame, because the order the
    //       GPU retires them in differs every frame.
    //   (c) a field whose DensityRepr is OFF is byte-identical to no field at
    //       all — the gates (heteroActive, the froxel dispatch, shade flag bit
    //       11) all stay shut, so a dust-free scene pays nothing and changes
    //       nothing. VulkanGolden_test covers the same claim scene-wide.
    {
        const char* capDir = capDirForDet;
        const auto capture = [&](const char* name) {
            if (!capDir) return;
            std::string p = std::string(capDir) + "/" + name + ".png";
            renderer.writeFramebuffer(p);
            std::printf("[capture] %s\n", p.c_str());
        };

        // Ambient/skylight: without it the medium has no light to scatter and
        // dust can only ever darken. applyParticleFog blends toward
        // ambient + env mean, exactly as scene fog does.
        auto amb = AmbientLight::create(0x8098c0, 0.45f);
        scene.add(amb);
        // The earlier phases render a deliberately flat white scene (they count
        // pixels, they don't look at them). Dust is judged by EYE, so the scene
        // is re-dressed here: mid-tone materials the haze can actually be seen
        // against, and a depth-graded row of markers so the extinction's
        // distance dependence reads as a gradient rather than a wash. A white
        // scene lit to 2.5 is already at the top of the range — nothing a
        // medium does to it is visible.
        light->intensity = 2.0f;
        std::dynamic_pointer_cast<MeshStandardMaterial>(ground->material())->color =
                Color(0.20f, 0.19f, 0.17f);
        auto boxMat = std::dynamic_pointer_cast<MeshStandardMaterial>(box->material());
        boxMat->color = Color(0.85f, 0.25f, 0.12f);

        auto backMat = MeshStandardMaterial::create();
        backMat->color = Color(0.40f, 0.42f, 0.46f);
        auto backdrop = Mesh::create(BoxGeometry::create(11.f, 3.4f, 0.2f), backMat);
        backdrop->position.set(0.f, 1.7f, -4.2f);
        scene.add(backdrop);
        constexpr std::uint16_t kBackdropClass = 39;
        renderer.setObjectClassId(*backdrop, kBackdropClass);

        // Markers at 5 depths through and behind the cloud, each with its own
        // semantic class so the ids AOV segments them. Bright and identical, so
        // the ONLY thing that distinguishes them in the final image is how much
        // dust the camera ray crossed to reach them: near ones stay crisp, far
        // ones wash out. That gradient IS the transmittance, and measuring it
        // per marker is what makes "occludes what is behind it" an assertion
        // rather than an impression.
        constexpr int kMarkers = 5;
        constexpr std::uint16_t kMarkerClass0 = 40;
        std::vector<std::shared_ptr<Mesh>> markers;
        for (int i = 0; i < kMarkers; ++i) {
            auto m = MeshStandardMaterial::create();
            m->color = Color(0.95f, 0.85f, 0.25f);
            auto pillar = Mesh::create(BoxGeometry::create(0.30f, 1.8f, 0.30f), m);
            pillar->position.set(-2.4f + 1.2f * float(i), 0.9f, 3.2f - 1.4f * float(i));
            scene.add(pillar);
            renderer.setObjectClassId(*pillar, std::uint32_t(kMarkerClass0 + i));
            markers.push_back(pillar);
        }

        // Mean luminance of each marker's own pixels, segmented by the class id
        // in the ids AOV (.z bits 8..15). Dust is not in the G-buffer, so the
        // mask is the same with and without it — one read is enough.
        std::vector<std::uint16_t> idsPx;
        const auto readIdsHere = [&] {
            std::vector<std::uint8_t> raw;
            int bw = 0, bh = 0, bpp = 0;
            if (!renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, bw, bh, bpp))
                return false;
            idsPx.resize(std::size_t(bw) * std::size_t(bh) * 4u);
            std::memcpy(idsPx.data(), raw.data(), idsPx.size() * sizeof(std::uint16_t));
            return true;
        };
        // Index 0..kMarkers-1 = the pillars, index kMarkers = the backdrop.
        const auto markerLuma = [&](const std::vector<unsigned char>& px) {
            std::array<double, kMarkers + 1> out{};
            std::array<std::size_t, kMarkers + 1> cnt{};
            const std::size_t n = std::min(px.size() / 3, idsPx.size() / 4);
            for (std::size_t i = 0; i < n; ++i) {
                if (idsPx[i * 4 + 0] == 0) continue;// sky
                const int c = int((idsPx[i * 4 + 2] >> 8) & 0xFFu);
                int slot = -1;
                if (c == int(kBackdropClass)) slot = kMarkers;
                else if (c >= int(kMarkerClass0) && c < int(kMarkerClass0) + kMarkers)
                    slot = c - int(kMarkerClass0);
                if (slot < 0) continue;
                out[std::size_t(slot)] += 0.2126 * px[i * 3 + 0] + 0.7152 * px[i * 3 + 1] +
                                          0.0722 * px[i * 3 + 2];
                ++cnt[std::size_t(slot)];
            }
            for (std::size_t i = 0; i < out.size(); ++i)
                out[i] /= double(std::max<std::size_t>(cnt[i], 1));
            return out;
        };
        const auto markerPixels = [&] {
            std::array<std::size_t, kMarkers + 1> cnt{};
            for (std::size_t i = 0; i + 3 < idsPx.size(); i += 4) {
                if (idsPx[i] == 0) continue;
                const int c = int((idsPx[i + 2] >> 8) & 0xFFu);
                if (c == int(kBackdropClass)) ++cnt[kMarkers];
                else if (c >= int(kMarkerClass0) && c < int(kMarkerClass0) + kMarkers)
                    ++cnt[std::size_t(c - int(kMarkerClass0))];
            }
            return cnt;
        };

        constexpr std::uint32_t kDust = 100'000;
        ParticleField::Config dcfg;
        dcfg.capacity = kDust;
        auto dust = ParticleField::create(dcfg);

        // The world box the volume covers: a slab standing between the camera
        // (z = 6) and the backdrop (z = -2).
        const Vector3 dCenter(0.f, 1.4f, 0.5f);
        const Vector3 dHalf(3.4f, 1.6f, 3.0f);
        // sigma per particle, chosen so 10^5 of them across this box give a
        // mean sigma_t of ~0.2 /m and therefore an optical depth of order 1
        // over the ~6 m the camera ray spends inside it — i.e. a cloud you can
        // see through but not past, which is what "dust" means.
        constexpr float kSigma = 0.5f;
        dust->setDensityRepr(dCenter, dHalf, kSigma, /*resolution*/ 64);
        // Warm dust against the cool skylight — and the value the froxel
        // injector uses for the point-light glow further down.
        dust->densityRepr().albedo = Color(0.95f, 0.87f, 0.74f);
        check(dust->densityRepr().enabled, "setDensityRepr enables the representation");
        check(dust->densityRepr().resolution == 64,
              "setDensityRepr latches the requested resolution");

        std::vector<ParticlePos> dpos(kDust);
        {
            std::mt19937 rng(20260810u);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            for (std::uint32_t i = 0; i < kDust; ++i) {
                dpos[i] = {dCenter.x + u(rng) * dHalf.x,
                           dCenter.y + u(rng) * dHalf.y,
                           dCenter.z + u(rng) * dHalf.z, 1.f};
            }
        }

        // Mean luminance, and THE directional metric.
        //
        // applyParticleFog is exactly  L' = L·T + H·(1 − T)  (Beer-Lambert
        // extinction toward the haze radiance H), so
        //
        //      L' − L  =  (T − 1)·(L − H)
        //
        // — the change in a pixel is LINEAR in that pixel's own brightness,
        // with slope T − 1. Regressing ΔL on L therefore recovers the mean
        // transmittance directly: the slope must be strictly negative when dust
        // is present, must go MORE negative as σ rises, and must be ~0 when
        // there is no dust. This is used instead of a mean or a std-dev because
        // both of those are sign-ambiguous — haze brightens a dark scene and
        // dims a bright one, so "the image changed" and "the image was
        // attenuated" are different statements and only this one is the second.
        const auto lumaOf = [](const std::vector<unsigned char>& px, std::size_t i) {
            return 0.2126 * px[i * 3 + 0] + 0.7152 * px[i * 3 + 1] + 0.0722 * px[i * 3 + 2];
        };
        const auto meanLuma = [&](const std::vector<unsigned char>& px) {
            const std::size_t n = px.size() / 3;
            if (n == 0) return 0.0;
            double sum = 0;
            for (std::size_t i = 0; i < n; ++i) sum += lumaOf(px, i);
            return sum / double(n);
        };
        // slope of (b − a) regressed on a. == mean transmittance − 1.
        const auto extinctionSlope = [&](const std::vector<unsigned char>& a,
                                         const std::vector<unsigned char>& b) {
            const std::size_t n = std::min(a.size(), b.size()) / 3;
            if (n == 0) return 0.0;
            double sx = 0, sy = 0;
            for (std::size_t i = 0; i < n; ++i) { sx += lumaOf(a, i); sy += lumaOf(b, i) - lumaOf(a, i); }
            const double mx = sx / double(n), my = sy / double(n);
            double cov = 0, var = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double dx = lumaOf(a, i) - mx;
                cov += dx * ((lumaOf(b, i) - lumaOf(a, i)) - my);
                var += dx * dx;
            }
            return var > 0.0 ? cov / var : 0.0;
        };
        const auto changedFraction = [](const std::vector<unsigned char>& a,
                                        const std::vector<unsigned char>& b) {
            if (a.size() != b.size() || a.empty()) return 1.0;
            std::size_t diff = 0;
            const std::size_t n = a.size() / 3;
            for (std::size_t i = 0; i < n; ++i) {
                if (std::abs(int(a[i * 3]) - int(b[i * 3])) > 2 ||
                    std::abs(int(a[i * 3 + 1]) - int(b[i * 3 + 1])) > 2 ||
                    std::abs(int(a[i * 3 + 2]) - int(b[i * 3 + 2])) > 2) ++diff;
            }
            return double(diff) / double(n);
        };

        constexpr int kSettle = 30;// TAA + the froxel EMA converge

        // The field goes in with its density representation OFF, so the A and
        // B legs below differ ONLY in that flag: same entry list, same TLAS,
        // same everything. Toggling it is not a structural change, so nothing
        // clears the TAA history between legs.
        dust->densityRepr().enabled = false;
        scene.add(dust);
        dust->submit(dpos.data(), kDust);
        for (int i = 0; i < 60; ++i) { dust->submit(dpos.data(), kDust); frame(); }
        const auto noFieldPx = renderer.readRGBPixels();
        capture("00_no_dust");

        // A/B, INTERLEAVED (repo rule for any A/B claim, and load-bearing
        // here): this renderer's probe GI is a slowly-converging irradiance
        // cache, so two captures taken minutes of frames apart differ by a
        // systematic drift that is LARGER than the effect being measured. A/B/A/B
        // with a same-leg control is what separates the two — the control is
        // literally "how much do two A legs differ", and the dust effect has to
        // clear it.
        constexpr int kPairs2 = 3;
        std::vector<std::vector<unsigned char>> legA, legB;
        const auto runLegs = [&](float sigma) {
            legA.clear();
            legB.clear();
            for (int p = 0; p < kPairs2; ++p) {
                dust->densityRepr().enabled = false;
                for (int i = 0; i < kSettle; ++i) { dust->submit(dpos.data(), kDust); frame(); }
                legA.push_back(renderer.readRGBPixels());
                dust->densityRepr().enabled          = true;
                dust->densityRepr().sigmaPerParticle = sigma;
                for (int i = 0; i < kSettle; ++i) { dust->submit(dpos.data(), kDust); frame(); }
                legB.push_back(renderer.readRGBPixels());
            }
        };
        const auto meanPairSlope = [&] {
            double s = 0;
            for (int p = 0; p < kPairs2; ++p) s += extinctionSlope(legA[p], legB[p]);
            return s / double(kPairs2);
        };
        const auto meanControlSlope = [&] {
            double s = 0;
            for (int p = 1; p < kPairs2; ++p) s += extinctionSlope(legA[p - 1], legA[p]);
            return s / double(kPairs2 - 1);
        };
        const auto meanPairChanged = [&] {
            double s = 0;
            for (int p = 0; p < kPairs2; ++p) s += changedFraction(legA[p], legB[p]);
            return s / double(kPairs2);
        };

        check(readIdsHere(), "ids AOV readback succeeded (marker segmentation)");

        runLegs(kSigma);
        capture("01_dust_sunlit");
        const auto dustPx      = legB.back();
        const auto fieldOffPx  = legA.back();
        const double changed   = meanPairChanged();
        const auto mOff  = markerLuma(fieldOffPx);
        const auto mDust = markerLuma(dustPx);
        std::printf("[info] 10^5-particle dust, %d interleaved A/B pairs: %.1f%% of "
                    "pixels changed; luma mean %.1f -> %.1f\n",
                    kPairs2, changed * 100.0, meanLuma(fieldOffPx), meanLuma(dustPx));
        const auto mPix = markerPixels();
        std::printf("[info] marker luma by depth (near -> far), no dust vs dust:\n");
        for (int i = 0; i < kMarkers; ++i) {
            std::printf("         marker %d (%5zu px)  %6.1f -> %6.1f  (%+6.1f)\n",
                        i, mPix[std::size_t(i)], mOff[std::size_t(i)],
                        mDust[std::size_t(i)],
                        mDust[std::size_t(i)] - mOff[std::size_t(i)]);
        }
        std::printf("         backdrop (%5zu px)  %6.1f -> %6.1f  (%+6.1f)\n",
                    mPix[kMarkers], mOff[kMarkers], mDust[kMarkers],
                    mDust[kMarkers] - mOff[kMarkers]);
        bool allVisible = true;
        for (std::size_t c : mPix) allVisible = allVisible && c > 200;
        check(changed > 0.15, "a 10^5-particle dust field changes the frame");
        check(allVisible, "every marker and the backdrop are visible in the frame");

        // ── The directional statement: DEPTH-GRADED OCCLUSION ───────────────
        //
        // Five IDENTICAL markers at five depths through one cloud. What the
        // medium does to each of them is exp(-tau) with tau proportional to the
        // dust the camera ray crossed to reach it, so the amount each marker's
        // pixels move must GROW WITH DEPTH — near ones barely, far ones a lot.
        // That gradient is Beer-Lambert measured on the image, and it is the
        // statement "objects behind the dust are hidden by it": marker 4 loses
        // its identity to the medium, marker 0 does not.
        //
        // Deliberately sign-free. The medium's radiance can sit above or below
        // a surface's — a lit dust cloud in front of a dark object BRIGHTENS it
        // — so "washed toward the medium" is the physical claim and "darkened"
        // is not. A signed test would pass or fail on the ambient level.
        //
        // (Contrast against the backdrop is printed for the reader but not
        // asserted: the far marker and the backdrop sit at nearly the same
        // depth, so the medium moves BOTH by nearly the same amount and their
        // mutual contrast is the one thing dust legitimately barely changes.)
        const auto contrast = [&](const std::array<double, kMarkers + 1>& m) {
            return std::fabs(m[kMarkers - 1] - m[kMarkers]);
        };
        const auto farDeltaOf = [&](const std::array<double, kMarkers + 1>& off,
                                    const std::array<double, kMarkers + 1>& on) {
            return std::fabs(on[kMarkers - 1] - off[kMarkers - 1]);
        };
        const double nearDelta = std::fabs(mDust[0] - mOff[0]);
        const double farDelta  = farDeltaOf(mOff, mDust);
        std::printf("[info] far-marker vs backdrop contrast (not asserted, same depth): "
                    "%.1f -> %.1f\n", contrast(mOff), contrast(mDust));
        std::printf("[info] |change| grows with path: near marker %.1f, far marker %.1f "
                    "(%.1fx)\n", nearDelta, farDelta,
                    nearDelta > 0 ? farDelta / nearDelta : 0.0);
        check(farDelta > 8.0,
              "dust visibly changes the objects behind it");
        check(farDelta > 2.0 * nearDelta,
              "the medium is INTEGRATED along the ray: the marker deepest in "
              "the cloud moves more than twice as far as the one at its near face");

        // Monotonicity in sigma: 5x must wash the far marker out harder still.
        runLegs(kSigma * 5.f);
        capture("02_dust_thick");
        const double farDeltaThick = farDeltaOf(markerLuma(legA.back()),
                                                markerLuma(legB.back()));
        std::printf("[info] 5x sigma: far-marker |change| %.1f -> %.1f\n",
                    farDelta, farDeltaThick);
        check(farDeltaThick > farDelta, "more sigma per particle occludes harder");
        dust->densityRepr().sigmaPerParticle = kSigma;
        dust->densityRepr().enabled          = true;
        for (int i = 0; i < kSettle; ++i) { dust->submit(dpos.data(), kDust); frame(); }

        // (b) DETERMINISM — the r32ui fixed-point claim, asserted where it
        // actually lives: on the VOLUME. The volume is cleared and rebuilt from
        // scratch every frame, so two consecutive frames of a static field are
        // two independent accumulations of the same 10^5 atomics in whatever
        // order the GPU chose that frame. Integer adds are associative, so the
        // bits must be identical; float atomics would differ in the low bits of
        // essentially every occupied voxel. Checked over several frames, and
        // against a re-submit of the same positions (which re-uploads the ring
        // slot, so the scatter genuinely re-runs rather than reading a cached
        // anything).
        {
            std::vector<std::uint32_t> volA, volB, volC;
            std::uint32_t vres = 0, vresB = 0, vresC = 0;
            for (int i = 0; i < 4; ++i) frame();
            const bool okA = renderer.readParticleDensityVolume(*dust, volA, vres);
            frame();
            const bool okB = renderer.readParticleDensityVolume(*dust, volB, vresB);
            for (int i = 0; i < 3; ++i) { dust->submit(dpos.data(), kDust); frame(); }
            const bool okC = renderer.readParticleDensityVolume(*dust, volC, vresC);
            check(okA && okB && okC, "density volume readback succeeded");
            check(vres == 64 && volA.size() == 64ull * 64ull * 64ull,
                  "the volume is the configured 64^3 of r32ui");

            std::size_t occupied = 0;
            std::uint32_t peak = 0;
            unsigned long long total = 0;
            for (std::uint32_t v : volA) {
                if (v != 0) ++occupied;
                peak = std::max(peak, v);
                total += v;
            }
            std::printf("[info] volume: %zu/%zu voxels occupied, peak fixed-point %u "
                        "(sigma_t %.4f /m), mean over occupied %.4f /m\n",
                        occupied, volA.size(), peak, double(peak) / 4096.0,
                        occupied ? double(total) / 4096.0 / double(occupied) : 0.0);
            check(occupied > 1000, "the scatter actually filled the volume");
            // Trilinear weights sum to 1, so the total fixed-point mass is the
            // particle count times sigma (bar boundary taps that fell outside
            // the box and per-tap rounding). Within 5% is the real bound.
            const double expected = double(kDust) * double(kSigma) * 4096.0;
            std::printf("[info] volume mass %.0f vs expected %.0f (%.2f%% off)\n",
                        double(total), expected,
                        100.0 * std::fabs(double(total) - expected) / expected);
            check(std::fabs(double(total) - expected) < 0.05 * expected,
                  "the 8-tap trilinear splat conserves optical mass");

            std::size_t dAB = 0, dAC = 0;
            for (std::size_t i = 0; i < volA.size(); ++i) {
                if (i < volB.size() && volA[i] != volB[i]) ++dAB;
                if (i < volC.size() && volA[i] != volC[i]) ++dAC;
            }
            std::printf("[info] determinism: %zu/%zu voxels differ across consecutive "
                        "frames, %zu across a re-submit\n", dAB, volA.size(), dAC);
            check(dAB == 0 && dAC == 0,
                  "the density volume is BIT-IDENTICAL across renders "
                  "(the r32ui fixed-point claim)");
        }

        // A point light INSIDE the cloud: the in-scatter half. The glow is
        // marched per pixel inside applyParticleFog, against the world-anchored
        // density volume — it used to come out of the froxel LUT (which
        // multiplied its clustered-light in-scatter by the medium's own sigma),
        // but a view-anchored 128x72x64 grid cannot estimate sigma inside a
        // plume while the camera moves, and it quantised the smoke into
        // cell-sized blocks glued to the screen (plans/particle-atmosphere.md,
        // the 2026-08-11 amendment). Either way this needs no new *scene* code:
        // the light simply becomes visible because there is something to
        // scatter off.
        auto glow = PointLight::create(0xffc070, 90.f, 14.f);
        glow->position.set(0.f, 1.4f, 0.8f);
        scene.add(glow);
        for (int i = 0; i < kSettle; ++i) frame();
        const auto glowPx = renderer.readRGBPixels();
        capture("03_dust_pointlight_inscatter");
        std::printf("[info] point light inside the dust: luma mean %.1f -> %.1f\n",
                    meanLuma(dustPx), meanLuma(glowPx));
        check(meanLuma(glowPx) > meanLuma(dustPx),
              "a point light inside the dust scatters (the frame gets brighter)");
        scene.remove(*glow);

        // Park the field: liveCount 0 must shut every gate again, so the frame
        // returns to the density-off image. This is the path where a stale
        // volume or a stuck heteroActive would show.
        dust->setLiveCount(0);
        for (int i = 0; i < kSettle; ++i) frame();
        const auto parkedPx = renderer.readRGBPixels();
        capture("04_parked");
        std::printf("[info] parked field vs density-off, same process: %.2f%% of pixels "
                    "differ (exact test is cross-process)\n",
                    changedFraction(fieldOffPx, parkedPx) * 100.0);
        // The gates must have shut. Same metric: with no live particles the far
        // marker must be back where it was with no dust at all.
        const double farDeltaParked = farDeltaOf(mOff, markerLuma(parkedPx));
        std::printf("[info] parked far-marker |change| %.1f (dust %.1f)\n",
                    farDeltaParked, farDelta);
        check(farDeltaParked < 0.2 * farDelta,
              "a parked (liveCount == 0) density field occludes nothing");

        // ── Checkpoint (b): the scatter's GPU cost, and the froxel bill ──────
        // Two numbers, per §3.3: the per-particle half (TP_ParticleDensity —
        // clear + splat, scaling with count) and the fixed half (TP_Froxel —
        // inject + integrate, scaling with the GRID, not the particles; dust
        // only adds the 8-fetch volume sampling to it). The dust field is the
        // only medium in this scene, so froxelMs with the field parked is 0 —
        // the honest statement of "what does dust cost the froxel passes" is
        // the whole froxelMs with dust live, and that is what is bounded.
        {
            const auto timeOver = [&](int n) {
                double pd = 0.0, fx = 0.0;
                for (int i = 0; i < n; ++i) {
                    frame();
                    const auto t = renderer.lastFrameTimings();
                    pd += t.particleDensityMs;
                    fx += t.froxelMs;
                }
                return std::pair<double, double>{pd / n, fx / n};
            };

            dust->submit(dpos.data(), kDust);// un-park the 10^5 field
            for (int i = 0; i < kSettle; ++i) frame();
            const auto [pd1e5, fx1e5] = timeOver(30);

            // 10^6: a second field (capacity is fixed for life), same box.
            // Ten taps land in each voxel's neighbourhood per 64^3 cell on
            // average — the atomics contend, which is the worst case wanted.
            constexpr std::uint32_t kDustM = 1'000'000;
            ParticleField::Config mcfg;
            mcfg.capacity = kDustM;
            auto dustM = ParticleField::create(mcfg);
            dustM->setDensityRepr(dCenter, dHalf, kSigma * 0.1f, 64);
            std::vector<ParticlePos> mpos(kDustM);
            {
                std::mt19937 rng(20260811u);
                std::uniform_real_distribution<float> u(-1.f, 1.f);
                for (std::uint32_t i = 0; i < kDustM; ++i)
                    mpos[i] = {dCenter.x + u(rng) * dHalf.x, dCenter.y + u(rng) * dHalf.y,
                               dCenter.z + u(rng) * dHalf.z, 1.f};
            }
            dust->setLiveCount(0);
            scene.add(dustM);
            dustM->submit(mpos.data(), kDustM);
            for (int i = 0; i < kSettle; ++i) frame();
            const auto [pd1e6, fx1e6] = timeOver(30);

            std::printf("[gate] GPU ms/frame, mean of 30: scatter %.3f (10^5) / %.3f "
                        "(10^6), froxel inject+integrate %.3f / %.3f\n",
                        pd1e5, pd1e6, fx1e5, fx1e6);
            check(pd1e5 > 0.0, "the scatter pass is bracketed (nonzero at 10^5)");
            check(pd1e6 < 2.0, "10^6 particles scatter in < 2 ms");
            check(fx1e6 < 0.8, "the froxel passes stay under 0.8 ms with dust live");

            scene.remove(*dustM);
            for (int i = 0; i < 4; ++i) frame();
        }

        scene.remove(*dust);
        scene.remove(*backdrop);
        scene.remove(*amb);
        for (auto& m : markers) scene.remove(*m);
        for (int i = 0; i < 4; ++i) frame();
    }

    // ── F0: PER-FIELD MEDIUM PARAMS + EMISSION ──────────────────────────────
    //
    // plans/particle-atmosphere.md F-A. Two claims, and they are the two the
    // phase exists for:
    //
    //   (b) EMISSION. A field with DensityRepr::emissiveIntensity > 0 emits
    //       blackbody radiance in proportion to its own sigma, ramped over the
    //       normalised height of its box. A cone of particles must therefore
    //       render as a flame-shaped gradient — hot and bright at the base,
    //       cooling and dimming toward the tip — with NO flame texture and no
    //       billboard anywhere in the renderer. Judged by EYE from the capture
    //       (house rule), and bounded here by the one thing an image metric can
    //       state honestly: the emission is RED-DOMINANT and falls with height.
    //   (d) PER-FIELD ALBEDO. The wart this phase removed: `albedoAniso` used to
    //       be one shared vec4 filled from whichever field enumerated first, so
    //       two dust clouds with different albedos rendered identically tinted.
    //       Asserted by a SWAP: give field L the warm albedo and field R the
    //       cool one, then exchange them. Per field, the two regions must move
    //       in OPPOSITE directions. Under the old shared value they would move
    //       TOGETHER (both regions always wore the first field's albedo), which
    //       is what makes this a regression test and not a smoke test.
    {
        const char* capDir = capDirForDet;
        const auto capture = [&](const char* name) {
            if (!capDir) return;
            const std::string p = std::string(capDir) + "/" + name + ".png";
            renderer.writeFramebuffer(p);
            std::printf("[capture] %s\n", p.c_str());
        };

        // Night: the emission has to be what lights the frame, so the sun is
        // dropped to a rim and the ambient to a whisper. A daylit flame is a
        // grey smudge and proves nothing.
        const float sunWas = light->intensity;
        light->intensity = 0.12f;
        auto amb = AmbientLight::create(0x30405a, 0.10f);
        scene.add(amb);
        ground->visible = true;
        box->visible    = false;

        const Vector3 camWas = camera->position;
        camera->position.set(0.f, 1.15f, 3.4f);
        camera->lookAt(Vector3(0.f, 0.80f, 1.5f));

        // ── (b) the flame cone ──────────────────────────────────────────────
        const Vector3 fCenter(0.f, 0.80f, 1.5f);
        const Vector3 fHalf(0.32f, 0.62f, 0.32f);
        constexpr std::uint32_t kFlame = 60'000;
        ParticleField::Config fcfg;
        fcfg.capacity = kFlame;
        auto flame = ParticleField::create(fcfg);
        flame->setDensityRepr(fCenter, fHalf, 1.2f, /*resolution*/ 48);
        {
            auto& dr = flame->densityRepr();
            dr.albedo             = Color(0.35f, 0.30f, 0.28f);// sooty: emission dominates
            dr.anisotropy         = 0.25f;
            dr.emissiveIntensity  = 45.f;
            dr.tempBottomK        = 1950.f;
            dr.tempTopK           = 900.f;
            dr.tempFalloff        = 1.6f;
        }
        check(flame->densityRepr().emissiveIntensity > 0.f,
              "DensityRepr carries an emission ramp");

        // A CONE, tapering to a point, denser at the base — the shape a
        // buoyant diffusion flame has, built from nothing but positions. Fixed
        // seed: this capture must be the same image every run.
        std::vector<ParticlePos> fpos(kFlame);
        {
            std::mt19937 rng(20260811u);
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            for (std::uint32_t i = 0; i < kFlame; ++i) {
                const float h  = std::pow(u01(rng), 0.75f);      // biased to the base
                const float rr = fHalf.x * std::pow(1.f - h, 0.85f) * std::sqrt(u01(rng));
                const float th = u01(rng) * 2.f * math::PI;
                fpos[i] = {fCenter.x + rr * std::cos(th),
                           fCenter.y - fHalf.y + h * 2.f * fHalf.y,
                           fCenter.z + rr * std::sin(th), 1.f};
            }
        }
        scene.add(flame);
        flame->submit(fpos.data(), kFlame);
        for (int i = 0; i < 45; ++i) { flame->submit(fpos.data(), kFlame); frame(); }
        const auto firePx = renderer.readRGBPixels();
        capture("10_emissive_cone_taa");

        // Row means over the flame's screen column, top to bottom. The ramp
        // says the base is hotter (whiter, brighter) than the tip, and the
        // Stefan-Boltzmann magnitude says it is much brighter — so both the
        // luminance and the red dominance must GROW downward. Sign-explicit on
        // purpose: "the image changed" would pass on a uniform orange block.
        int fw = kW, fh = kH;
        const auto colStats = [&](const std::vector<unsigned char>& px, int y0, int y1) {
            double lum = 0, r = 0, b = 0;
            std::size_t n = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = fw * 7 / 16; x < fw * 9 / 16; ++x) {
                    const std::size_t i = std::size_t(y) * std::size_t(fw) + std::size_t(x);
                    if (i * 3 + 2 >= px.size()) continue;
                    lum += 0.2126 * px[i * 3] + 0.7152 * px[i * 3 + 1] + 0.0722 * px[i * 3 + 2];
                    r += px[i * 3];
                    b += px[i * 3 + 2];
                    ++n;
                }
            const double d = double(std::max<std::size_t>(n, 1));
            return std::array<double, 3>{lum / d, r / d, b / d};
        };
        // The flame spans roughly the middle third of the frame vertically at
        // this framing; sample its upper and lower halves.
        const auto top = colStats(firePx, fh * 30 / 100, fh * 45 / 100);
        const auto bot = colStats(firePx, fh * 55 / 100, fh * 70 / 100);
        std::printf("[info] emissive cone, flame column: upper luma %.1f (R %.1f, B %.1f), "
                    "lower luma %.1f (R %.1f, B %.1f)\n",
                    top[0], top[1], top[2], bot[0], bot[1], bot[2]);
        check(bot[0] > top[0] * 1.3,
              "the emission ramp is hotter at the base: the lower flame is "
              "markedly brighter than the tip");
        check(bot[1] > bot[2] * 1.5,
              "the emitted radiance is red-dominant (a blackbody at flame "
              "temperature, not a white glow)");

        // MSAA leg of the same capture — the plan asks for both, because the
        // emissive core is the highest-contrast silhouette this renderer draws
        // and MSAA is the one path with no temporal averaging to hide banding.
        renderer.setGbufferMsaa(4);
        for (int i = 0; i < 45; ++i) { flame->submit(fpos.data(), kFlame); frame(); }
        capture("11_emissive_cone_msaa");
        renderer.setGbufferMsaa(1);
        for (int i = 0; i < 20; ++i) { flame->submit(fpos.data(), kFlame); frame(); }

        // Emission off must return the frame to plain dust: the whole emissive
        // path (blackbody term, 32-step bump, hash dither) is behind one
        // uniform branch, and this is the assertion that the branch is real.
        flame->densityRepr().emissiveIntensity = 0.f;
        for (int i = 0; i < 40; ++i) { flame->submit(fpos.data(), kFlame); frame(); }
        const auto darkPx = renderer.readRGBPixels();
        capture("12_emission_off");
        const auto botOff = colStats(darkPx, fh * 55 / 100, fh * 70 / 100);
        std::printf("[info] emissiveIntensity 0: flame column luma %.1f -> %.1f\n",
                    bot[0], botOff[0]);
        check(botOff[0] < 0.25 * bot[0],
              "emissiveIntensity = 0 puts the field back to pure dust");
        scene.remove(*flame);
        for (int i = 0; i < 4; ++i) frame();

        // ── (d) two fields, two albedos, measured by SWAP ───────────────────
        light->intensity = 2.2f;
        amb->intensity   = 0.9f;
        camera->position.set(0.f, 1.6f, 5.2f);
        camera->lookAt(Vector3(0.f, 1.2f, 0.f));

        constexpr std::uint32_t kTwo = 40'000;
        const Vector3 lC(-1.30f, 1.20f, 0.f), rC(1.30f, 1.20f, 0.f);
        const Vector3 tHalf(0.55f, 0.75f, 0.55f);
        const auto slab = [&](const Vector3& c, unsigned seed) {
            std::vector<ParticlePos> out(kTwo);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            for (std::uint32_t i = 0; i < kTwo; ++i)
                out[i] = {c.x + u(rng) * tHalf.x, c.y + u(rng) * tHalf.y,
                          c.z + u(rng) * tHalf.z, 1.f};
            return out;
        };
        ParticleField::Config tcfg;
        tcfg.capacity = kTwo;
        auto fieldL = ParticleField::create(tcfg);
        auto fieldR = ParticleField::create(tcfg);
        fieldL->setDensityRepr(lC, tHalf, 0.9f, 48);
        fieldR->setDensityRepr(rC, tHalf, 0.9f, 48);
        const auto posL = slab(lC, 771u), posR = slab(rC, 991u);
        scene.add(fieldL);
        scene.add(fieldR);

        const Color warm(1.00f, 0.55f, 0.18f), cool(0.18f, 0.48f, 1.00f);
        // Mean (R - B) over each field's own screen half. Chromatic on purpose:
        // luminance would move with the albedo's brightness and could be
        // matched by two greys, but the SIGN of R-B is the tint itself.
        const auto tint = [&](const std::vector<unsigned char>& px, int x0, int x1) {
            double s = 0;
            std::size_t n = 0;
            for (int y = kH / 5; y < kH * 4 / 5; ++y)
                for (int x = x0; x < x1; ++x) {
                    const std::size_t i = std::size_t(y) * std::size_t(kW) + std::size_t(x);
                    if (i * 3 + 2 >= px.size()) continue;
                    s += double(px[i * 3]) - double(px[i * 3 + 2]);
                    ++n;
                }
            return s / double(std::max<std::size_t>(n, 1));
        };
        const auto legs = [&](const Color& cl, const Color& cr, const char* tag) {
            fieldL->densityRepr().albedo = cl;
            fieldR->densityRepr().albedo = cr;
            for (int i = 0; i < 40; ++i) {
                fieldL->submit(posL.data(), kTwo);
                fieldR->submit(posR.data(), kTwo);
                frame();
            }
            const auto px = renderer.readRGBPixels();
            capture(tag);
            return std::pair<double, double>{tint(px, kW / 12, kW * 5 / 12),
                                             tint(px, kW * 7 / 12, kW * 11 / 12)};
        };
        const auto [l1, r1] = legs(warm, cool, "13_two_fields_warm_left");
        const auto [l2, r2] = legs(cool, warm, "14_two_fields_warm_right");
        const double dL = l1 - l2, dR = r1 - r2;
        std::printf("[info] two fields, albedo SWAP: left tint(R-B) %.2f -> %.2f (%+.2f), "
                    "right %.2f -> %.2f (%+.2f)\n", l1, l2, dL, r1, r2, dR);
        check(std::fabs(dL) > 1.5 && std::fabs(dR) > 1.5,
              "each field's own screen region responds to ITS OWN albedo");
        check(dL * dR < 0.0,
              "swapping two fields' albedos moves their regions in OPPOSITE "
              "directions — the medium params are PER FIELD, not first-field-wins");

        scene.remove(*fieldL);
        scene.remove(*fieldR);
        scene.remove(*amb);
        light->intensity = sunWas;
        box->visible     = true;
        camera->position.copy(camWas);
        camera->lookAt(Vector3(0.f, 0.8f, 0.f));
        for (int i = 0; i < 4; ++i) frame();
    }

    // ── F3: BILLBOARDS (plans/particle-atmosphere.md F-D) ───────────────────
    //
    // One indirect draw of (4 vertices x liveCount) per field, composited in
    // the post-upscaler overlay slot. Three claims, and the third is the one
    // that would silently rot:
    //
    //   (a) a billboard field PUTS PIXELS ON THE SCREEN. Trivial to state and
    //       the only thing that catches "the draw was recorded but the
    //       descriptor allocation failed" — which is exactly how this pass
    //       failed its first run, silently, for 64 frames at a time.
    //   (b) it is REPRODUCIBLE in process: two renders of the same field at the
    //       same emitter time agree to within the same-scene control, measured
    //       here with the field parked. The quads carry a hashed brightness and
    //       a re-derived age, and both are pure functions of (seed, slot, t).
    //   (c) a field whose BillboardRepr is OFF is unaffected — the whole pass
    //       is behind that flag, so a scene that does not ask for quads must
    //       not pay for them or change because of them.
    {
        const char* capDir = capDirForDet;
        // A dark backdrop and no other content: additive quads over a plain
        // clear colour is the one framing where "how many pixels did the
        // billboards light up" is a COUNT rather than an estimate.
        Color clearWas;
        renderer.getClearColor(clearWas);
        renderer.setClearColor(Color(0.02f, 0.02f, 0.03f));
        const float sunWas2 = light->intensity;
        light->intensity = 0.05f;
        box->visible     = false;
        ground->visible  = false;
        const Vector3 camWas2 = camera->position;
        camera->position.set(0.f, 1.4f, 4.6f);
        camera->lookAt(Vector3(0.f, 1.4f, 0.f));

        constexpr std::uint32_t kSparks = 4'000;
        ParticleField::Config bcfg;
        bcfg.capacity      = kSparks;
        bcfg.ownership     = ParticleField::Ownership::Renderer;
        bcfg.wSemantic     = ParticleField::WSemantic::Radius;
        bcfg.uniformRadius = 0.02f;
        auto sparks = ParticleField::create(bcfg);

        ParticleField::EmitterParams se;
        se.spawnCenter.set(0.f, 0.6f, 0.f);
        se.spawnHalfExtent.set(0.9f, 0.05f, 0.9f);
        se.velocity.set(0.f, 0.9f, 0.f);
        se.speedSpread = 0.35f;
        se.lifetime    = 2.0f;
        se.lifetimeJitter = 0.4f;
        se.dutyCycle   = 0.85f;
        se.size        = 0.02f;
        se.sizeJitter  = 0.5f;
        se.seed        = 20260813u;
        sparks->setEmitter(se);
        // dt 0 FREEZES the field: f(t) and f(t - 0) are the same expression, so
        // every particle reprojects onto itself and there is nothing temporal
        // left for the comparison below to blame.
        sparks->setEmitterTime(3.0f, 0.f);
        scene.add(sparks);

        // (c) first, while the representation is still OFF: this is the control
        // AND the "unaffected" assertion's baseline.
        check(!sparks->billboardRepr().enabled,
              "BillboardRepr is off until it is asked for");
        for (int i = 0; i < 24; ++i) frame();
        const auto offA = renderer.readRGBPixels();
        for (int i = 0; i < 8; ++i) frame();
        const auto offB = renderer.readRGBPixels();

        // "Pixels the billboards lit" measured AGAINST the same frame without
        // them, not against an absolute threshold. The sky/ambient floor of
        // this scene sits well above black after the display transform, so an
        // absolute cut counts the whole frame; a difference against the
        // representation-off capture counts exactly what the quads added, and
        // the scene is otherwise byte-stable (no geometry, so no GI variance),
        // which makes that difference attributable with no slack at all.
        const auto changedPixels = [](const std::vector<unsigned char>& a,
                                      const std::vector<unsigned char>& b) {
            std::size_t n = 0;
            if (a.size() != b.size()) return n;
            for (std::size_t i = 0; i + 2 < a.size(); i += 3) {
                const int dr = std::abs(int(a[i]) - int(b[i]));
                const int dg = std::abs(int(a[i + 1]) - int(b[i + 1]));
                const int db = std::abs(int(a[i + 2]) - int(b[i + 2]));
                if (dr > 6 || dg > 6 || db > 6) ++n;
            }
            return n;
        };
        const auto diffBytesV = [](const std::vector<unsigned char>& a,
                                   const std::vector<unsigned char>& b) -> long long {
            if (a.empty() || a.size() != b.size()) return -1;
            long long d = 0;
            for (std::size_t i = 0; i < a.size(); ++i)
                if (a[i] != b[i]) ++d;
            return d;
        };

        const long long ctlDiff = diffBytesV(offA, offB);
        // The control's OWN "changed pixels": whatever the scene does to itself
        // between two renders with no billboards in it. The floor the claim
        // below has to clear.
        const std::size_t ctlChanged = changedPixels(offA, offB);

        // Now switch the representation on. Nothing else about the field, the
        // scene or the camera changes.
        sparks->setBillboardRepr(Color(1.00f, 0.72f, 0.34f), Color(1.00f, 0.16f, 0.02f),
                                 /*intensity*/ 4.0f);
        sparks->billboardRepr().sizeTaper    = 0.5f;
        sparks->billboardRepr().brightJitter = 0.5f;
        check(sparks->billboardRepr().enabled, "setBillboardRepr enables the representation");

        for (int i = 0; i < 24; ++i) frame();
        const auto onA = renderer.readRGBPixels();
        if (capDir) {
            const std::string p = std::string(capDir) + "/15_billboards.png";
            renderer.writeFramebuffer(p);
            std::printf("[capture] %s\n", p.c_str());
        }
        for (int i = 0; i < 8; ++i) frame();
        const auto onB = renderer.readRGBPixels();

        const std::size_t onChanged = changedPixels(onA, offA);
        const long long onDiff      = diffBytesV(onA, onB);
        std::printf("[info] billboards: pixels changed vs the repr-off frame %zu "
                    "(control's own run-to-run change: %zu) of %zu; run-to-run "
                    "bytes %lld (on) vs %lld (control)\n",
                    onChanged, ctlChanged, onA.size() / 3, onDiff, ctlDiff);

        // (a)
        check(onChanged > ctlChanged + 500,
              "a field with BillboardRepr on renders non-background pixels");
        // (b) Against the SAME-SCENE control, not against absolute identity —
        // the backend's GI/ReSTIR/TAA are stochastic per frame index and a
        // still scene can differ run to run (see the cross-process notes
        // above). The claim is that the quads add no NEW variance. On this
        // deliberately geometry-free scene the control comes out at exactly 0,
        // so the slack below is a guard for less quiet scenes, not headroom
        // this one needs.
        check(onDiff >= 0 && ctlDiff >= 0 &&
                      onDiff <= std::max<long long>(ctlDiff, long long(onA.size()) / 1000),
              "a frozen billboard field is reproducible in process (its "
              "brightness hash and its re-derived age are pure functions of "
              "seed, slot and t)");

        // ── SECONDARY VIEWS SEE THE BILLBOARDS ──────────────────────────────
        // The billboard draw rides the post-upscaler overlay SLOT, and almost
        // everything else in that pass is primary-only by scope (wireframe,
        // lines, HUD sprites, the MSAA inject/resolve, the lens stage). This
        // one is not, deliberately: embers and rain are SCENE CONTENT, so a
        // CameraSensor pointed at a campfire has to see them or the sensor and
        // the display disagree about what is in the world. A secondary view
        // therefore gets its own small render-pass instance onto its colour
        // target, with this view's camera and the 1-sample pipeline.
        //
        // Asserted the same way as the primary: the same view, with the
        // representation off and on, compared against itself.
        {
            auto secCam = PerspectiveCamera::create(55.f, float(kW) / float(kH), 0.1f, 100.f);
            secCam->position.copy(camera->position);
            secCam->lookAt(Vector3(0.f, 1.4f, 0.f));
            const std::uint32_t vh = renderer.addView(*secCam, kW, kH);
            if (vh == 0u) {
                check(false, "addView returned a handle");
            } else {
                sparks->billboardRepr().enabled = false;
                for (int i = 0; i < 16; ++i) frame();
                const auto secOff = renderer.readViewRGBPixels(vh);
                sparks->billboardRepr().enabled = true;
                for (int i = 0; i < 16; ++i) frame();
                const auto secOn = renderer.readViewRGBPixels(vh);
                const std::size_t secChanged = changedPixels(secOn, secOff);
                std::printf("[info] secondary view: pixels changed by the "
                            "billboards %zu of %zu\n",
                            secChanged, secOn.size() / 3);
                check(!secOn.empty() && secChanged > 500,
                      "a SECONDARY view renders the field's billboards too "
                      "(they are scene content, not a primary-view garnish)");
                (void) renderer.removeView(vh);
                for (int i = 0; i < 4; ++i) frame();
            }

            // ── ORTHOGRAPHIC ────────────────────────────────────────────────
            // The quad is built in VIEW space — offset the particle's
            // view-space centre along (1,0,0)/(0,1,0), then project — which is
            // exact under an orthographic projection as well as a perspective
            // one and needs no camera position. (The legacy billboard path
            // instead scales in CLIP space by proj[1][1] / |viewPos|, which is
            // a perspective-only approximation and would collapse here.) A
            // secondary view is the cheapest way to actually RUN that claim.
            auto orthoCam = OrthographicCamera::create(-2.4f, 2.4f, 1.5f, -1.5f, 0.1f, 40.f);
            orthoCam->position.set(0.f, 1.4f, 4.6f);
            orthoCam->lookAt(Vector3(0.f, 1.4f, 0.f));
            const std::uint32_t oh = renderer.addView(*orthoCam, kW, kH);
            if (oh == 0u) {
                check(false, "addView returned a handle for the ortho camera");
            } else {
                sparks->billboardRepr().enabled = false;
                for (int i = 0; i < 16; ++i) frame();
                const auto orthoOff = renderer.readViewRGBPixels(oh);
                sparks->billboardRepr().enabled = true;
                for (int i = 0; i < 16; ++i) frame();
                const auto orthoOn = renderer.readViewRGBPixels(oh);
                const std::size_t orthoChanged = changedPixels(orthoOn, orthoOff);
                std::printf("[info] ORTHOGRAPHIC view: pixels changed by the "
                            "billboards %zu of %zu\n",
                            orthoChanged, orthoOn.size() / 3);
                check(!orthoOn.empty() && orthoChanged > 500,
                      "billboards face an ORTHOGRAPHIC camera too (the quad is "
                      "built in view space, so the projection never enters the "
                      "sizing)");
                (void) renderer.removeView(oh);
                for (int i = 0; i < 4; ++i) frame();
            }
        }

        // (c) Turn it off again and the frame must come back to the control.
        sparks->billboardRepr().enabled = false;
        for (int i = 0; i < 24; ++i) frame();
        const auto offC = renderer.readRGBPixels();
        const long long backDiff = diffBytesV(offA, offC);
        std::printf("[info] billboards off again: bytes vs the original off "
                    "frame %lld (control %lld)\n", backDiff, ctlDiff);
        check(backDiff >= 0 &&
                      backDiff <= std::max<long long>(ctlDiff, long long(offA.size()) / 1000),
              "a field with BillboardRepr OFF is unaffected by the pass");

        scene.remove(*sparks);
        renderer.setClearColor(clearWas);
        light->intensity = sunWas2;
        box->visible     = true;
        ground->visible  = true;
        camera->position.copy(camWas2);
        camera->lookAt(Vector3(0.f, 0.8f, 0.f));
        for (int i = 0; i < 4; ++i) frame();
    }

    // ── PHASE 2, checkpoint (d): a dust-free scene costs no pixels ──────────
    //
    // Cross-process, at a FIXED frame index, so the stochastic-per-frame-index
    // parts of the renderer (GI, ReSTIR, TAA history) are asked the same
    // question twice rather than two different ones.
    //
    // It cannot be an exact byte comparison, and the CONTROL is what says so:
    // rendering the identical scene in two processes already differs by ~1% of
    // bytes on this backend, before any ParticleField exists. (That floor is a
    // pre-existing property — `nofield` vs `nofield` measures it with no field
    // in the scene at all — not something phase 2 introduced, and the exact
    // determinism claim is asserted on the density VOLUME above, where it is
    // both meaningful and exactly true.) So the assertion is that adding a
    // dust-free ParticleField does not move the image MORE than re-running the
    // same scene does.
    //
    //   nofield    vs nofield     — the control: the renderer's own run-to-run floor.
    //   nofield    vs densityoff  — a field whose DensityRepr is off.
    //   densityoff vs parked      — a density field with no live particles:
    //                               heteroActive, the froxel dispatch and shade
    //                               flag bit 11 must all stay shut.
    if (selfExe.empty()) {
        std::printf("\n[note] argv[0] unavailable — the cross-process comparisons "
                    "are skipped.\n");
    } else {
        const std::string tmp = capDirForDet ? std::string(capDirForDet) : std::string(".");
        const auto renderChild = [&](const char* mode, const char* tag) {
            const std::string p = tmp + "/det_" + mode + "_" + tag + ".raw";
            const int rc = spawnDetChild(selfExe, mode, p);
            return rc == kSkipCode ? std::vector<unsigned char>{} : readFileBytes(p);
        };
        const auto diffBytes = [](const std::vector<unsigned char>& a,
                                  const std::vector<unsigned char>& b) -> long long {
            if (a.empty() || b.empty() || a.size() != b.size()) return -1;
            long long d = 0;
            for (std::size_t i = 0; i < a.size(); ++i)
                if (a[i] != b[i]) ++d;
            return d;
        };
        std::printf("\n[phase 2] cross-process comparisons "
                    "(%d frames per child, raw RGB, byte for byte)\n", kDetFrames);
        const auto noFieldA = renderChild("nofield", "a");
        const auto noFieldB = renderChild("nofield", "b");
        const auto offA     = renderChild("densityoff", "a");
        const auto parkedA  = renderChild("parked", "a");
        // F0 checkpoint (c): emission must not cost determinism. The march's
        // banding dither is a hash of the PIXEL COORDINATE with no frame term,
        // so two processes rendering the same emissive scene to the same frame
        // index must agree to within the renderer's own run-to-run floor —
        // measured by the `nofield` control in the very same sweep, which is
        // what makes this a statement about emission rather than about the
        // backend's pre-existing RT/GI non-reproducibility.
        const auto fireA = renderChild("fire", "a");
        const auto fireB = renderChild("fire", "b");
        // F2 checkpoint (b): a DEVICE-emitted field, moving, in two processes.
        // Everything about it is a hash of (seed, slot) evaluated at a
        // caller-supplied t, so if it is ever not reproducible the cause is
        // something that should not be in the shader at all.
        const auto emitA = renderChild("emit", "a");
        const auto emitB = renderChild("emit", "b");
        const long long control = diffBytes(noFieldA, noFieldB);
        if (control < 0) {
            std::printf("[skip] cross-process comparisons (a child could not render)\n");
        } else {
            const long long total = static_cast<long long>(noFieldA.size());
            const long long dOff  = diffBytes(noFieldA, offA);
            const long long dPark = diffBytes(offA, parkedA);
            std::printf("[info] control    nofield vs nofield   : %lld / %lld bytes "
                        "(the renderer's run-to-run floor, no field in the scene)\n",
                        control, total);
            std::printf("[info]            nofield vs densityoff: %lld / %lld bytes\n",
                        dOff, total);
            std::printf("[info]            densityoff vs parked : %lld / %lld bytes\n",
                        dPark, total);
            // 2x the control, or 0.5% of the frame, whichever is larger: a real
            // regression here (dust leaking into a dust-free scene) is a
            // double-digit percentage, as the live-dust measurement above shows.
            const long long bound = std::max<long long>(control * 2, total / 200);
            check(dOff >= 0 && dOff <= bound,
                  "a ParticleField with DensityRepr OFF changes no more than "
                  "re-running the same scene does");
            check(dPark >= 0 && dPark <= bound,
                  "a parked density field changes no more than re-running the "
                  "same scene does");

            const long long dFire = diffBytes(fireA, fireB);
            std::printf("[info]            fire    vs fire      : %lld / %lld bytes "
                        "(F0 checkpoint c — emission keeps determinism)\n", dFire, total);
            if (dFire < 0) {
                std::printf("[skip] emissive determinism pair (a child could not render)\n");
            } else {
                // Same bound as above, and for the same reason: an emissive
                // march that had picked up a frame-keyed jitter would differ in
                // essentially every pixel the flame covers — a double-digit
                // percentage, nothing like this floor.
                check(dFire <= bound,
                      "an EMISSIVE particle field renders reproducibly across "
                      "processes (the step dither is a pixel hash, not a "
                      "temporal jitter)");
            }

            const long long dEmit = diffBytes(emitA, emitB);
            std::printf("[info]            emit    vs emit      : %lld / %lld bytes "
                        "(F2 checkpoint b — the device emitter is deterministic)\n",
                        dEmit, total);
            if (dEmit < 0) {
                std::printf("[skip] device-emitter determinism pair (a child could not render)\n");
            } else {
                // Same bound, same control. A field of 50k MOVING particles
                // whose positions came off a compute shader lands inside the
                // renderer's own run-to-run floor: no RNG state, no wall clock,
                // no integration history, and the atomics that would have
                // reordered are not in this path at all.
                check(dEmit <= bound,
                      "a DEVICE-EMITTED particle field renders reproducibly across "
                      "processes (closed form, hashed seed, caller's clock)");
            }
        }
    }

    // ── (a) validation ──────────────────────────────────────────────────────
    // Read AFTER everything above, so it covers field creation, the ring's slot
    // rotation, structural add/remove churn and the retire path.
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed — the zero-error\n"
                    "       assertion is skipped (it would be vacuously true).\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBeforeFields;
        std::printf("\n[info] validation errors provoked by the ParticleField phases: %u\n", errs);
        check(errs == 0, "ParticleField frames provoke no validation errors");
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
