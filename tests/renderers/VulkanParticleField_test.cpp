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
    enum class DetMode { NoField, DensityOff, Parked, Dust };

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
        if (mode != DetMode::NoField) {
            ParticleField::Config cfg;
            cfg.capacity = kDust;
            field = ParticleField::create(cfg);
            dpos = makeDustSlab(kDust, dCenter, dHalf);
            field->submit(dpos.data(), kDust);
            if (mode != DetMode::DensityOff) {
                field->setDensityRepr(dCenter, dHalf, 0.02f, 64);
            }
            if (mode == DetMode::Parked) field->setLiveCount(0);
            scene.add(field);
        }

        auto camera = PerspectiveCamera::create(55.f, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0.f, 2.2f, 6.f);
        camera->lookAt(Vector3(0.f, 0.8f, 0.f));

        for (int i = 0; i < kDetFrames; ++i) {
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

        // A point light INSIDE the cloud: the in-scatter half. froxel_inject
        // multiplies its clustered-light glow by the medium's own sigma, so
        // this needs no new shading code at all — the light simply becomes
        // visible as a glow because there is now something to scatter off.
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
