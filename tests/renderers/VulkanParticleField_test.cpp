// VulkanParticleField_test — the PHASE 0 GATE for plans/particle-field.md.
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
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 256, kH = 160;
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

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

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
