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
