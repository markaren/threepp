// VulkanMultiView_test — regression net for multi-view deferred rendering.
//
// VulkanRenderer::addView attaches a persistent extra view: its own G-buffer,
// its own temporal history, its own camera state, while the acceleration
// structures, lights, materials and textures are built once and shared. Every
// render() then produces the primary and every added view in ONE queue
// submission — the robot-camera-rig shape, where N cameras must see the SAME
// simulated instant.
//
// These were originally the gate modes of an examples/vulkan/vulkan_multi_view
// demo. They are assertions, not a demo, so they live here; the editor's camera
// dock is the thing to LOOK at. Run standalone (a plain exit-code program, not
// Catch2), or via `ctest -R VulkanMultiView_test`. Exits 42 (CTest "Skipped")
// when no Vulkan/RT GPU is available.
//
//   VulkanMultiView_test           run every gate below
//   VulkanMultiView_test --bench   opt-in: 0 vs 3 views, interleaved (not a gate)
//   VulkanMultiView_test --sweep   opt-in: marginal cost of one view, per res
//
// A NOTE ON MEASUREMENT, since it is what makes these gates trustworthy.
// Adding a view, removing one, or even re-aiming one changes frame time, and
// frame time feeds the TAA's dt-scaled temporal constants. A whole-image PSNR
// between two such configurations therefore moves by SEVERAL dB for reasons
// that have nothing to do with what is being tested. Every gate below either
// holds cost constant, measures against its own control run, or looks at a
// quantity the drift cannot touch. Two of them were wrong in exactly that way
// first — see the cull gate, whose earlier form compared captures taken 110
// frames apart and was measuring convergence, not culling.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "capture_util.hpp"// examples/vulkan (shared via target include dir)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 640, kH = 400;
    constexpr int kSkipCode = 42;// CTest SKIP_RETURN_CODE (no Vulkan/RT GPU)

    int failures = 0;

    // --shot <dir>: the gates' compared frames as binary PPMs, so a failing
    // PSNR can be LOOKED at instead of guessed at.
    std::string shotDir;
    void shot(const char* name, const std::vector<unsigned char>& rgb, int w, int h) {
        if (shotDir.empty() || rgb.size() != size_t(w) * h * 3) return;
        const std::string path = shotDir + "/" + name + ".ppm";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fprintf(f, "P6\n%d %d\n255\n", w, h);
            std::fwrite(rgb.data(), 1, rgb.size(), f);
            std::fclose(f);
        }
    }

    void check(bool ok, const char* what) {
        std::printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // A scene with the three things a secondary view has to get right: cast
    // shadows, bounced indirect light, and a mirror that reflects geometry it
    // cannot see directly. Deliberately NO fog — froxel volumetrics ARE per-view
    // (each view owns its DeferredShade, so it owns its froxel grid and cloud
    // march), but their in-scatter is view-direction dependent, so a
    // primary-vs-secondary parity comparison of a foggy scene would be comparing
    // two things that are not meant to be identical.
    void buildScene(Scene& scene) {
        scene.background = Color(0.03f, 0.035f, 0.06f);

        auto floorMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.72f, 0.70f, 0.66f)).roughness(0.85f).metalness(0.f));
        auto floor = Mesh::create(PlaneGeometry::create(40.f, 40.f), floorMat);
        floor->rotateX(-math::PI / 2.f);
        scene.add(floor);

        // Strongly coloured walls: the only way their colour reaches the floor
        // is a GI bounce, so a view that loses indirect light shows it loudly.
        auto redMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.85f, 0.10f, 0.09f)).roughness(0.9f).metalness(0.f));
        auto red = Mesh::create(BoxGeometry::create(0.3f, 6.f, 12.f), redMat);
        red->position.set(-5.f, 3.f, 0.f);
        scene.add(red);

        auto greenMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.10f, 0.72f, 0.16f)).roughness(0.9f).metalness(0.f));
        auto green = Mesh::create(BoxGeometry::create(0.3f, 6.f, 12.f), greenMat);
        green->position.set(5.f, 3.f, 0.f);
        scene.add(green);

        auto knotMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.95f, 0.42f, 0.30f)).roughness(0.35f).metalness(0.f));
        auto knot = Mesh::create(TorusKnotGeometry::create(0.9f, 0.28f, 128, 24), knotMat);
        knot->position.set(0.f, 1.9f, 0.f);
        scene.add(knot);

        // Near-mirror: reflections are a separate code path from GI, with their
        // own denoiser and their own history.
        auto mirrorMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.95f, 0.93f, 0.88f)).roughness(0.05f).metalness(1.f));
        auto mirror = Mesh::create(SphereGeometry::create(1.1f, 64, 32), mirrorMat);
        mirror->position.set(-2.4f, 1.1f, 1.6f);
        scene.add(mirror);

        auto boxMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.35f, 0.45f, 0.85f)).roughness(0.5f).metalness(0.f));
        auto box = Mesh::create(BoxGeometry::create(1.3f, 2.2f, 1.3f), boxMat);
        box->position.set(2.5f, 1.1f, -1.2f);
        scene.add(box);

        auto sun = DirectionalLight::create(0xffffff, 3.0f);
        sun->position.set(4.f, 8.f, 5.f);
        scene.add(sun);
        scene.add(AmbientLight::create(0x404050, 0.6f));
    }

    // Everything that makes the primary and a secondary comparable. Secondaries
    // are native-res built-in TAA by scope, so the primary is pinned to the same
    // resolve: no external upscaler, no render-scale, no sharpen, no motion
    // blur. Same reasoning (and the same switches) as VulkanGolden_test.
    void pinToTaaResolve(VulkanRenderer& r) {
        r.setDlss(false);
        r.setFsr(false);
        r.setRenderScale(1.0f);
        r.setSharpenStrength(0.f);
        r.setMotionBlur(0.f);
    }

    std::unique_ptr<PerspectiveCamera> makeCam(const Vector3& pos, const Vector3& at, float fov = 50.f) {
        auto c = std::make_unique<PerspectiveCamera>(fov, static_cast<float>(kW) / kH, 0.1f, 100.f);
        c->position.copy(pos);
        c->lookAt(at);
        c->updateMatrixWorld();
        return c;
    }

    // Drive n frames. `perFrame` runs AFTER this frame's render, so it sees a
    // completed frame and can read back or reconfigure for the next one.
    void runFrames(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& cam,
                   int n, const std::function<void(int)>& perFrame = {}) {
        for (int i = 0; i < n; ++i) {
            canvas.animateOnce([&] {
                r.render(scene, cam);
                if (perFrame) perFrame(i);
            });
        }
    }

    // Count pixels that read as the yellow wireframe marker.
    long yellowPixels(const std::vector<unsigned char>& img) {
        long n = 0;
        for (size_t i = 0; i + 2 < img.size(); i += 3)
            if (img[i] > 150 && img[i + 1] > 150 && img[i + 2] < 100) ++n;
        return n;
    }

    // ── Gate: parity vs primary ─────────────────────────────────────────────
    // The same camera, rendered as the primary and as a secondary view, over a
    // static scene. Any difference is the multi-view chain diverging from the
    // one that has always worked. Both chains run the same deterministic TAA
    // resolve, but their ray-traced GI/reflection/shadow samples are stochastic
    // and their histories started at different frames, so this is a convergence
    // gate, not a bit-exactness one.
    void gateParity(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        std::printf("\n[parity] same camera as primary and as secondary\n");
        constexpr int kWarm = 160;// past the 64-frame minimum: both chains must
                                  // be CONVERGED, not merely started
        auto secCam = makeCam(primaryCam.position, Vector3(0.f, 1.5f, 0.f));

        uint32_t vh = 0;
        // Added after the first frame: a view shares the primary's render pass,
        // which is built on the first render.
        runFrames(canvas, r, scene, primaryCam, kWarm, [&](int i) {
            if (i == 0) vh = r.addView(*secCam, kW, kH);
        });
        if (!vh) {
            check(false, "addView returned a handle");
            return;
        }

        const auto prim = r.readRGBPixels();
        const auto sec = r.readViewRGBPixels(vh);
        shot("parity_primary", prim, kW, kH);
        shot("parity_secondary", sec, kW, kH);
        check(!sec.empty() && sec.size() == prim.size(), "secondary readback matches primary size");
        if (sec.size() == prim.size() && !sec.empty()) {
            const auto d = capture::imageDiff(prim, sec);
            std::printf("  PSNR=%.2f dB  maxD=%d  hot=%.3f%%\n", d.psnr, d.maxD, d.hotPct);
            check(d.psnr >= 34.0, "secondary agrees with primary (PSNR >= 34 dB)");
        }
        (void) r.removeView(vh);
    }

    // ── Gate: history independence ──────────────────────────────────────────
    // Two secondary views. One is left alone; the other's camera is yanked
    // across the scene. If any temporal state is shared, the still view shows
    // the disturbance.
    //
    // Comparing the still view's own before/after inside ONE run does not
    // isolate anything: this scene keeps converging, so that number measures
    // convergence, not contamination — it reads ~39 dB whether or not the other
    // camera ever moves. What answers the question is running the whole
    // scenario TWICE, once with the cut and once without, and comparing the two
    // still images against the renderer's own run-to-run floor. The floor is
    // measured here rather than assumed, by running the no-cut scenario twice.
    void gateHistory(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        std::printf("\n[history] a cut in one view must not disturb another\n");
        constexpr int kSettle = 100, kAfter = 100;

        // Both cameras sit INSIDE the room and frame the whole scene. A view
        // pointed at a flat wall would pass trivially — no shadow, reflection or
        // GI structure left to disturb.
        const Vector3 stillPos(-3.4f, 2.7f, 5.2f), stillAt(0.f, 1.4f, 0.f);
        const Vector3 movePos(3.6f, 2.5f, 4.8f), moveAt(0.f, 1.5f, 0.f);

        // One scenario run: fresh views (fresh cameras → fresh uuids → fresh
        // temporal history), settle, optionally cut the OTHER camera, settle
        // again, return the still view's image.
        const auto runScenario = [&](bool cut) {
            auto movingCam = makeCam(movePos, moveAt);
            auto stillCam = makeCam(stillPos, stillAt);
            const uint32_t movingH = r.addView(*movingCam, kW, kH);
            const uint32_t stillH = r.addView(*stillCam, kW, kH);

            runFrames(canvas, r, scene, primaryCam, kSettle);
            if (cut) {
                // A hard cut for the OTHER view — the largest disturbance its
                // history can experience.
                movingCam->position.set(0.f, 7.5f, -7.f);
                movingCam->lookAt(Vector3(0.f, 1.f, 0.f));
                movingCam->updateMatrixWorld();
            }
            runFrames(canvas, r, scene, primaryCam, kAfter);

            auto still = r.readViewRGBPixels(stillH);
            (void) r.removeView(movingH);
            (void) r.removeView(stillH);
            // Let the removes retire before the next scenario allocates.
            runFrames(canvas, r, scene, primaryCam, 3);
            return still;
        };

        const auto controlA = runScenario(false);
        const auto controlB = runScenario(false);
        const auto test = runScenario(true);
        if (controlA.empty() || controlB.empty() || test.empty() ||
            controlA.size() != test.size() || controlA.size() != controlB.size()) {
            check(false, "history scenarios produced comparable images");
            return;
        }

        // The floor: two identical no-cut runs. Whatever they differ by is the
        // renderer's own run-to-run noise, and the cut run has to land there.
        const auto floor = capture::imageDiff(controlA, controlB);
        const auto cut = capture::imageDiff(controlA, test);
        std::printf("  control-vs-control : PSNR=%.2f dB (the floor)\n", floor.psnr);
        std::printf("  control-vs-cut     : PSNR=%.2f dB\n", cut.psnr);
        check(cut.psnr >= floor.psnr - 6.0,
              "a cut in the other view leaves the still view at its noise floor");
    }

    // ── Gate: add / render / read / remove / re-add ─────────────────────────
    // Exercises every lifecycle edge: creating views, rendering them, reading
    // them while live, destroying them mid-life, and re-creating at a different
    // resolution so nothing can quietly depend on the old extents. Run under the
    // validation layers (THREEPP_VULKAN_VALIDATION=1) for the real value.
    void gateLifecycle(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        std::printf("\n[lifecycle] add / render / read / remove / re-add\n");
        auto camA = makeCam(Vector3(5.f, 3.f, 5.f), Vector3(0.f, 1.5f, 0.f));
        auto camB = makeCam(Vector3(-4.f, 2.f, -5.f), Vector3(0.f, 1.5f, 0.f), 70.f);

        bool allSized = true, staleRefused = true, removesHonoured = true;
        for (int cycle = 0; cycle < 3; ++cycle) {
            const uint32_t hA = r.addView(*camA, 320, 200);
            const uint32_t hB = r.addView(*camB, 512, 320);
            if (!hA || !hB) {
                check(false, "addView returned handles every cycle");
                return;
            }
            runFrames(canvas, r, scene, primaryCam, 6);

            // Read while live — the readback path is part of the lifecycle.
            const auto a = r.readViewRGBPixels(hA);
            const auto b = r.readViewRGBPixels(hB);
            if (a.size() != size_t(320) * 200 * 3 || b.size() != size_t(512) * 320 * 3) allSized = false;

            // Remove out of order, and remove one twice: a stale handle must be
            // inert, not fatal.
            if (!r.removeView(hB) || !r.removeView(hA)) removesHonoured = false;
            if (r.removeView(hA)) staleRefused = false;
            runFrames(canvas, r, scene, primaryCam, 3);
        }
        check(allSized, "each view reads back at its OWN resolution");
        check(removesHonoured, "removeView honours a live handle");
        check(staleRefused, "removeView refuses a stale handle");
    }

    // ── Gate: the overlay sees the PRIMARY's frustum cull ───────────────────
    // The frustum-cull result used to be a single bool on the shared mesh entry.
    // Two readers consume it: the indirect draw builder (per view, fine) and the
    // primary's overlay DEPTH PREPASS, which lays down occluders so a wireframe
    // is correctly hidden behind solid geometry. With a shared bit the last view
    // to cull won, so a secondary aimed away from the occluder would delete that
    // occluder from the PRIMARY's prepass and a buried wireframe would show
    // through. No crash, no validation error, just wrong pixels, and only when a
    // secondary happens to run.
    //
    // The marker is BURIED inside the solid box, so correct behaviour is ~zero
    // visible yellow and the failure is hundreds of pixels. That is what makes
    // this immune to the drift confound: an earlier version of this check
    // compared two captures taken 110 frames apart and measured the background's
    // convergence rather than the culling — with the marker merely poking out of
    // the box, its anti-aliased edge pixels crossed the yellow threshold as the
    // scene settled, and the gate "failed" with nothing wrong. The exposed step
    // below keeps the check from being vacuous: a detector that can never see
    // yellow would otherwise pass this gate by seeing nothing.
    void gateCullIsolation(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        std::printf("\n[cull] the primary's overlay occlusion ignores a secondary's cull\n");
        auto wireMat = MeshBasicMaterial::create();
        wireMat->color = Color(1.f, 1.f, 0.f);
        wireMat->wireframe = true;
        auto wire = Mesh::create(SphereGeometry::create(0.45f, 16, 10), wireMat);
        // Step 1 position: out in the open, nothing in front of it.
        wire->position.set(2.5f, 3.6f, -1.2f);
        scene.add(wire);

        auto awayCam = makeCam(Vector3(0.f, 2.f, -5.5f), Vector3(0.f, 2.f, -20.f));// out of the room
        const uint32_t vh = r.addView(*awayCam, 320, 200);// present for the WHOLE gate, so cost never changes
        if (!vh) {
            check(false, "cull gate: addView returned a handle");
            scene.remove(*wire);
            return;
        }

        runFrames(canvas, r, scene, primaryCam, 60);
        const long exposed = yellowPixels(r.readRGBPixels());

        // Bury it: the box is 1.3 x 2.2 x 1.3 at (2.5, 1.1, -1.2), so a
        // radius-0.45 sphere at its centre is entirely inside with margin.
        wire->position.set(2.5f, 1.1f, -1.2f);
        wire->updateMatrixWorld();
        runFrames(canvas, r, scene, primaryCam, 60);
        const long buriedAway = yellowPixels(r.readRGBPixels());

        // Same view, same cost — now it sees the whole scene, so its frustum
        // cull keeps everything the "away" aim rejected.
        awayCam->position.set(3.8f, 2.4f, 4.4f);
        awayCam->lookAt(Vector3(0.f, 1.5f, 0.f));
        awayCam->updateMatrixWorld();
        runFrames(canvas, r, scene, primaryCam, 60);
        const long buriedAt = yellowPixels(r.readRGBPixels());

        std::printf("  visible wireframe px: exposed=%ld  buried/secondary-away=%ld  buried/secondary-at=%ld\n",
                    exposed, buriedAway, buriedAt);
        constexpr long kTol = 12;// a few anti-aliased edge pixels may survive
        check(exposed > 100, "the yellow detector sees an unoccluded marker (not vacuous)");
        check(buriedAway <= kTol, "buried marker stays hidden with the secondary aimed away");
        check(buriedAt <= kTol, "buried marker stays hidden with the secondary aimed at the scene");

        (void) r.removeView(vh);
        scene.remove(*wire);
        runFrames(canvas, r, scene, primaryCam, 3);
    }

    // ── Gate: per-view G-buffer AOVs (depth + segmentation) ─────────────────
    // The payoff for a synthetic-data rig: each camera yields not just colour
    // but lossless depth and the Ids attachment, whose .x channel is a stable
    // per-object instance id — so one readback per camera gives depth and
    // instance segmentation for the same simulated instant, from N viewpoints.
    //
    // Checks the two things that could be quietly wrong: that each view's AOV
    // has that VIEW's dimensions (not the primary's), and that two cameras
    // looking at the same scene from different places return DIFFERENT depth,
    // which is what a shared G-buffer would fail.
    void gateAov(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        std::printf("\n[aov] every camera in the rig yields depth and segmentation\n");
        auto camA = makeCam(Vector3(4.0f, 2.5f, 4.5f), Vector3(0.f, 1.5f, 0.f));
        auto camB = makeCam(Vector3(-4.0f, 2.5f, 4.5f), Vector3(0.f, 1.5f, 0.f));
        const uint32_t hA = r.addView(*camA, 320, 200);
        const uint32_t hB = r.addView(*camB, 512, 320);
        runFrames(canvas, r, scene, primaryCam, 40);

        std::vector<uint8_t> dA, dB, idA;
        int w = 0, h = 0, bpp = 0;
        const bool okA = r.readViewGBufferAOV(hA, VulkanRenderer::GBufferAOV::Depth, dA, w, h, bpp);
        check(okA && w == 320 && h == 200 && bpp == 4, "view A depth is 320x200 bpp=4");

        int wB = 0, hB2 = 0, bppB = 0;
        const bool okB = r.readViewGBufferAOV(hB, VulkanRenderer::GBufferAOV::Depth, dB, wB, hB2, bppB);
        check(okB && wB == 512 && hB2 == 320 && bppB == 4, "view B depth is 512x320 bpp=4");

        int wI = 0, hI = 0, bppI = 0;
        const bool okI = r.readViewGBufferAOV(hA, VulkanRenderer::GBufferAOV::Ids, idA, wI, hI, bppI);
        // Ids is RGBA16UI: .x = instanceCustomIndex + 1, 0 = sky. An Ids
        // attachment that was accidentally the primary's would still be
        // non-empty, so the dimension check is what separates them; this
        // confirms the attachment carries real labels rather than zeroes.
        long hits = 0;
        if (okI) {
            const auto* px = reinterpret_cast<const uint16_t*>(idA.data());
            for (long i = 0; i < long(wI) * hI; ++i)
                if (px[i * 4] != 0) ++hits;
        }
        std::printf("  view A ids: %dx%d bpp=%d, %ld/%d pixels carry an instance id\n",
                    wI, hI, bppI, hits, wI * hI);
        check(okI && wI == 320 && hI == 200 && bppI == 8 && hits > 0,
              "view A ids is 320x200 bpp=8 and carries real labels");

        // Two cameras, same scene, different places: their depth must not agree.
        // Compared over the overlapping top-left region, since the views differ
        // in size.
        long differing = 0, compared = 0;
        if (okA && okB) {
            const auto* fa = reinterpret_cast<const float*>(dA.data());
            const auto* fb = reinterpret_cast<const float*>(dB.data());
            for (int y = 0; y < 200; ++y)
                for (int x = 0; x < 320; ++x, ++compared)
                    if (std::fabs(fa[y * 320 + x] - fb[y * 512 + x]) > 1e-4f) ++differing;
        }
        const double pct = compared ? 100.0 * double(differing) / double(compared) : 0.0;
        std::printf("  A vs B depth differs on %.1f%% of compared pixels\n", pct);
        check(pct > 50.0, "two views disagree on depth (own G-buffer, not shared)");

        // A stale handle must refuse rather than quietly answer with the
        // primary's G-buffer.
        (void) r.removeView(hA);
        std::vector<uint8_t> junk;
        int jw = 0, jh = 0, jb = 0;
        check(!r.readViewGBufferAOV(hA, VulkanRenderer::GBufferAOV::Depth, junk, jw, jh, jb),
              "a removed view refuses an AOV read immediately");
        (void) r.removeView(hB);
        runFrames(canvas, r, scene, primaryCam, 3);
    }

    // ── Gate: splats are per-view, opt-in ───────────────────────────────────
    // The splat pass was primary-only: a secondary view (which is what a
    // CameraSensor is) saw EMPTY SPACE where a scan stood. setViewSplats
    // lifts that per view, and the whole point is that it is per view — two
    // views at the SAME pose in the SAME frame, differing only in the flag,
    // must differ only by the cloud.
    //
    // Unjittered raster (setGbufferMsaa(2), VulkanSplat_test's trick) for the
    // determinism half: with jitter on, consecutive frames legitimately differ
    // by a sub-pixel projection offset and "the same frame twice" is not.
    void gateSplats(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        std::printf("\n[splats] a secondary view renders a SplatCloud only if it asks\n");

        // High above the rest of the scene, so the cloud stands against the
        // background and coverage is countable without segmenting geometry.
        SplatGenerator::Options o;
        o.count = 60000;
        o.seed = 90210u;
        o.shDegree = 0;
        o.extent.set(1.6f, 1.0f, 1.6f);
        o.minScale = 0.05f;
        o.maxScale = 0.12f;
        o.minOpacity = 0.5f;
        o.maxOpacity = 0.95f;
        auto data = SplatGenerator::generate(o);
        for (size_t i = 0; i < data.count(); ++i) data.setDcColor(i, Vector3{0.95f, 0.35f, 0.10f});
        auto cloud = SplatCloud::create(data);
        cloud->position.set(0.f, 20.f, 0.f);
        scene.add(cloud);

        const uint32_t savedMsaa = r.gbufferMsaa();
        r.setGbufferMsaa(2);

        // Same pose, same size, same frame — the flag is the only difference.
        auto camOn  = makeCam(Vector3(0.f, 20.f, 6.f), Vector3(0.f, 20.f, 0.f));
        auto camOff = makeCam(Vector3(0.f, 20.f, 6.f), Vector3(0.f, 20.f, 0.f));
        constexpr int kVW = 256, kVH = 192;
        const uint32_t hOn  = r.addView(*camOn, kVW, kVH);
        const uint32_t hOff = r.addView(*camOff, kVW, kVH);
        check(hOn != 0 && hOff != 0, "splats: addView returned handles");
        check(r.setViewSplats(hOn, true) && !r.setViewSplats(0u, true),
              "setViewSplats takes a secondary handle and refuses the primary");
        check(r.viewSplats(hOn) && !r.viewSplats(hOff), "the flag reads back per view");

        runFrames(canvas, r, scene, primaryCam, 90);

        // Coverage: anything materially brighter than the background, which is
        // all these two views contain besides the cloud.
        const auto coverage = [](const std::vector<unsigned char>& img) {
            long n = 0;
            for (size_t i = 0; i + 2 < img.size(); i += 3)
                if (img[i] > 70 && img[i] > img[i + 2] + 25) ++n;
            return n;
        };
        const auto imgOn  = r.readViewRGBPixels(hOn);
        const auto imgOff = r.readViewRGBPixels(hOff);
        const long covOn = coverage(imgOn), covOff = coverage(imgOff);
        std::printf("  flagged view %ld / %d px carry the cloud, unflagged %ld\n",
                    covOn, kVW * kVH, covOff);
        check(covOn > 500 && covOff == 0,
              "the flagged view sees the cloud; the view beside it sees nothing");

        // Byte-identical across two renders: the sensor contract. Same camera,
        // same scene, converged, unjittered.
        runFrames(canvas, r, scene, primaryCam, 1);
        const auto again = r.readViewRGBPixels(hOn);
        check(again.size() == imgOn.size() && !again.empty() &&
                      std::memcmp(again.data(), imgOn.data(), imgOn.size()) == 0,
              "the flagged view's capture is byte-identical across two renders");

        // The primary must not move because a SECONDARY took the flag. Bounded
        // by a measured self-control rather than asserted byte-exact: this
        // scene's GI/reflections are stochastic and the renderer moves the
        // primary on its own between any two frames (plans/splat-surface-bake.md
        // P2 measured the same thing).
        const auto capturePrimary = [&] {
            runFrames(canvas, r, scene, primaryCam, 1);
            return r.readRGBPixels();
        };
        const auto maxDelta = [](const std::vector<unsigned char>& a,
                                 const std::vector<unsigned char>& b) {
            return a.size() == b.size() && !a.empty() ? capture::imageDiff(a, b).maxD : 255;
        };
        // The control has to carry the SAME churn as the measurement — the same
        // frame gap and the same setter call — or it measures a shorter interval
        // than the thing it is bounding.
        const auto step = [&](bool on) {
            (void) r.setViewSplats(hOn, on);
            runFrames(canvas, r, scene, primaryCam, 4);
            return capturePrimary();
        };
        const auto pOffA = step(false);
        const auto pOffB = step(false);
        const int control = maxDelta(pOffA, pOffB);
        const auto pOn = step(true);
        const int dFlag = maxDelta(pOffB, pOn);
        std::printf("  primary maxDelta: control %d, flag off vs on %d\n", control, dFlag);

        // Bounded by its own control rather than asserted byte-exact, and that
        // is a property of the renderer, not of this feature: two captures of
        // this static scene already differ by `control` counts on their own
        // (stochastic GI/ReSTIR plus the temporal resolve), and turning the flag
        // on changes frame time, which feeds the TAA's dt-scaled blend — see
        // this file's header. What the flag must not do is move the primary MORE
        // than the renderer moves it by itself. Measured: control ~10-15,
        // flag ~17-18 — the gap is the frame-time change (an opted-in view is
        // ~2.7 ms, printed below) feeding that dt-scaled blend, and it is why
        // the margin is this wide rather than 2. A real leak is not subtle: the
        // flagged view above puts 12000 bright pixels on screen.
        check(dFlag <= control + 12, "the primary image does not move with the flag");

        // The byte-exact half of the absence contract, aimed at the mechanism
        // that is actually at risk: the two targets SHARE the sort scratch, the
        // tile range buffer and one UBO allocation, so a secondary's dispatch
        // disturbing the primary's would land here first. The primary needs a
        // cloud of its own for this to say anything.
        //
        // The EXPANSION COUNT is the quantity: it is the primary's own
        // (splat, tile) pair count, a pure function of its projection. Not the
        // composited-pixel hash — that is taken over sceneHdr AFTER the shade
        // wrote it, so it inherits the shade's stochastic GI and is not equal
        // even between two consecutive frames here (it is printed, so the
        // difference is visible rather than assumed).
        auto primaryCloud = SplatCloud::create(data);
        primaryCloud->position.set(0.f, 1.6f, 2.6f);
        scene.add(primaryCloud);
        r.setSplatDebugChecksum(true);
        std::uint64_t sA[4]{}, sB[4]{}, sOn[4]{};
        const auto csum = [&](bool on, std::uint64_t out[4]) {
            (void) r.setViewSplats(hOn, on);
            // Warmed: the shared expansion budget grows once, on a truncated
            // frame, and that frame is genuinely different from the rest.
            runFrames(canvas, r, scene, primaryCam, 12);
            return r.splatDebugChecksum(out);
        };
        const bool got = csum(false, sA) && csum(false, sB) && csum(true, sOn);
        r.setSplatDebugChecksum(false);
        std::printf("  primary splat entries %llu; pixel hash: control %s, flag on %s\n",
                    (unsigned long long) sOn[3],
                    sA[2] == sB[2] ? "equal" : "differs",
                    sB[2] == sOn[2] ? "equal" : "differs");
        check(got && sA[3] > 0 && sA[3] == sB[3] && sB[3] == sOn[3],
              "the PRIMARY's own splat expansion is identical, flag on vs off");
        scene.remove(*primaryCloud);

        // ── measured, not asserted: what an opted-in view costs ─────────────
        // Interleaved A/B on the same binary, same frame, same everything but
        // the flag. Blocks rather than single frames so the TAA's dt-scaled
        // constants are not the thing being measured.
        const auto blockMs = [&](bool on, int n) {
            (void) r.setViewSplats(hOn, on);
            runFrames(canvas, r, scene, primaryCam, 8);
            const auto t0 = std::chrono::steady_clock::now();
            runFrames(canvas, r, scene, primaryCam, n);
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / n;
        };
        const auto measure = [&](const char* what) {
            double onMs = 0, offMs = 0;
            for (int rep = 0; rep < 2; ++rep) {
                offMs += blockMs(false, 40);
                onMs += blockMs(true, 40);
            }
            std::printf("  cost %-12s off %.3f ms/frame, on %.3f ms/frame, "
                        "per-view splat cost %.3f ms\n",
                        what, offMs / 2, onMs / 2, (onMs - offMs) / 2);
        };
        measure("60k splats");

        SplatGenerator::Options big = o;
        big.count = 500000;
        big.seed = 1337u;
        auto bigCloud = SplatCloud::create(SplatGenerator::generate(big));
        bigCloud->position.set(0.f, 20.f, 0.f);
        scene.remove(*cloud);
        scene.add(bigCloud);
        runFrames(canvas, r, scene, primaryCam, 10);
        measure("500k splats");

        scene.remove(*bigCloud);
        (void) r.removeView(hOn);
        (void) r.removeView(hOff);
        r.setGbufferMsaa(savedMsaa);
        runFrames(canvas, r, scene, primaryCam, 3);
    }

    // ── Gate: G-buffer MSAA + secondary views coexist ───────────────────────
    // setGbufferMsaa routes the raster through an MS pass whose dominant-sample
    // resolve is ONE shared GbufResolve instance with descriptors naming the
    // PRIMARY's images. Secondaries used to be given an MS framebuffer too, so
    // their record path re-entered that resolve: it re-read the primary's
    // samples, the secondary's own resolved G-buffer was never written, and its
    // depth was overwritten with the PRIMARY's. Secondaries now raster through
    // the 1x pass (no framebufferMS is created for them). This pins both
    // symptoms: a same-camera secondary must agree with the primary (it used to
    // shade from stale/garbage resolve targets), and two differently-placed
    // secondaries must DISAGREE on depth (they used to both carry the
    // primary's).
    void gateMsaa(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        std::printf("\n[msaa] secondary views coexist with setGbufferMsaa(2)\n");
        r.setGbufferMsaa(2);

        auto secCam = makeCam(primaryCam.position, Vector3(0.f, 1.5f, 0.f));
        auto sideA = makeCam(Vector3(4.0f, 2.5f, 4.5f), Vector3(0.f, 1.5f, 0.f));
        auto sideB = makeCam(Vector3(-4.0f, 2.5f, 4.5f), Vector3(0.f, 1.5f, 0.f));
        uint32_t vh = 0, hA = 0, hB = 0;
        runFrames(canvas, r, scene, primaryCam, 160, [&](int i) {
            if (i == 0) {
                vh = r.addView(*secCam, kW, kH);
                hA = r.addView(*sideA, 320, 200);
                hB = r.addView(*sideB, 320, 200);
            }
        });
        if (!vh || !hA || !hB) {
            check(false, "msaa gate: addView returned handles");
            r.setGbufferMsaa(1);
            return;
        }

        const auto prim = r.readRGBPixels();
        const auto sec = r.readViewRGBPixels(vh);
        shot("msaa_primary", prim, kW, kH);
        shot("msaa_secondary", sec, kW, kH);
        check(!sec.empty() && sec.size() == prim.size(), "msaa: secondary readback matches primary size");
        if (!sec.empty() && sec.size() == prim.size()) {
            const auto d = capture::imageDiff(prim, sec);
            std::printf("  PSNR=%.2f dB  maxD=%d  hot=%.3f%%\n", d.psnr, d.maxD, d.hotPct);
            check(d.psnr >= 30.0, "msaa: secondary agrees with primary (PSNR >= 30 dB)");
        }

        std::vector<uint8_t> dA, dB;
        int wA = 0, hgtA = 0, bppA = 0, wB = 0, hgtB = 0, bppB = 0;
        const bool okA = r.readViewGBufferAOV(hA, VulkanRenderer::GBufferAOV::Depth, dA, wA, hgtA, bppA);
        const bool okB = r.readViewGBufferAOV(hB, VulkanRenderer::GBufferAOV::Depth, dB, wB, hgtB, bppB);
        long differing = 0, compared = 0;
        if (okA && okB && dA.size() == dB.size()) {
            const auto* fa = reinterpret_cast<const float*>(dA.data());
            const auto* fb = reinterpret_cast<const float*>(dB.data());
            for (long i = 0; i < long(wA) * hgtA; ++i, ++compared)
                if (std::fabs(fa[i] - fb[i]) > 1e-4f) ++differing;
        }
        const double pct = compared ? 100.0 * double(differing) / double(compared) : 0.0;
        std::printf("  side A vs side B depth differs on %.1f%% of pixels\n", pct);
        check(okA && okB && pct > 50.0, "msaa: two secondaries carry their OWN depth");

        (void) r.removeView(vh);
        (void) r.removeView(hA);
        (void) r.removeView(hB);
        r.setGbufferMsaa(1);
        runFrames(canvas, r, scene, primaryCam, 3);
    }

    // ── Opt-in measurement: 0 vs 3 secondary views, interleaved ─────────────
    // Interleaved within ONE run, alternating in blocks, because per-session
    // clock/thermal drift makes two separate runs untrustworthy. Not a gate —
    // it reports numbers, it does not assert them.
    void benchViews(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        constexpr int kBlock = 60, kBlocks = 8, kWarm = 80;
        const int resW[3] = {320, 640, 640};
        const int resH[3] = {200, 400, 400};
        std::vector<std::unique_ptr<PerspectiveCamera>> cams;
        for (int i = 0; i < 3; ++i) {
            const float a = static_cast<float>(i) * 2.094f;
            cams.push_back(makeCam(Vector3(6.5f * std::cos(a), 2.6f + 0.7f * static_cast<float>(i), 6.5f * std::sin(a)),
                                   Vector3(0.f, 1.5f, 0.f)));
        }

        std::vector<uint32_t> handles;
        bool withViews = false;
        double sumWith = 0.0, sumWithout = 0.0;
        int nWith = 0, nWithout = 0, inBlock = 0;
        auto tPrev = std::chrono::high_resolution_clock::now();

        runFrames(canvas, r, scene, primaryCam, kWarm);
        for (int block = 0; block < kBlocks * 2; ++block) {
            for (inBlock = 0; inBlock < kBlock; ++inBlock) {
                canvas.animateOnce([&] { r.render(scene, primaryCam); });
                const auto now = std::chrono::high_resolution_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(now - tPrev).count();
                tPrev = now;
                // Skip each block's first frames (the add/remove stall).
                if (inBlock < 8) continue;
                if (withViews) { sumWith += ms; ++nWith; }
                else { sumWithout += ms; ++nWithout; }
            }
            // Flip configuration. Views are persistent objects, so this
            // add/remove is the BENCHMARK's churn, not the feature's — which is
            // why the first frames of each block are discarded above.
            withViews = !withViews;
            if (withViews) {
                for (int i = 0; i < 3; ++i) handles.push_back(r.addView(*cams[i], resW[i], resH[i]));
            } else {
                for (uint32_t h : handles) (void) r.removeView(h);
                handles.clear();
            }
        }
        for (uint32_t h : handles) (void) r.removeView(h);

        const double a = nWithout ? sumWithout / nWithout : 0.0;
        const double b = nWith ? sumWith / nWith : 0.0;
        std::printf("\nbench (vsync off, interleaved A/B in one run, %dx%d primary)\n", kW, kH);
        std::printf("  0 secondary views : %.3f ms/frame  (n=%d)\n", a, nWithout);
        std::printf("  3 secondary views : %.3f ms/frame  (n=%d)   [320x200, 640x400, 640x400]\n", b, nWith);
        std::printf("  delta             : %+.3f ms  (%.2fx)\n", b - a, a > 0 ? b / a : 0.0);
        std::printf("  scene work (acceleration structures, deformers, lights, materials)\n"
                    "  is built once and shared; what a view costs is its own pixels.\n");
    }

    // ── Opt-in measurement: marginal cost of ONE view, per resolution ───────
    // The headline 0-vs-3 number mixes three resolutions together. This walks one
    // resolution at a time, still interleaved 0-vs-1 within the same run, so
    // "what does another camera cost me at WxH" is measured, not extrapolated.
    void sweepViews(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& primaryCam) {
        struct Res {
            int w, h;
        };
        static const Res kRes[] = {{320, 200}, {640, 400}, {960, 600}, {1280, 800}};
        constexpr int kBlock = 50, kBlocks = 6, kWarm = 80;
        auto secCam = makeCam(Vector3(4.2f, 2.6f, 4.6f), Vector3(0.f, 1.5f, 0.f));

        std::printf("\nsweep (vsync off, interleaved 0-vs-1 view per resolution, %dx%d primary)\n", kW, kH);
        runFrames(canvas, r, scene, primaryCam, kWarm);
        auto tPrev = std::chrono::high_resolution_clock::now();

        for (const auto& res : kRes) {
            double sOn = 0.0, sOff = 0.0;
            int nOn = 0, nOff = 0;
            bool on = false;
            uint32_t h = 0;
            for (int block = 0; block < kBlocks * 2; ++block) {
                for (int i = 0; i < kBlock; ++i) {
                    canvas.animateOnce([&] { r.render(scene, primaryCam); });
                    const auto now = std::chrono::high_resolution_clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(now - tPrev).count();
                    tPrev = now;
                    if (i < 8) continue;
                    if (on) { sOn += ms; ++nOn; }
                    else { sOff += ms; ++nOff; }
                }
                on = !on;
                if (on) h = r.addView(*secCam, res.w, res.h);
                else { (void) r.removeView(h); h = 0; }
            }
            if (h) (void) r.removeView(h);
            const double a = nOff ? sOff / nOff : 0.0, b = nOn ? sOn / nOn : 0.0;
            const double mpx = double(res.w) * res.h / 1.0e6;
            std::printf("  %4dx%-4d : +%6.3f ms  (%.3f ms/Mpx)   base %.3f -> %.3f\n",
                        res.w, res.h, b - a, mpx > 0 ? (b - a) / mpx : 0.0, a, b);
        }
    }

}// namespace

int main(int argc, char** argv) {
    bool bench = false, sweep = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--sweep") == 0) sweep = true;
        else if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotDir = argv[++i];
    }

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters()
                        .title("Vulkan - Multi-View")
                        .size(kW, kH)
                        .vsync(false)
                        .headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    pinToTaaResolve(renderer);

    Scene scene;
    buildScene(scene);
    auto primaryCam = makeCam(Vector3(0.f, 3.2f, 8.5f), Vector3(0.f, 1.5f, 0.f));

    if (bench || sweep) {
        if (bench) benchViews(canvas, renderer, scene, *primaryCam);
        if (sweep) sweepViews(canvas, renderer, scene, *primaryCam);
        return 0;
    }

    gateParity(canvas, renderer, scene, *primaryCam);
    gateHistory(canvas, renderer, scene, *primaryCam);
    gateLifecycle(canvas, renderer, scene, *primaryCam);
    gateCullIsolation(canvas, renderer, scene, *primaryCam);
    gateAov(canvas, renderer, scene, *primaryCam);
    gateMsaa(canvas, renderer, scene, *primaryCam);
    gateSplats(canvas, renderer, scene, *primaryCam);

    std::printf("\nmulti-view: %d failed\n", failures);
    return failures == 0 ? 0 : 1;
}
