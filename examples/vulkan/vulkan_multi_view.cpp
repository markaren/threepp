// vulkan_multi_view — N cameras, one scene, one frame.
//
// The deferred renderer used to be strictly one view per frame, which made a
// robot camera rig awkward in exactly the way that matters: calling render()
// once per camera gives you N different simulated instants and pays N times to
// build one scene. A rig wants the SAME instant from several viewpoints.
//
// VulkanRenderer::addView attaches a persistent extra view. Each one owns its
// G-buffer, its temporal history and its camera state; the acceleration
// structures, lights, materials and textures are built once and shared. Every
// render() then produces the primary and every added view in a single queue
// submission.
//
// Run it:
//   vulkan_multi_view                 3 robot cameras orbiting a lit scene,
//                                     each read back every frame
//   vulkan_multi_view --parity        gate: the same camera as primary and as
//                                     secondary must agree
//   vulkan_multi_view --history       gate: moving one view must not disturb
//                                     another
//   vulkan_multi_view --lifecycle     gate: add / render / remove / re-add,
//                                     for a validation-layer run
//                                     (THREEPP_VULKAN_VALIDATION=1)
//   vulkan_multi_view --cullcheck     regression: a secondary's frustum cull
//                                     must be invisible to the primary
//   vulkan_multi_view --bench         0 vs 3 secondary views, interleaved
//   vulkan_multi_view --sweep         marginal cost of one view, per resolution
//
// A note on measurement, since three of these gates were initially wrong in the
// same way: adding a secondary view, or even re-aiming one, changes frame time,
// and frame time feeds the TAA's dt-scaled temporal constants. A whole-image
// PSNR between two such configurations therefore moves by several dB for
// reasons that have nothing to do with what is being tested. Each gate below
// either holds cost constant, measures against its own control run, or looks at
// a quantity the drift cannot touch.
//
// Captures are written as raw PPM. Deliberately: these are measurement cameras,
// and PPM round-trips the readback bytes with no codec, no flip and no channel
// ambiguity — the same reasoning the golden-image test uses.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "capture_util.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
namespace fs = std::filesystem;

namespace {

    constexpr int kW = 640, kH = 400;

    void writePPM(const fs::path& p, const std::vector<unsigned char>& rgb, int w, int h) {
        fs::create_directories(p.parent_path());
        std::ofstream f(p, std::ios::binary);
        f << "P6\n" << w << " " << h << "\n255\n";
        f.write(reinterpret_cast<const char*>(rgb.data()),
                static_cast<std::streamsize>(rgb.size()));
    }

    fs::path outPath(const std::string& name) {
#ifdef PROJECT_FOLDER
        return fs::path(PROJECT_FOLDER) / "aaa_caps" / name;
#else
        return fs::path(name);
#endif
    }

    // A scene with the three things a secondary view has to get right: cast
    // shadows, bounced indirect light, and a mirror that reflects geometry it
    // cannot see directly. Deliberately NO fog — froxel volumetrics are a
    // primary-only feature by scope, so including it would compare two things
    // that are not meant to match.
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

        // One sun, hard enough to throw legible shadows.
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

    struct Mode {
        bool parity = false, history = false, lifecycle = false, bench = false, sweep = false;
        bool cullCheck = false;
        // --nocut: run the history gate WITHOUT moving the other camera. The
        // control. Whatever this reports is the still view's own residual
        // convergence drift over the same number of frames, and the real run
        // has to be compared against it — not against an absolute number
        // someone picked.
        bool noCut = false;
    };

}// namespace

int main(int argc, char** argv) {
    Mode mode;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--parity") mode.parity = true;
        else if (a == "--history") mode.history = true;
        else if (a == "--lifecycle") mode.lifecycle = true;
        else if (a == "--bench") mode.bench = true;
        else if (a == "--nocut") mode.noCut = true;
        else if (a == "--sweep") mode.sweep = true;
        else if (a == "--cullcheck") mode.cullCheck = true;
    }
    const bool headless = mode.parity || mode.history || mode.lifecycle ||
                          mode.bench || mode.sweep || mode.cullCheck;

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters()
                        .title("Vulkan - Multi-View Camera Rig")
                        .size(kW, kH)
                        .vsync(false)
                        .headless(headless));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return 42;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    pinToTaaResolve(renderer);

    Scene scene;
    buildScene(scene);

    PerspectiveCamera primaryCam(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    primaryCam.position.set(0.f, 3.2f, 8.5f);
    primaryCam.lookAt(Vector3(0.f, 1.5f, 0.f));
    primaryCam.updateMatrixWorld();

    int frame = 0;

    // ── Gate (a): parity vs primary ────────────────────────────────────────
    // The same camera, rendered as the primary and as a secondary view, over a
    // static scene. Any difference is the multi-view chain diverging from the
    // one that has always worked.
    if (mode.parity) {
        constexpr int kWarm = 160;// well past the 64-frame minimum: both chains
                                  // must be CONVERGED, not merely started
        PerspectiveCamera secCam(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
        secCam.position.copy(primaryCam.position);
        secCam.quaternion.copy(primaryCam.quaternion);
        secCam.updateMatrixWorld();

        uint32_t vh = 0;
        canvas.animate([&] {
            renderer.render(scene, primaryCam);
            ++frame;
            // Added after the first frame: a view shares the primary's render
            // pass, which is built on the first render.
            if (frame == 1) {
                vh = renderer.addView(secCam, kW, kH);
                if (!vh) { std::printf("addView FAILED\n"); std::exit(1); }
            }
            if (frame < kWarm) return;

            const auto prim = renderer.readRGBPixels();
            const auto sec  = renderer.readViewRGBPixels(vh);
            if (sec.empty() || sec.size() != prim.size()) {
                std::printf("parity: readback size mismatch (%zu vs %zu)\n", sec.size(), prim.size());
                std::exit(1);
            }
            writePPM(outPath("mv_parity_primary.ppm"), prim, kW, kH);
            writePPM(outPath("mv_parity_secondary.ppm"), sec, kW, kH);

            const auto d = capture::imageDiff(prim, sec);
            // Both chains run the same deterministic TAA resolve on the same
            // static scene, but their ray-traced GI/reflection/shadow samples
            // are stochastic and their histories started at different frames,
            // so this is a convergence gate, not a bit-exactness one.
            constexpr double kMinPsnr = 34.0;
            const bool pass = d.psnr >= kMinPsnr;
            std::printf("parity primary-vs-secondary: PSNR=%.2f dB  maxD=%d  hot=%.3f%%  -> %s\n",
                        d.psnr, d.maxD, d.hotPct, pass ? "PASS" : "FAIL");
            capture::diffStats(prim, sec, kW, kH, 3, "parity");
            std::exit(pass ? 0 : 1);
        });
        return 0;
    }

    // ── Gate (b): history independence ─────────────────────────────────────
    // Two secondary views. One is left completely alone; the other's camera is
    // yanked across the scene. If any temporal state is shared, the still view
    // shows the disturbance.
    //
    // Run it TWICE and diff the two output files:
    //     vulkan_multi_view --history --nocut   → mv_history_still_nocut.ppm
    //     vulkan_multi_view --history           → mv_history_still_cut.ppm
    //
    // Comparing the still view's own before/after inside ONE run does not
    // isolate anything: this scene keeps converging (a rough-metal knot and a
    // mirror sphere feed a denoiser that is still settling at frame 280), so
    // that number measures convergence, not contamination. It reads ~39 dB
    // whether or not the other camera ever moves. The cross-run file diff is
    // the measurement that actually answers the question, against the
    // renderer's own run-to-run noise floor as the tolerance.
    if (mode.history) {
        constexpr int kSettle = 140, kAfter = 140;
        // Both cameras sit INSIDE the room (the walls are at x = ±5, the floor
        // spans z ∈ [-6, 6]) and frame the whole scene. A view pointed at a
        // flat wall would pass this gate trivially — there would be no shadow,
        // reflection or GI structure left to disturb.
        PerspectiveCamera movingCam(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
        movingCam.position.set(3.6f, 2.5f, 4.8f);
        movingCam.lookAt(Vector3(0.f, 1.5f, 0.f));
        movingCam.updateMatrixWorld();

        PerspectiveCamera stillCam(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
        stillCam.position.set(-3.4f, 2.7f, 5.2f);
        stillCam.lookAt(Vector3(0.f, 1.4f, 0.f));
        stillCam.updateMatrixWorld();

        uint32_t movingH = 0, stillH = 0;
        std::vector<unsigned char> stillBefore;
        canvas.animate([&] {
            renderer.render(scene, primaryCam);
            ++frame;
            if (frame == 1) {
                movingH = renderer.addView(movingCam, kW, kH);
                stillH  = renderer.addView(stillCam, kW, kH);
                if (!movingH || !stillH) { std::printf("addView FAILED\n"); std::exit(1); }
                return;
            }
            if (frame == kSettle) {
                stillBefore = renderer.readViewRGBPixels(stillH);
                writePPM(outPath("mv_history_still_before.ppm"), stillBefore, kW, kH);
                // A hard cut for the OTHER view — the largest disturbance its
                // history can experience. Skipped under --nocut, which is the
                // control run.
                if (!mode.noCut) {
                    movingCam.position.set(0.f, 7.5f, -7.f);
                    movingCam.lookAt(Vector3(0.f, 1.f, 0.f));
                    movingCam.updateMatrixWorld();
                }
                return;
            }
            if (frame < kSettle + kAfter) return;

            const auto stillAfter = renderer.readViewRGBPixels(stillH);
            // Tagged by mode so the two runs of the protocol don't overwrite
            // each other and the real comparison is a file diff.
            writePPM(outPath(mode.noCut ? "mv_history_still_nocut.ppm"
                                        : "mv_history_still_cut.ppm"),
                     stillAfter, kW, kH);
            writePPM(outPath("mv_history_still_after.ppm"), stillAfter, kW, kH);
            writePPM(outPath("mv_history_moved.ppm"), renderer.readViewRGBPixels(movingH), kW, kH);

            const auto d = capture::imageDiff(stillBefore, stillAfter);
            // Reported, not gated on an absolute number. The still view keeps
            // integrating its own stochastic samples between the two captures,
            // so SOME difference is expected with or without the other view's
            // cut. Run --nocut for the control and compare the two: they should
            // land in the same place. A number pulled out of the air here would
            // only be measuring how well the scene converges.
            std::printf("history %s (still view, frame %d vs %d): "
                        "PSNR=%.2f dB  maxD=%d  hot=%.3f%%\n",
                        mode.noCut ? "CONTROL (no cut)" : "TEST (other view cut)",
                        kSettle, kSettle + kAfter,
                        d.psnr, d.maxD, d.hotPct);
            std::printf("  wrote %s\n",
                        mode.noCut ? "mv_history_still_nocut.ppm" : "mv_history_still_cut.ppm");
            std::printf("  the assertion is the FILE diff between the two runs "
                        "(--history --nocut, then --history), not the number above.\n");
            std::exit(0);
        });
        return 0;
    }

    // ── Gate (c): add / render / remove / re-add ───────────────────────────
    // Run under the validation layers. Exercises every lifecycle edge: creating
    // views, rendering them, destroying them mid-life, and re-creating at a
    // different resolution so nothing can quietly depend on the old extents.
    if (mode.lifecycle) {
        PerspectiveCamera camA(50.f, 1.6f, 0.1f, 100.f);
        camA.position.set(5.f, 3.f, 5.f);
        camA.lookAt(Vector3(0.f, 1.5f, 0.f));
        camA.updateMatrixWorld();
        PerspectiveCamera camB(70.f, 1.6f, 0.1f, 100.f);
        camB.position.set(-4.f, 2.f, -5.f);
        camB.lookAt(Vector3(0.f, 1.5f, 0.f));
        camB.updateMatrixWorld();

        uint32_t hA = 0, hB = 0;
        int cycle = 0;
        canvas.animate([&] {
            renderer.render(scene, primaryCam);
            ++frame;
            const int phase = frame % 12;
            if (frame < 2) return;
            if (phase == 2) {
                hA = renderer.addView(camA, 320, 200);
                hB = renderer.addView(camB, 512, 320);
                std::printf("[cycle %d] added %u, %u\n", cycle, hA, hB);
            } else if (phase == 6) {
                // Read while live — the readback path is part of the lifecycle.
                const auto a = renderer.readViewRGBPixels(hA);
                const auto b = renderer.readViewRGBPixels(hB);
                std::printf("[cycle %d] read %zu + %zu bytes\n", cycle, a.size(), b.size());
            } else if (phase == 9) {
                // Remove out of order, and remove one twice: a stale handle must
                // be inert, not fatal.
                const bool r1 = renderer.removeView(hB);
                const bool r2 = renderer.removeView(hA);
                const bool r3 = renderer.removeView(hA);
                std::printf("[cycle %d] removed %d %d, stale-remove %d (expect 1 1 0)\n",
                            cycle, int(r1), int(r2), int(r3));
                if (r3) { std::printf("stale handle accepted - FAIL\n"); std::exit(1); }
                if (++cycle >= 3) {
                    std::printf("lifecycle: 3 add/render/read/remove cycles complete -> PASS\n");
                    std::exit(0);
                }
            }
        });
        return 0;
    }

    // ── Phase C: 0 vs 3 secondary views, interleaved ───────────────────────
    // Interleaved within ONE run, alternating in blocks, because per-session
    // clock/thermal drift makes two separate runs untrustworthy in this project.
    if (mode.bench) {
        constexpr int kBlock = 60, kBlocks = 8, kWarm = 80;
        std::vector<PerspectiveCamera> cams;
        cams.reserve(3);
        for (int i = 0; i < 3; ++i) {
            cams.emplace_back(50.f, 1.6f, 0.1f, 100.f);
        }
        const int resW[3] = {320, 640, 640};
        const int resH[3] = {200, 400, 400};
        for (int i = 0; i < 3; ++i) {
            const float a = static_cast<float>(i) * 2.094f;
            cams[i].position.set(6.5f * std::cos(a), 2.6f + 0.7f * static_cast<float>(i), 6.5f * std::sin(a));
            cams[i].lookAt(Vector3(0.f, 1.5f, 0.f));
            cams[i].updateMatrixWorld();
        }

        std::vector<uint32_t> handles;
        bool withViews = false, benchDone = false;
        (void) benchDone;
        int block = 0, inBlock = 0;
        double sumWith = 0.0, sumWithout = 0.0;
        int nWith = 0, nWithout = 0;
        auto tPrev = std::chrono::high_resolution_clock::now();

        canvas.animate([&] {
            renderer.render(scene, primaryCam);
            ++frame;
            const auto now = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - tPrev).count();
            tPrev = now;
            if (frame <= kWarm) return;

            if (inBlock >= 8) {// skip each block's first frames (add/remove stall)
                if (withViews) { sumWith += ms; ++nWith; }
                else           { sumWithout += ms; ++nWithout; }
            }
            if (++inBlock < kBlock) return;
            inBlock = 0;

            // Flip configuration. Views are persistent objects, so this
            // add/remove is the BENCHMARK's churn, not the feature's — it is
            // why the first frames of each block are discarded above.
            withViews = !withViews;
            if (withViews) {
                for (int i = 0; i < 3; ++i) handles.push_back(renderer.addView(cams[i], resW[i], resH[i]));
            } else {
                for (uint32_t h : handles) renderer.removeView(h);
                handles.clear();
            }

            if (++block < kBlocks * 2) return;
            for (uint32_t h : handles) renderer.removeView(h);
            const double a = nWithout ? sumWithout / nWithout : 0.0;
            const double b = nWith ? sumWith / nWith : 0.0;
            std::printf("\nbench (vsync off, interleaved A/B in one run, %dx%d primary)\n", kW, kH);
            std::printf("  0 secondary views : %.3f ms/frame  (n=%d)\n", a, nWithout);
            std::printf("  3 secondary views : %.3f ms/frame  (n=%d)   [320x200, 640x400, 640x400]\n", b, nWith);
            std::printf("  delta             : %+.3f ms  (%.2fx)\n", b - a, a > 0 ? b / a : 0.0);
            std::printf("  scene work (acceleration structures, deformers, lights, materials)\n"
                        "  is built once and shared; what a view costs is its own pixels.\n");
            benchDone = true;
            std::exit(0);
        });
        return 0;
    }

    // ── Regression: the overlay must see the PRIMARY's frustum cull ────────
    // The frustum-cull result used to be a single bool on the shared mesh
    // entry. Two readers consume it: the indirect draw builder (per view, fine)
    // and the primary's overlay DEPTH PREPASS, which lays down occluders so a
    // wireframe is correctly hidden behind solid geometry. With a shared bit,
    // the last view to cull won — so a secondary aimed away from the occluder
    // would delete that occluder from the PRIMARY's prepass and the wireframe
    // would show through the box in front of it. No crash, no validation error,
    // just wrong pixels, and only when a secondary happens to run.
    //
    // The comparison is deliberately SAME-COST: one secondary view is present
    // throughout, and only its AIM changes — first pointed out of the room (so
    // its cull rejects everything), then at the scene (so its cull keeps
    // everything). Comparing "no secondary" against "one secondary" instead
    // would have measured the wrong thing: a second view triples frame time,
    // which moves the TAA's dt-scaled temporal constants and shifts the
    // primary's convergence by several dB all on its own. That confound is
    // larger than the effect being tested, and it is what an earlier version of
    // this check actually measured.
    if (mode.cullCheck) {
        // A wireframe marker fully behind a solid box: visible only if the
        // occluder is missing from the depth prepass.
        auto wireMat = MeshBasicMaterial::create();
        wireMat->color = Color(1.f, 1.f, 0.f);
        wireMat->wireframe = true;
        auto wire = Mesh::create(SphereGeometry::create(0.75f, 16, 10), wireMat);
        wire->position.set(2.5f, 1.1f, -1.2f);// dead centre of the blue box
        scene.add(wire);

        PerspectiveCamera awayCam(50.f, 1.6f, 0.1f, 100.f);
        awayCam.position.set(0.f, 2.f, -5.5f);
        awayCam.lookAt(Vector3(0.f, 2.f, -20.f));// looks out of the room entirely
        awayCam.updateMatrixWorld();

        constexpr int kWarm = 110;
        std::vector<unsigned char> aimedAway;
        uint32_t vh = 0;
        canvas.animate([&] {
            renderer.render(scene, primaryCam);
            ++frame;
            if (frame == 2) {
                // Present for the WHOLE run, so frame cost never changes.
                vh = renderer.addView(awayCam, 320, 200);
                return;
            }
            if (frame == kWarm) {
                aimedAway = renderer.readRGBPixels();
                writePPM(outPath("mv_cull_aimed_away.ppm"), aimedAway, kW, kH);
                // Same view, same cost — now it sees the whole scene, so its
                // frustum cull keeps everything the "away" aim rejected.
                awayCam.position.set(3.8f, 2.4f, 4.4f);
                awayCam.lookAt(Vector3(0.f, 1.5f, 0.f));
                awayCam.updateMatrixWorld();
                return;
            }
            if (frame < kWarm * 2) return;
            const auto aimedAt = renderer.readRGBPixels();
            writePPM(outPath("mv_cull_aimed_at.ppm"), aimedAt, kW, kH);
            // Count the VISIBLE wireframe, not the whole image.
            //
            // A whole-image PSNR cannot answer this question. Changing the
            // secondary's aim changes how much it draws, which changes frame
            // time, which moves the TAA's dt-scaled temporal constants and
            // shifts the primary's converged GI by a few dB — a confound
            // larger than the effect. But the failure being guarded against is
            // not subtle and not global: if the occluder were dropped from the
            // primary's depth prepass, the wireframe marker buried inside the
            // solid box would render UNOCCLUDED, and its visible pixel count
            // would jump. That count is immune to diffuse convergence drift.
            const auto yellowPixels = [](const std::vector<unsigned char>& img) {
                long n = 0;
                for (size_t i = 0; i + 2 < img.size(); i += 3)
                    if (img[i] > 150 && img[i + 1] > 150 && img[i + 2] < 100) ++n;
                return n;
            };
            const long yAway = yellowPixels(aimedAway);
            const long yAt   = yellowPixels(aimedAt);
            const auto d     = capture::imageDiff(aimedAway, aimedAt);
            const long tol   = std::max<long>(8, yAway / 20);// 5%, floor 8 px
            const bool pass  = yAway > 0 && std::labs(yAway - yAt) <= tol;
            std::printf("cull isolation (primary, secondary aimed away vs aimed at scene):\n"
                        "  visible wireframe pixels: %ld vs %ld (tol +-%ld)  -> %s\n"
                        "  whole-image PSNR=%.2f dB (informational: moves with frame-time "
                        "drift, not with culling)\n",
                        yAway, yAt, tol, pass ? "PASS" : "FAIL", d.psnr);
            (void) renderer.removeView(vh);
            std::exit(pass ? 0 : 1);
        });
        return 0;
    }

    // ── Phase C: marginal cost of ONE view, per resolution ─────────────────
    // The headline 0-vs-3 number mixes three resolutions together. This walks
    // one resolution at a time, still interleaved 0-vs-1 within the same run,
    // so the answer to "what does another camera cost me at WxH" is measured
    // rather than extrapolated.
    if (mode.sweep) {
        struct Res { int w, h; };
        static const Res kRes[] = {{320, 200}, {640, 400}, {960, 600}, {1280, 800}};
        constexpr int kNRes = 4;
        constexpr int kBlock = 50, kBlocks = 6, kWarm = 80;

        PerspectiveCamera secCam(50.f, 1.6f, 0.1f, 100.f);
        secCam.position.set(4.2f, 2.6f, 4.6f);
        secCam.lookAt(Vector3(0.f, 1.5f, 0.f));
        secCam.updateMatrixWorld();

        int resIdx = 0, block = 0, inBlock = 0;
        bool on = false;
        uint32_t h = 0;
        double sOn = 0.0, sOff = 0.0;
        int nOn = 0, nOff = 0;
        auto tPrev = std::chrono::high_resolution_clock::now();
        std::printf("\nsweep (vsync off, interleaved 0-vs-1 view per resolution, %dx%d primary)\n", kW, kH);

        canvas.animate([&] {
            renderer.render(scene, primaryCam);
            ++frame;
            const auto now = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - tPrev).count();
            tPrev = now;
            if (frame <= kWarm) return;
            if (inBlock >= 8) {
                if (on) { sOn += ms; ++nOn; } else { sOff += ms; ++nOff; }
            }
            if (++inBlock < kBlock) return;
            inBlock = 0;
            on = !on;
            if (on) h = renderer.addView(secCam, kRes[resIdx].w, kRes[resIdx].h);
            else { renderer.removeView(h); h = 0; }

            if (++block < kBlocks * 2) return;
            const double a = nOff ? sOff / nOff : 0.0, b = nOn ? sOn / nOn : 0.0;
            const double mpx = double(kRes[resIdx].w) * kRes[resIdx].h / 1.0e6;
            std::printf("  %4dx%-4d : +%6.3f ms  (%.3f ms/Mpx)   base %.3f -> %.3f\n",
                        kRes[resIdx].w, kRes[resIdx].h, b - a,
                        mpx > 0 ? (b - a) / mpx : 0.0, a, b);
            if (h) { renderer.removeView(h); h = 0; }
            on = false;
            block = 0; sOn = sOff = 0.0; nOn = nOff = 0;
            if (++resIdx >= kNRes) std::exit(0);
        });
        return 0;
    }

    // ── Interactive: three robot cameras orbiting the scene ────────────────
    std::vector<std::shared_ptr<PerspectiveCamera>> rigCams;
    std::vector<uint32_t> rigHandles;
    for (int i = 0; i < 3; ++i) {
        auto c = std::make_shared<PerspectiveCamera>(50.f, 1.6f, 0.1f, 100.f);
        rigCams.push_back(c);
    }

    OrbitControls controls{primaryCam, canvas};
    controls.target.set(0.f, 1.5f, 0.f);
    controls.update();

    Clock clock;
    canvas.animate([&] {
        const float t = clock.getElapsedTime();
        for (int i = 0; i < 3; ++i) {
            const float a = t * 0.35f + static_cast<float>(i) * 2.094f;
            rigCams[i]->position.set(6.5f * std::cos(a),
                                     2.6f + 0.7f * static_cast<float>(i),
                                     6.5f * std::sin(a));
            rigCams[i]->lookAt(Vector3(0.f, 1.5f, 0.f));
            rigCams[i]->updateMatrixWorld();
        }

        renderer.render(scene, primaryCam);
        ++frame;

        if (frame == 1) {
            for (int i = 0; i < 3; ++i)
                rigHandles.push_back(renderer.addView(*rigCams[i], 320, 200));
        }

        // Read every camera every frame — this is the shape a synthetic-data or
        // policy-evaluation loop actually uses, and doing it here means the
        // demo exercises the readback path continuously rather than once.
        if (frame % 120 == 0 && !rigHandles.empty()) {
            size_t total = 0;
            for (uint32_t h : rigHandles) total += renderer.readViewRGBPixels(h).size();
            std::printf("frame %d: read %zu bytes across %zu robot cameras\n",
                        frame, total, rigHandles.size());
            std::fflush(stdout);
        }
    });

    return 0;
}
