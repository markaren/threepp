// VulkanValidation_test — the Vulkan backend's cross-hardware CI gate.
//
// Every other Vulkan test in this directory asserts something about PIXELS, so
// none of them can gate CI: their references were captured on one GPU, and the
// stochastic GI/reflection samples behind them are not reproducible across
// drivers even in principle. The consequence was that CI compiled the largest
// and most actively developed subsystem in the repo and ran none of it.
//
// This test asserts something a conformant driver must agree on regardless of
// vendor: that a representative frame sequence provokes NO validation errors.
// That makes a software rasteriser (Mesa lavapipe, which CI has) a legitimate
// judge, and it happens to target the defect class that has actually recurred
// here — descriptor sets rewritten while a frame that reads them is still in
// flight (VUID-...-03047 and the stale-handle 02996), which is invisible to a
// pixel comparison, intermittent on real hardware, and reported precisely and
// deterministically by the layer.
//
// So the phases below are not "a scene that looks like something". Each one is a
// RECONFIGURATION — the operations that drain the device, reallocate images and
// rewrite descriptors, i.e. where that defect class lives. A phase that renders
// happily and reports one error has done its job.
//
// Plain exit-code program (not Catch2), matching the other tests here:
//   0   clean
//   1   validation errors were reported (they are printed above the summary)
//   42  CTest "Skipped" — no Vulkan/RT device, or no validation layer, which
//       would make every assertion below vacuously true
//
// Run standalone or via `ctest -R VulkanValidation_test`. On a software
// rasteriser expect this to take a while: it is ~70 frames of ray-traced
// deferred shading with no GPU behind it.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // Small on purpose. Every frame here is ray-traced deferred shading, and on
    // lavapipe that is CPU work — the gate is about which Vulkan calls are made,
    // not how many pixels they cover, so resolution buys nothing and costs
    // linearly.
    constexpr int kW = 256, kH = 160;
    constexpr int kSkipCode = 42;

    // Enough frames to get past the first-frame allocation path and have a
    // temporal history for a reconfiguration to invalidate — the interesting
    // state for this test is "a frame is in flight", which needs at least the
    // frames-in-flight depth to be real.
    constexpr int kWarmFrames  = 8;
    constexpr int kPhaseFrames = 5;

    void runFrames(Canvas& canvas, VulkanRenderer& r, Scene& scene, Camera& cam,
                   int n, const std::function<void(int)>& perFrame = {}) {
        for (int i = 0; i < n; ++i) {
            canvas.animateOnce([&] {
                r.render(scene, cam);
                if (perFrame) perFrame(i);
            });
        }
    }

    // Per-phase bookkeeping. The running total is what fails the test; the
    // per-phase delta is what makes a failure diagnosable, since it names the
    // reconfiguration that provoked the error rather than leaving a reader to
    // correlate layer output against a frame counter by hand.
    struct PhaseTally {
        std::string name;
        std::uint32_t errors;
        std::uint32_t warnings;
    };

    std::vector<PhaseTally> tallies;
    std::uint32_t seenErrors = 0, seenWarnings = 0;

    void endPhase(const char* name) {
        const auto e = vulkan::validationErrorCount();
        const auto w = vulkan::validationWarningCount();
        tallies.push_back({name, e - seenErrors, w - seenWarnings});
        std::printf("[phase] %-28s errors +%u  warnings +%u\n",
                    name, e - seenErrors, w - seenWarnings);
        seenErrors = e;
        seenWarnings = w;
    }

    // A scene wide enough to put several pipelines in the frame — opaque
    // standard, a transmissive dielectric (the glass retrace pass), a
    // shadow-casting directional light (RT shadow rays), and a line overlay
    // (OverlayPass, whose geometry cache has had lifetime bugs of its own).
    // Deliberately no HDR environment: PMREM would add an asset dependency and
    // real cost, and the default env image already exercises the sampling path.
    //
    // THREEPP_GATE_UNLIT=1 swaps every mesh to opaque MeshBasicMaterial. It
    // exists for lavapipe, where shading a LIT pixel — which traces rays into a
    // non-empty TLAS — segfaults inside llvmpipe's JIT-compiled shader code
    // (Mesa 24.x and 25.2.8, first frame, no validation message precedes it).
    // Bisected to exactly that: an unlit scene runs every phase below clean,
    // toggling off AO / probe GI / ReSTIR / denoise / shadows / lights one at a
    // time changes nothing, and forcing the no-ray-query fallback crashes the
    // same way via the RT pipeline. The gate's quarry is API-level lifecycle
    // errors — descriptor updates, reallocation, swapchain churn — and every
    // phase provokes those identically with Basic materials, so unlit mode
    // keeps the gate's value while the lit path stays a known llvmpipe gap.
    // Real-GPU runs (the default) keep the full lit scene.
    struct World {
        std::shared_ptr<Mesh> floor, box, glass;
        std::shared_ptr<AxesHelper> axes;
    };

    bool unlitMode() {
        const char* env = std::getenv("THREEPP_GATE_UNLIT");
        return env && *env && *env != '0';
    }

    // Opaque on purpose in unlit mode: a TRANSPARENT flat-color basic mesh is
    // routed to the raster overlay pass and leaves the traced scene entirely,
    // which would quietly drain the TLAS this test means to keep populated.
    std::shared_ptr<Material> sceneMaterial(const Color& color, float roughness, float metalness) {
        if (unlitMode()) {
            return MeshBasicMaterial::create({{"color", color}});
        }
        return MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(color).roughness(roughness).metalness(metalness));
    }

    World buildScene(Scene& scene) {
        World w;

        w.floor = Mesh::create(PlaneGeometry::create(20.f, 20.f),
                               sceneMaterial(Color(0.55f, 0.56f, 0.58f), 0.8f, 0.f));
        w.floor->rotation.x = -math::PI / 2.f;
        w.floor->receiveShadow = true;
        scene.add(w.floor);

        w.box = Mesh::create(BoxGeometry::create(1.2f, 1.2f, 1.2f),
                             sceneMaterial(Color(0.9f, 0.35f, 0.2f), 0.3f, 0.1f));
        w.box->position.set(-1.1f, 0.6f, 0.f);
        w.box->castShadow = true;
        scene.add(w.box);

        w.glass = Mesh::create(
                SphereGeometry::create(0.8f, 32, 16),
                unlitMode() ? std::static_pointer_cast<Material>(
                                      MeshBasicMaterial::create({{"color", Color(0.95f, 0.97f, 1.f)}}))
                            : std::static_pointer_cast<Material>(MeshPhysicalMaterial::create(
                                      MeshPhysicalMaterial::Params{}
                                              .color(Color(0.95f, 0.97f, 1.f))
                                              .roughness(0.05f)
                                              .transmission(0.9f)
                                              .thickness(0.6f))));
        w.glass->position.set(1.1f, 0.8f, 0.f);
        scene.add(w.glass);

        w.axes = AxesHelper::create(1.5f);
        scene.add(w.axes);

        auto sun = DirectionalLight::create(0xffffff, 2.2f);
        sun->position.set(4.f, 6.f, 3.f);
        sun->castShadow = true;
        scene.add(sun);
        scene.add(AmbientLight::create(0x404050, 1.f));

        return w;
    }

}// namespace

int main() {
    // Unbuffered stdout. Under ctest, stdout is a pipe and therefore fully
    // buffered — a segfault discards everything printf'd since the last flush,
    // which on the first lavapipe run meant a crash after 493 s reported ZERO
    // phase lines and looked like a startup crash. The stderr interleaving this
    // costs is nothing next to knowing which phase died.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // Force the layer on from inside the test rather than relying on the build
    // type or on the caller's environment. Release builds default validation
    // OFF, which is the whole reason spec violations went unnoticed there — a
    // gate that only works in a Debug build is a gate nobody runs. Must happen
    // before the Canvas, which is what constructs the instance.
#ifdef _WIN32
    _putenv_s("THREEPP_VULKAN_VALIDATION", "1");
#else
    setenv("THREEPP_VULKAN_VALIDATION", "1", /*overwrite*/ 1);
#endif

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanValidation_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    // Without the layer every assertion below is vacuously true, so this is a
    // skip, not a pass. A green run of this test must MEAN something.
    if (!vulkan::validationActive()) {
        std::printf("[skip] VK_LAYER_KHRONOS_validation is not installed — nothing to assert.\n"
                    "       Install the Vulkan validation layers (vulkan-validationlayers, or the\n"
                    "       LunarG SDK) and re-run.\n");
        return kSkipCode;
    }

    // Pin the configuration the gate runs in, for the same reason the goldens
    // do: DLSS/FSR are optional third-party upscalers whose own SPIR-V is not
    // ours to keep spec-clean (the messenger already filters one known FSR
    // report), and leaving them on would make this test's verdict depend on
    // which SDK happened to be built in.
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setDenoise(true);
    renderer.setRenderScale(1.0f);
    renderer.setClearColor(Color(0.05f, 0.06f, 0.08f));

    Scene scene;
    auto world = buildScene(scene);

    auto camera = PerspectiveCamera::create(55.f, float(kW) / float(kH), 0.1f, 100.f);
    camera->position.set(0.f, 2.2f, 5.f);
    camera->lookAt(Vector3(0.f, 0.8f, 0.f));

    std::printf("=== VulkanValidation_test: %dx%d, layer active, %s scene ===\n", kW, kH,
                unlitMode() ? "UNLIT (THREEPP_GATE_UNLIT)" : "lit");

    // Instance / device / swapchain creation happened above, before any phase
    // could bracket it. Close it as its own phase so that a violation there is
    // attributed to it rather than showing up under the first frame loop — and,
    // more importantly, so it is counted at all: it is the one stretch no later
    // phase can re-provoke.
    endPhase("context creation");

    // ── Phase 1: steady state ───────────────────────────────────────────────
    // The control. If this phase alone reports errors, no reconfiguration is
    // implicated and the plain frame path is at fault.
    runFrames(canvas, renderer, scene, *camera, kWarmFrames, [&](int) {
        world.box->rotation.y += 0.05f;
    });
    endPhase("steady state");

    // ── Phase 2: G-buffer AOV readback ──────────────────────────────────────
    // The debug-resolve path that has produced VUID-...-03047 three separate
    // times: reading an attachment rewrites a descriptor that a frame in flight
    // may still be reading. Every AOV, so no single attachment's resolve is left
    // uncovered.
    {
        const VulkanRenderer::GBufferAOV aovs[] = {
                VulkanRenderer::GBufferAOV::Depth,
                VulkanRenderer::GBufferAOV::Normal,
                VulkanRenderer::GBufferAOV::Motion,
                VulkanRenderer::GBufferAOV::Ids,
                VulkanRenderer::GBufferAOV::Albedo,
        };
        runFrames(canvas, renderer, scene, *camera, kPhaseFrames, [&](int i) {
            std::vector<std::uint8_t> out;
            int w = 0, h = 0, bpp = 0;
            const auto aov = aovs[std::size_t(i) % (sizeof(aovs) / sizeof(aovs[0]))];
            if (!renderer.readGBufferAOV(aov, out, w, h, bpp)) {
                std::printf("[warn] readGBufferAOV returned false on frame %d "
                            "(not a validation failure; the AOV was not available)\n",
                            i);
            }
        });
        // Also read every AOV within a single frame, which is the shape a sensor
        // rig actually uses and puts several resolves between one pair of
        // submissions rather than one per frame.
        runFrames(canvas, renderer, scene, *camera, 2, [&](int) {
            for (const auto aov : aovs) {
                std::vector<std::uint8_t> out;
                int w = 0, h = 0, bpp = 0;
                (void) renderer.readGBufferAOV(aov, out, w, h, bpp);
            }
        });
    }
    endPhase("gbuffer AOV readback");

    // ── Phase 3: secondary-view lifecycle ───────────────────────────────────
    // addView / removeView drain the device and rebuild the per-view entry list.
    // Mid-list removal is the documented hazard (it invalidates temporal history
    // and, before pooling, recycled entry pointers), so add two and remove the
    // FIRST — removing the last would miss it.
    {
        auto camB = PerspectiveCamera::create(50.f, float(kW) / float(kH), 0.1f, 100.f);
        camB->position.set(3.f, 2.f, 3.f);
        camB->lookAt(Vector3(0.f, 0.8f, 0.f));
        auto camC = PerspectiveCamera::create(50.f, float(kW) / float(kH), 0.1f, 100.f);
        camC->position.set(-3.f, 2.5f, -2.f);
        camC->lookAt(Vector3(0.f, 0.8f, 0.f));

        const auto viewB = renderer.addView(*camB, kW / 2, kH / 2);
        const auto viewC = renderer.addView(*camC, kW / 2, kH / 2);
        runFrames(canvas, renderer, scene, *camera, kPhaseFrames, [&](int) {
            (void) renderer.readViewRGBPixels(viewB);
        });
        // Mid-list removal, then more frames so the rebuilt list is actually
        // recorded and submitted rather than merely constructed.
        renderer.removeView(viewB);
        runFrames(canvas, renderer, scene, *camera, kPhaseFrames, [&](int) {
            (void) renderer.readViewRGBPixels(viewC);
        });
        renderer.removeView(viewC);
        runFrames(canvas, renderer, scene, *camera, 3);
    }
    endPhase("secondary view lifecycle");

    // ── Phase 4: render-scale realloc ───────────────────────────────────────
    // Changes the render extent, so every extent-sized image and every
    // descriptor pointing at one is reallocated. Called mid-run (not at setup)
    // because the deferred-apply queue that handles a mid-frame call is itself
    // the thing worth exercising.
    renderer.setRenderScale(0.6f);
    runFrames(canvas, renderer, scene, *camera, kPhaseFrames);
    renderer.setRenderScale(1.0f);
    runFrames(canvas, renderer, scene, *camera, kPhaseFrames);
    endPhase("render-scale realloc");

    // ── Phase 5: G-buffer MSAA switch ───────────────────────────────────────
    // Rebuilds the render pass, the multisample images and every pipeline bound
    // to them — the widest reallocation the renderer performs, and it goes both
    // ways so the tear-down direction is covered too.
    renderer.setGbufferMsaa(2);
    runFrames(canvas, renderer, scene, *camera, kPhaseFrames);
    renderer.setGbufferMsaa(1);
    runFrames(canvas, renderer, scene, *camera, 3);
    endPhase("gbuffer MSAA switch");

    // ── Phase 6: swapchain recreate ─────────────────────────────────────────
    // setSize retires the swapchain and its views while previously submitted
    // frames may still reference them. On a headless surface there is no window
    // manager involved, which is exactly why it is worth checking here: the
    // resize path normally only ever runs behind a real window.
    renderer.setSize({kW * 3 / 4, kH * 3 / 4});
    runFrames(canvas, renderer, scene, *camera, kPhaseFrames);
    renderer.setSize({kW, kH});
    runFrames(canvas, renderer, scene, *camera, 3);
    endPhase("swapchain recreate");

    // ── Phase 7: scene topology churn ───────────────────────────────────────
    // Adding and removing drawables rebuilds the entry list and the
    // acceleration structures mid-flight. Removal is the direction that frees
    // GPU-side records, so it is the one that can leave a stale handle behind.
    {
        auto extra = Mesh::create(BoxGeometry::create(0.5f, 0.5f, 0.5f),
                                  sceneMaterial(Color(0.2f, 0.8f, 0.4f), 0.5f, 0.f));
        extra->position.set(0.f, 0.3f, 2.f);
        extra->castShadow = true;
        scene.add(extra);
        runFrames(canvas, renderer, scene, *camera, 3);
        scene.remove(*extra);
        runFrames(canvas, renderer, scene, *camera, 3);

        // Add-after-remove: the sequence that used to fail-fast 0xC0000409
        // (VulkanCameraModel_test structured its whole scene around avoiding
        // it). The historical shape had a LINE object (an overlay entry, which
        // holds an entry slot but pushes no TLAS instance) as the thing
        // removed, and a traced triangle mesh as the thing added after. Fixed
        // by later entry-list work; pinned here so it stays fixed.
        {
            const std::vector<float> verts = {-0.3f, 0.f, 0.f, 0.3f, 0.f, 0.f,
                                              0.f, -0.3f, 0.f, 0.f, 0.3f, 0.f};
            auto lineGeo = BufferGeometry::create();
            lineGeo->setAttribute("position", FloatBufferAttribute::create(verts, 3));
            auto lineMat = LineBasicMaterial::create();
            lineMat->color = Color(0.f, 1.f, 0.f);
            auto cross = LineSegments::create(lineGeo, lineMat);
            cross->position.set(0.5f, 1.2f, 1.f);
            scene.add(cross);
            runFrames(canvas, renderer, scene, *camera, 3);
            scene.remove(*cross);
            runFrames(canvas, renderer, scene, *camera, 3);
        }
        auto extra2 = Mesh::create(SphereGeometry::create(0.4f, 16, 8),
                                   sceneMaterial(Color(0.9f, 0.7f, 0.1f), 0.4f, 0.f));
        extra2->position.set(-0.5f, 0.4f, 2.2f);
        scene.add(extra2);
        runFrames(canvas, renderer, scene, *camera, 3);

        // A material-side change too: the per-entry GPU patch path rather than
        // the whole-list rebuild.
        if (unlitMode()) {
            world.box->material()->as<MeshBasicMaterial>()->color.setRGB(0.2f, 0.4f, 0.9f);
        } else {
            world.box->material()->as<MeshStandardMaterial>()->color.setRGB(0.2f, 0.4f, 0.9f);
        }
        world.box->material()->needsUpdate();
        runFrames(canvas, renderer, scene, *camera, 3);
    }
    endPhase("scene topology churn");

    // ── Phase 8: instanced-span churn ───────────────────────────────────────
    // Instanced spans are the only geometry the GPU instance-expansion pass
    // (instance_expand.comp) touches, so without one in the scene this gate
    // covers that pass by not running it. Everything the pass does that this
    // test exists to catch happens here: per-frame host writes into a
    // frame-in-flight-owned mapped pool, a descriptor set rewritten between the
    // fence wait and recording, and — when a second span pushes the pools past
    // their capacity mid-run — a reallocation that hands the old buffer to the
    // retire queue while the sibling frame may still name it.
    {
        auto field = InstancedMesh::create(BoxGeometry::create(0.15f, 0.15f, 0.15f),
                                           sceneMaterial(Color(0.3f, 0.6f, 0.9f), 0.5f, 0.f), 512);
        {
            Matrix4 m;
            for (size_t i = 0; i < field->count(); ++i) {
                const float t = static_cast<float>(i) * 0.05f;
                m.setPosition(std::sin(t) * 3.f, 0.2f + 0.02f * static_cast<float>(i % 20),
                              std::cos(t) * 3.f);
                field->setMatrixAt(i, m);
            }
            field->instanceMatrix()->needsUpdate();
        }
        scene.add(field);
        runFrames(canvas, renderer, scene, *camera, kPhaseFrames);

        // Per-frame instance edits: the version-gated upload path, hit on every
        // frame and therefore on both frames-in-flight.
        for (int f = 0; f < kPhaseFrames; ++f) {
            Matrix4 m;
            for (size_t i = 0; i < field->count(); ++i) {
                const float t = static_cast<float>(i) * 0.05f + static_cast<float>(f) * 0.1f;
                m.setPosition(std::sin(t) * 3.f, 0.2f, std::cos(t) * 3.f);
                field->setMatrixAt(i, m);
            }
            field->instanceMatrix()->needsUpdate();
            runFrames(canvas, renderer, scene, *camera, 1);
        }

        // A second, much larger span mid-run: grows the per-frame matrix pool
        // AND the shared world-matrix buffer while frames are in flight.
        auto field2 = InstancedMesh::create(SphereGeometry::create(0.1f, 8, 4),
                                            sceneMaterial(Color(0.9f, 0.5f, 0.2f), 0.6f, 0.f), 6000);
        {
            Matrix4 m;
            for (size_t i = 0; i < field2->count(); ++i) {
                const float t = static_cast<float>(i) * 0.011f;
                m.setPosition(std::sin(t) * 5.f, 1.5f + std::sin(t * 3.f), std::cos(t) * 5.f);
                field2->setMatrixAt(i, m);
            }
            field2->instanceMatrix()->needsUpdate();
        }
        scene.add(field2);
        runFrames(canvas, renderer, scene, *camera, kPhaseFrames);

        // Shrink the live count, then remove both — the direction that frees
        // records and re-stamps the per-slot upload bookkeeping.
        field2->setCount(1000);
        runFrames(canvas, renderer, scene, *camera, 3);
        scene.remove(*field2);
        runFrames(canvas, renderer, scene, *camera, 3);
        scene.remove(*field);
        runFrames(canvas, renderer, scene, *camera, 3);
    }
    endPhase("instanced-span churn");

    // ── Verdict ─────────────────────────────────────────────────────────────
    const auto errors   = vulkan::validationErrorCount();
    const auto warnings = vulkan::validationWarningCount();

    std::printf("\n=== summary ===\n");
    for (const auto& t : tallies) {
        std::printf("  %-28s errors %u  warnings %u\n", t.name.c_str(), t.errors, t.warnings);
    }
    std::printf("  %-28s errors %u  warnings %u\n", "TOTAL", errors, warnings);

    if (errors != 0) {
        std::printf("\nFAILED: the validation layer reported %u error(s). The messages are above,\n"
                    "        each prefixed \"[Vulkan] ERROR:\"; the phase table names which\n"
                    "        reconfiguration provoked them.\n",
                    errors);
        return 1;
    }
    std::printf("\nPASSED: no validation errors across %zu phases.\n", tallies.size());
    return 0;
}
