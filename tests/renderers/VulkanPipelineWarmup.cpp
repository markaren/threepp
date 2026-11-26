// VulkanPipelineWarmup: the setup step of every Vulkan test in this directory
// (the CTest fixture vulkan_pipelines). It builds the pipelines those tests
// share once, before any of them starts. After a shader edit each pipeline is
// a cold compile. Without this step every test compiled the deferred-shade
// permutations it reached on its render thread, and under ctest -j several
// tests compiled the same one at once: 5 to 6.5 minutes each on an RTX 4060
// laptop (2026-11-26). Here the seven ray-traced ones the suite uses took
// 324 s together, and the cold suite at -j4 took 464 s including that.
//
//   1. A renderer with ray tracing, where the device has it: its startup
//      pipelines, every deferred-shade permutation in this machine's shade
//      history, built on worker threads (VulkanRenderer::
//      prewarmShadePermutations), and the path tracer's pipelines, from a
//      one-sample trace of an emitter over a glass ball (the glass is what
//      builds the photon pipeline).
//   2. A renderer in the no-ray-tracing mode (THREEPP_VULKAN_NO_RT=1): its
//      startup pipelines and its own shade permutations.
//
// The history is the list of permutations the tests and anything else on this
// machine have built, so the first run after it appears warms nothing new;
// from then on a shader edit is paid here. Warm, the step costs the two
// renderers' startup, a few seconds. It checks nothing and exits 0 whatever
// happens: a failure here must not stop the tests that require it, which then
// compile what they need themselves.
//
// The first renderer runs with the pipeline cache's size cap at 192 MB
// instead of 256 (THREEPP_VK_PIPELINE_CACHE_CAP_MB). A cache over the cap is
// dropped by the process that loads it, and when that process was a test,
// the others behind it found an emptied cache and compiled every permutation
// again. The lower cap here makes this step the one that drops it, before it
// builds them; what it then adds stays under the cap the tests load with.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "vulkan_test_util.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 64, kH = 48;

    using SteadyClock = std::chrono::steady_clock;

    double secondsSince(SteadyClock::time_point t0) {
        return std::chrono::duration<double>(SteadyClock::now() - t0).count();
    }

    // An emissive panel over a glass ball over a floor: the path tracer builds
    // its photon pipeline for a scene with emitters and a transmissive
    // material, and its trace and resolve pipelines for any.
    void buildTraceScene(Scene& scene) {
        auto floorMat   = MeshStandardMaterial::create();
        floorMat->color = Color(0.8f, 0.8f, 0.8f);
        auto floor        = Mesh::create(PlaneGeometry::create(4.f, 4.f), floorMat);
        floor->rotation.x = -math::PI / 2;
        scene.add(floor);

        auto panelMat               = MeshStandardMaterial::create();
        panelMat->color             = Color(0.f, 0.f, 0.f);
        panelMat->emissive          = Color(1.f, 1.f, 1.f);
        panelMat->emissiveIntensity = 10.f;
        panelMat->side              = Side::Double;
        auto panel        = Mesh::create(PlaneGeometry::create(0.5f, 0.5f), panelMat);
        panel->rotation.x = math::PI / 2;
        panel->position.y = 2.f;
        scene.add(panel);

        auto glass          = MeshPhysicalMaterial::create();
        glass->transmission = 1.f;
        glass->roughness    = 0.f;
        glass->ior          = 1.5f;
        glass->thickness    = 0.5f;
        auto ball        = Mesh::create(SphereGeometry::create(0.3f, 24, 16), glass);
        ball->position.y = 0.6f;
        scene.add(ball);
    }

    // The renderer is returned alive: destroying one mid-process waits on the
    // driver for seconds (vkDestroyDevice), while at exit both are left to the
    // OS. Each phase saves the pipeline cache itself, so the exit has nothing
    // left to write: were the first renderer to save there, its copy of the
    // cache, without the second one's pipelines, would replace the file.
    vktest::HeadlessVulkan warm(const char* mode) {
        const auto t0 = SteadyClock::now();
        auto vk = vktest::openHeadlessVulkan("VulkanPipelineWarmup", kW, kH);
        if (!vk) {
            std::printf("%s: no Vulkan device, nothing to warm\n", mode);
            return vk;
        }
        VulkanRenderer& renderer = *vk.renderer;
        // Native output, as most tests pin it: with DLSS Ray Reconstruction on
        // (the default where it is built) the frames below would build a shade
        // permutation of their own that most tests never use.
        vktest::pinNativeOutput(renderer);
        std::printf("%s: renderer up in %.1f s (ray tracing %s)\n", mode, secondsSince(t0),
                    renderer.rayTracingSupported() ? "on" : "off");

        const auto tShade = SteadyClock::now();
        const std::size_t built = renderer.prewarmShadePermutations();
        std::printf("%s: %zu shade permutation(s) built in %.1f s\n", mode, built, secondsSince(tShade));

        Scene scene;
        scene.environment = vktest::constantEnv(0.2f, 0.2f, 0.2f);
        buildTraceScene(scene);
        auto camera = PerspectiveCamera::create(45, float(kW) / float(kH), 0.1f, 50.f);
        camera->position.set(0, 1.2f, 3.f);
        camera->lookAt(0, 0.5f, 0);
        camera->updateMatrixWorld();

        // The scene's own frame, then the trace: what is not built yet of the
        // frame's pipelines and the tracer's is built here, on this thread.
        const auto tFrames = SteadyClock::now();
        for (int i = 0; i < 2; ++i) renderer.render(scene, *camera);
        if (renderer.rayTracingSupported()) {
            VulkanRenderer::PathTraceParams params;
            params.samplesPerFrame = 1;
            params.maxBounces      = 2;
            std::vector<float> rgb;
            int w = 0, h = 0;
            const bool traced = renderer.pathTrace(scene, *camera, 1, rgb, w, h, params);
            std::printf("%s: frames and a %s trace in %.1f s\n", mode, traced ? "one-sample" : "failed",
                        secondsSince(tFrames));
        } else {
            std::printf("%s: frames in %.1f s\n", mode, secondsSince(tFrames));
        }
        // Builds nothing now; saves what the frames and the trace compiled.
        (void) renderer.prewarmShadePermutations();
        return vk;
    }

}// namespace

int main() {
    std::printf("VulkanPipelineWarmup\n");
    const auto t0 = SteadyClock::now();
    vktest::HeadlessVulkan rt, noRt;
    vktest::setEnv("THREEPP_VK_PIPELINE_CACHE_CAP_MB", "192");
    try {
        rt = warm("ray traced");
    } catch (const std::exception& e) {
        std::printf("[warn] ray-traced warm-up stopped: %s\n", e.what());
    }
    // The second renderer loads what the first saved, so with the tests' cap:
    // under the lower one it could drop the first one's work again.
    vktest::setEnv("THREEPP_VK_PIPELINE_CACHE_CAP_MB", "");
    // A renderer reads THREEPP_VULKAN_NO_RT when its context is created, so the
    // second one runs the no-ray-tracing mode and its own shade module. Under
    // that knob already (the lavapipe CI job), the first one did the same work.
    vktest::setEnv("THREEPP_VULKAN_NO_RT", "1");
    try {
        noRt = warm("no ray tracing");
    } catch (const std::exception& e) {
        std::printf("[warn] no-ray-tracing warm-up stopped: %s\n", e.what());
    }
    std::printf("warm-up done in %.1f s\n", secondsSince(t0));
    return 0;
}
