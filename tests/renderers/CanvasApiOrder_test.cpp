// CanvasApiOrder_test — a GL canvas must still work after a Vulkan one.
//
// GLFW window hints are sticky, process-global state: they persist across
// glfwCreateWindow until something resets them. Canvas sets
// GLFW_CLIENT_API=GLFW_NO_API for a Vulkan canvas, so before Canvas::initWindow
// called glfwDefaultWindowHints() a GL canvas created *afterwards* in the same
// process silently got a window with no OpenGL context. glfwMakeContextCurrent
// then failed, glad loaded nothing, and the process died via a bare
// exit(EXIT_FAILURE) — which in an embedded host (the Python module) killed the
// interpreter with no traceback at all.
//
// This lives in its own executable on purpose: the hazard only exists when both
// backends are used in ONE process, and it is invisible in any ordering where
// the GL canvas comes first (which is what the alphabetical test ordering
// happened to give, hiding the bug).
//
// Plain exit-code program, not Catch2. Exits 42 (→ CTest "Skipped") when no
// Vulkan-capable GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/renderers/VulkanRenderer.hpp"
#include "vulkan_test_util.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

using namespace threepp;

namespace {

    using vktest::kSkipCode;

    using vktest::check;
    using vktest::failures;

    // A lit box on a coloured background — enough that a working context
    // produces a non-uniform image.
    std::shared_ptr<Scene> makeScene() {
        auto scene = Scene::create();
        scene->add(HemisphereLight::create(0xffffff, 0x404040, 1.f));
        auto key = DirectionalLight::create(0xffffff, 2.f);
        key->position.set(3, 5, 2);
        scene->add(key);
        scene->add(Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create()));
        return scene;
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    constexpr int kW = 128, kH = 96;

    // ── 1. Vulkan first: this is what leaks GLFW_CLIENT_API=GLFW_NO_API ──────
    auto gpu = vktest::openHeadlessVulkan("CanvasApiOrder-vk", kW, kH);
    if (!gpu) return kSkipCode;
    std::printf("  [ ok ] Vulkan renderer constructed\n");

    // The Vulkan renderer stays ALIVE across the GL construction below — that is
    // the shape the Python test session has (a module-scoped Vulkan fixture
    // outliving the creation of the session's GL renderer).

    // ── 2. GL second: the step that used to hard-exit the process ───────────
    auto scene = makeScene();
    auto camera = PerspectiveCamera::create(60, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera->position.set(0, 1.5f, 4);
    camera->lookAt(0, 0, 0);

    // Any throw from here on is a real failure, not a missing-GPU skip: a Vulkan
    // device was already obtained above. Caught (rather than left to terminate)
    // so a regression reports the reason instead of aborting the process.
    try {
        Canvas glCanvas(Canvas::Parameters().title("CanvasApiOrder-gl").size(kW, kH).headless(true));
        GLRenderer glRenderer(glCanvas);
        glRenderer.setClearColor(Color(0x202830));
        std::printf("  [ ok ] GL renderer constructed after Vulkan\n");

        // Constructing without dying is necessary but not sufficient — a
        // context-less window could in principle construct and then draw
        // nothing. Prove the context works by rendering and measuring pixels.
        glRenderer.render(*scene, *camera);
        const auto px = glRenderer.readRGBPixels();

        check(px.size() == static_cast<size_t>(kW) * kH * 3, "GL readback has the expected size");
        if (!px.empty()) {
            const auto lo = *std::min_element(px.begin(), px.end());
            const auto hi = *std::max_element(px.begin(), px.end());
            check(hi > lo, "GL render after Vulkan is not flat (context is live)");
        }
    } catch (const std::exception& e) {
        check(false, std::string("GL renderer usable after Vulkan (threw: ") + e.what() + ")");
    }

    std::printf(failures == 0 ? "\nOK\n" : "\n%d CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
