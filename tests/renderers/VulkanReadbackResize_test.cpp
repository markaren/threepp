// VulkanReadbackResize_test: readRGBPixels returns the frame that was rendered,
// or nothing, across a resize; never a freshly made image.
//
// The bug this pins: a readback inside the animate body calls endFrame(), and a
// resize due at that frame end recreated the swapchain (here the headless ring)
// BEFORE the copy. The copy then read an image of the new ring, which nothing
// had rendered into, from a layout it was never in: undefined pixels at the new
// size, and a validation error. A readback after a resize with no frame since
// read the same kind of image.
//
// Plain exit-code program, like the other Vulkan tests here. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available. Headless: no window opens.

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/scenes/Scene.hpp"

#include "vulkan_test_util.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    using vktest::check;
    using vktest::failures;

    std::size_t bytesOf(const WindowSize& s) {
        return static_cast<std::size_t>(s.width()) * static_cast<std::size_t>(s.height()) * 3;
    }

    // The centre pixel of an RGB8 readback is the red background.
    bool centreIsRed(const std::vector<unsigned char>& px, int w, int h) {
        if (px.size() != static_cast<std::size_t>(w) * h * 3) return false;
        const std::size_t i = (static_cast<std::size_t>(h / 2) * w + w / 2) * 3;
        return px[i + 0] > 150 && px[i + 1] < 60 && px[i + 2] < 60;
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto vk = vktest::openHeadlessVulkan("VulkanReadbackResize_test", 96, 64);
    if (!vk) return vktest::kSkipCode;
    Canvas& canvas           = *vk.canvas;
    VulkanRenderer& renderer = *vk.renderer;

    Scene scene;
    scene.background = Color(1.f, 0.f, 0.f);
    PerspectiveCamera camera(60.f, canvas.aspect(), 0.1f, 100.f);

    std::printf("before any frame\n");
    check(renderer.readRGBPixels().empty(), "no pixels before the first render()");

    std::printf("a plain frame\n");
    canvas.animateOnce([&] { renderer.render(scene, camera); });
    const WindowSize before = renderer.framebufferSize();
    const auto first        = renderer.readRGBPixels();
    check(first.size() == bytesOf(before), "w*h*3 bytes at " + std::to_string(before.width()) + "x" +
                                                   std::to_string(before.height()));
    check(centreIsRed(first, before.width(), before.height()), "and they are the red background");

    std::printf("a resize due at the end of the frame being read\n");
    canvas.setSize({128, 80});
    std::vector<unsigned char> held;
    canvas.animateOnce([&] {
        renderer.render(scene, camera);
        held = renderer.readRGBPixels();
    });
    // The frame was recorded at the old extent; those are the pixels it made,
    // and framebufferSize() (which callers size the pixels by) still says so.
    check(renderer.framebufferSize() == before, "framebufferSize() is still the old size");
    check(held.size() == bytesOf(before), "the readback is the frame rendered, at the old size");
    check(centreIsRed(held, before.width(), before.height()), "and it holds the red background");
    const auto again = renderer.readRGBPixels();
    check(again.size() == bytesOf(before) && centreIsRed(again, before.width(), before.height()),
          "a second read returns the same frame");

    std::printf("the next render() makes the resize\n");
    canvas.animateOnce([&] { renderer.render(scene, camera); });
    const WindowSize after = renderer.framebufferSize();
    check(!(after == before), "the resize happened (" + std::to_string(after.width()) + "x" +
                                      std::to_string(after.height()) + ")");
    const auto next = renderer.readRGBPixels();
    check(next.size() == bytesOf(after), "w*h*3 bytes at the new size");
    check(centreIsRed(next, after.width(), after.height()), "and they are the red background");

    std::printf("a resize made at a frame end with no readback in the frame\n");
    canvas.setSize({112, 72});
    canvas.animateOnce([&] { renderer.render(scene, camera); });
    check(renderer.readRGBPixels().empty(), "no pixels: the new images hold nothing yet");
    canvas.animateOnce([&] { renderer.render(scene, camera); });
    const WindowSize last = renderer.framebufferSize();
    const auto lastPx     = renderer.readRGBPixels();
    check(lastPx.size() == bytesOf(last) && centreIsRed(lastPx, last.width(), last.height()),
          "the next frame reads at " + std::to_string(last.width()) + "x" + std::to_string(last.height()));

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
