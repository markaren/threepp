// VulkanSplatViewGrow_test: a secondary view larger than the primary turns its
// splats on, and the primary keeps drawing its splats.
//
// The bug this pins: the splat pass sizes ONE tile-range buffer to the largest
// claimed view. A view with more tiles than any before it grew the buffer
// (destroy + create), and only that view's descriptor sets were rewritten; the
// primary's sets still named the destroyed buffer. The validation layer is
// what sees that (VUID-vkCmdDispatch-None-08114), so CTest runs this under
// strict validation (tests/renderers/CMakeLists.txt): any validation error
// exits non-zero. The pixel checks below hold either way on most drivers,
// since freed memory tends to stay readable. Where no validation layer is
// installed, strict mode says so and checks nothing.
//
// Plain exit-code program, like the other Vulkan tests here. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available. Headless: no window opens.

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/splats/SplatData.hpp"

#include "vulkan_test_util.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    // Pixels materially brighter than the dark background and orange-ish: the cloud.
    long coverage(const std::vector<unsigned char>& img) {
        long n = 0;
        for (size_t i = 0; i + 2 < img.size(); i += 3)
            if (img[i] > 70 && img[i] > img[i + 2] + 25) ++n;
        return n;
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    constexpr int kW = 160, kH = 120;
    auto vk = vktest::openHeadlessVulkan("VulkanSplatViewGrow_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    Canvas& canvas    = *vk.canvas;
    VulkanRenderer& r = *vk.renderer;

    Scene scene;
    scene.background = Color(0.02f, 0.02f, 0.03f);

    SplatGenerator::Options o;
    o.count      = 20000;
    o.seed       = 4242u;
    o.shDegree   = 0;
    o.extent.set(1.2f, 0.8f, 1.2f);
    o.minScale   = 0.05f;
    o.maxScale   = 0.10f;
    o.minOpacity = 0.6f;
    o.maxOpacity = 0.95f;
    auto data = SplatGenerator::generate(o);
    for (size_t i = 0; i < data.count(); ++i) data.setDcColor(i, Vector3{0.95f, 0.35f, 0.10f});
    scene.add(SplatCloud::create(data));

    PerspectiveCamera camera(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0.f, 0.f, 5.f);
    camera.lookAt(Vector3());

    const auto frames = [&](int n) {
        for (int i = 0; i < n; ++i) canvas.animateOnce([&] { r.render(scene, camera); });
    };

    std::printf("the primary alone\n");
    frames(30);
    const long before = coverage(r.readRGBPixels());
    check(before > 200, "the primary draws the cloud (" + std::to_string(before) + " px)");

    std::printf("a secondary view with more tiles than the primary turns splats on\n");
    PerspectiveCamera viewCam(50.f, 4.f / 3.f, 0.1f, 100.f);
    viewCam.position.set(0.f, 0.f, 5.f);
    viewCam.lookAt(Vector3());
    const uint32_t big = r.addView(viewCam, 640, 480);
    check(big != 0, "addView returned a handle");
    check(r.setViewSplats(big, true), "setViewSplats on the big view");
    frames(30);

    const long after = coverage(r.readRGBPixels());
    check(after > before / 2, "the primary still draws the cloud (" + std::to_string(after) + " px)");
    const long inView = coverage(r.readViewRGBPixels(big));
    check(inView > 4 * before, "the big view draws it too (" + std::to_string(inView) + " px)");

    (void) r.removeView(big);
    frames(3);

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
