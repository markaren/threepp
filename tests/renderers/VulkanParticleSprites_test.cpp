// VulkanParticleSprites_test: ParticleSprites on the Vulkan renderer, whose particle pass
// draws them (the GL twin is gl/GLParticleSprites_test).
//
//  - a sprite is drawn, and square on a 2:1 canvas (the billboard used one scale for both
//    axes and came out twice as wide as tall);
//  - the material's anisotropy reaches the particle lighting: with the sun behind the
//    camera a dense billow (g < 0) is brighter than thin haze (g > 0), and looking into
//    the sun the other way round;
//  - self-shadowing reaches it too: it dims the sun on a sprite, and not the ambient light.
//
// Plain exit-code program, like the other Vulkan tests here. Exits 42 (CTest "Skipped")
// when no Vulkan GPU is available. Headless: no window opens.

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/ParticleSprites.hpp"
#include "threepp/scenes/Scene.hpp"

#include "vulkan_test_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 160, kH = 80;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    int lum(const std::vector<unsigned char>& px, int x, int y) {
        const size_t i = (static_cast<size_t>(y) * kW + x) * 3;
        if (px.size() < i + 3) return -1;
        return px[i] + px[i + 1] + px[i + 2];
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto vk = vktest::openHeadlessVulkan("VulkanParticleSprites_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    Canvas& canvas = *vk.canvas;
    VulkanRenderer& r = *vk.renderer;

    Scene scene;
    scene.background = Color(0, 0, 0);
    PerspectiveCamera camera(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0.f, 0.f, 5.f);
    camera.lookAt(Vector3());

    auto sprites = ParticleSprites::create(2);
    sprites->hideAll();
    scene.add(sprites);

    const auto frames = [&](int n) {
        for (int i = 0; i < n; ++i) canvas.animateOnce([&] { r.render(scene, camera); });
    };

    std::printf("one sprite in ambient light\n");
    auto ambient = AmbientLight::create(Color(1, 1, 1), math::PI);
    scene.add(ambient);
    sprites->set(0, Vector3(), 1.6f, 1.f);
    sprites->commit();
    frames(6);
    {
        const auto px = r.readRGBPixels();
        const int threshold = 60;
        int w = 0, h = 0;
        for (int x = 0; x < kW; ++x) w += lum(px, x, kH / 2) > threshold;
        for (int y = 0; y < kH; ++y) h += lum(px, kW / 2, y) > threshold;
        check(h > 4, "drawn: " + std::to_string(h) + " px tall");
        check(std::abs(w - h) <= 3, "square on a 2:1 canvas: " + std::to_string(w) + " x " + std::to_string(h) + " px");
    }
    scene.remove(*ambient);

    std::printf("a grey sprite lit by a sun, for g = -0.2 and 0.35\n");
    auto sun = DirectionalLight::create(Color(1, 1, 1), 0.6f);
    scene.add(sun);
    const auto centre = [&](float sunZ, float g) {
        sun->position.set(0.f, 0.f, sunZ);
        sprites->setAnisotropy(g);
        sprites->set(0, Vector3(), 2.f, 1.f, Color(0.5f, 0.5f, 0.5f));
        sprites->commit();
        frames(6);
        return lum(r.readRGBPixels(), kW / 2, kH / 2);
    };
    const int behindDense = centre(10.f, -0.2f), behindThin = centre(10.f, 0.35f);
    const int aheadDense = centre(-10.f, -0.2f), aheadThin = centre(-10.f, 0.35f);
    check(behindDense > behindThin + 30, "sun behind the camera: dense " + std::to_string(behindDense) +
                                                 " brighter than thin " + std::to_string(behindThin));
    check(aheadThin > aheadDense + 30, "looking into the sun: thin " + std::to_string(aheadThin) +
                                               " brighter than dense " + std::to_string(aheadDense));

    std::printf("self-shadowing: the sun on a grey sprite, then an ambient light\n");
    const auto selfShadowed = [&](float k) {
        sprites->setAnisotropy(0.35f);
        sprites->setSelfShadowing(k);
        sprites->set(0, Vector3(), 2.f, 1.f, Color(0.5f, 0.5f, 0.5f));
        sprites->commit();
        frames(6);
        return lum(r.readRGBPixels(), kW / 2, kH / 2);
    };
    sun->position.set(0.f, 0.f, 10.f);
    const int sunOpen = selfShadowed(0.f), sunDense = selfShadowed(0.5f);
    check(sunOpen > sunDense + 30, "sun: open " + std::to_string(sunOpen) + " brighter than self-shadowed " +
                                           std::to_string(sunDense));
    scene.remove(*sun);
    auto fill = AmbientLight::create(Color(1, 1, 1), 1.5f);
    scene.add(fill);
    const int ambOpen = selfShadowed(0.f), ambDense = selfShadowed(0.9f);
    check(ambOpen > 30 && std::abs(ambOpen - ambDense) <= 3,
          "ambient: open " + std::to_string(ambOpen) + ", self-shadowed " + std::to_string(ambDense) + " (the same)");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
