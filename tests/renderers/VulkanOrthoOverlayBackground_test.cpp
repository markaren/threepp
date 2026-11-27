// VulkanOrthoOverlayBackground_test: a standalone render() with an
// OrthographicCamera (the 2D overlay path) clears to the scene's background
// colour, and screen-space sprites are placed in pixels whatever the camera.
//
// Two bugs this pins, both seen in examples/projects/MotorControl:
//  - the ortho-only frame cleared to the renderer's clearColor and ignored
//    scene.background, so a white 2D scene came out black (GLBackground clears
//    to the background colour at alpha 1, and so does this path now);
//  - a screenSpace sprite drawn by a standalone ortho render() or a HUD
//    render() went through the caller's camera, whose units are world units,
//    so a label at pixel (20, 20) landed far outside the frustum.
//
// Plain exit-code program, like the other Vulkan tests here. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available. Headless: no window opens.

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include "vulkan_test_util.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 128, kH = 96;

    using vktest::check;
    using vktest::failures;

    std::shared_ptr<Texture> solidTexture(unsigned char r, unsigned char g, unsigned char b) {
        std::vector<unsigned char> px(4 * 4 * 4);
        for (size_t i = 0; i < px.size(); i += 4) {
            px[i + 0] = r;
            px[i + 1] = g;
            px[i + 2] = b;
            px[i + 3] = 255;
        }
        auto tex = Texture::create(Image{std::move(px), 4u, 4u});
        tex->format = Format::RGBA;
        tex->type = Type::UnsignedByte;
        tex->needsUpdate();
        return tex;
    }

    // A 16 px red square, its centre at `anchor` (0..1 of the frame, y up)
    // plus `offset` pixels.
    std::shared_ptr<Sprite> screenSpaceSprite(float ax, float ay, float ox, float oy) {
        auto mat = SpriteMaterial::create();
        mat->map = solidTexture(255, 0, 0);
        mat->color = Color(1, 1, 1);
        auto s = Sprite::create(mat);
        s->screenSpace = true;
        s->screenAnchor.set(ax, ay);
        s->position.set(ox, oy, 0.f);
        s->scale.set(16.f, 16.f, 1.f);
        return s;
    }

    std::array<int, 3> pixel(const std::vector<unsigned char>& im, int x, int y) {
        const size_t i = (static_cast<size_t>(y) * kW + x) * 3;
        if (im.size() < i + 3) return {-1, -1, -1};
        return {im[i], im[i + 1], im[i + 2]};
    }

    bool near(const std::array<int, 3>& p, int r, int g, int b, int tol = 3) {
        return std::abs(p[0] - r) <= tol && std::abs(p[1] - g) <= tol && std::abs(p[2] - b) <= tol;
    }

    std::string str(const std::array<int, 3>& p) {
        return "(" + std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + ")";
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto vk = vktest::openHeadlessVulkan("VulkanOrthoOverlayBackground_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    Canvas& canvas = *vk.canvas;
    VulkanRenderer& r = *vk.renderer;

    // A 2D scene in world units: 4 x 3 across the frame, a black square in
    // the middle, a screen-space sprite 20 px in from the bottom-right corner.
    OrthographicCamera camera(-2.f, 2.f, 1.5f, -1.5f, 0.1f, 10.f);
    camera.position.z = 1.f;

    Scene scene;
    scene.background = Color(0x3366cc);// sRGB hex: shown as (51 102 204)
    scene.add(Mesh::create(PlaneGeometry::create(1.f, 1.f),
                           MeshBasicMaterial::create(MeshBasicMaterial::Params{}.color(0x000000))));
    scene.add(screenSpaceSprite(1.f, 0.f, -20.f, 20.f));

    const auto frames = [&](Scene& s, Camera& c, int n) {
        for (int i = 0; i < n; ++i) canvas.animateOnce([&] { r.render(s, c); });
    };

    std::printf("standalone ortho render, background colour\n");
    frames(scene, camera, 4);
    {
        const auto im = r.readRGBPixels();
        check(near(pixel(im, 4, 4), 51, 102, 204), "a corner is the background colour " + str(pixel(im, 4, 4)));
        check(near(pixel(im, kW / 2, kH / 2), 0, 0, 0), "the middle is the black square " + str(pixel(im, kW / 2, kH / 2)));
        check(near(pixel(im, kW - 20, kH - 20), 255, 0, 0, 8),
              "the screen-space sprite is 20 px in from the bottom-right " + str(pixel(im, kW - 20, kH - 20)));
    }

    std::printf("no background: the clear colour\n");
    scene.background = Background();
    r.setClearColor(Color(0x808080));
    frames(scene, camera, 2);
    {
        const auto im = r.readRGBPixels();
        check(near(pixel(im, 4, 4), 128, 128, 128), "a corner is the clear colour " + str(pixel(im, 4, 4)));
    }
    r.setClearColor(Color(0x000000));

    std::printf("HUD render() over a perspective frame, world-unit HUD camera\n");
    Scene world;
    world.background = Color(0x3366cc);
    PerspectiveCamera persp(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    persp.position.set(0.f, 0.f, 5.f);
    Scene hud;
    hud.add(screenSpaceSprite(0.f, 1.f, 20.f, -20.f));// 20 px in from the top-left
    for (int i = 0; i < 4; ++i) {
        canvas.animateOnce([&] {
            r.render(world, persp);
            r.render(hud, camera);
        });
    }
    {
        const auto im = r.readRGBPixels();
        check(near(pixel(im, 20, 20), 255, 0, 0, 8),
              "the HUD's screen-space sprite is 20 px in from the top-left " + str(pixel(im, 20, 20)));
    }

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
