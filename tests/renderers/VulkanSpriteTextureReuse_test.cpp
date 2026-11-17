// VulkanSpriteTextureReuse_test: a sprite's texture can die and a new one take
// its place, and the renderer draws the new one.
//
// The bug this pins: world-space sprites and ParticleSystems share a texture
// cache keyed by the raw Texture*, checked only by version and size. A texture
// freed and a new one of the same size made at its address hit the dead
// texture's entry and drew ITS image. Nothing ever removed an entry either, so
// every texture a sprite had used kept its GPU image until the renderer was
// destroyed. The screen-space (HUD) sprite atlas had the second half: it
// noticed the dead texture but never freed its image. tlasStats() counts the
// images both caches hold (spriteTextures, screenSpriteAtlases).
//
// CTest runs it under strict validation (tests/renderers/CMakeLists.txt): the
// caches retire images, and a retire too early is a use-after-free the
// validation layer reports.
//
// Plain exit-code program, like the other Vulkan tests here. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available. Headless: no window opens.

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include "vulkan_test_util.hpp"

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 128, kH = 96;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    Image solidImage(unsigned char r, unsigned char g, unsigned char b) {
        std::vector<unsigned char> px(4 * 4 * 4);
        for (size_t i = 0; i < px.size(); i += 4) {
            px[i + 0] = r;
            px[i + 1] = g;
            px[i + 2] = b;
            px[i + 3] = 255;
        }
        return Image{std::move(px), 4u, 4u};
    }

    void setUp(Texture& tex) {
        tex.format = Format::RGBA;
        tex.type   = Type::UnsignedByte;
        tex.needsUpdate();
    }

    std::shared_ptr<Texture> solidTexture(unsigned char r, unsigned char g, unsigned char b) {
        auto tex = Texture::create(solidImage(r, g, b));
        setUp(*tex);
        return tex;
    }

    // A Texture built in fixed storage, so the next one built there has the
    // dead one's address for certain rather than when the heap feels like it.
    struct PlacedTexture: Texture {
        explicit PlacedTexture(Image image): Texture({std::move(image)}) {}
    };
    alignas(PlacedTexture) unsigned char placedStorage[sizeof(PlacedTexture)];

    std::shared_ptr<Texture> placedTexture(unsigned char r, unsigned char g, unsigned char b) {
        auto* tex = new (placedStorage) PlacedTexture(solidImage(r, g, b));
        setUp(*tex);
        return {tex, [](Texture* t) { t->~Texture(); }};
    }

    std::shared_ptr<Sprite> makeSprite(const std::shared_ptr<Texture>& map, bool screenSpace) {
        auto mat   = SpriteMaterial::create();
        mat->map   = map;
        mat->color = Color(1, 1, 1);
        auto s     = Sprite::create(mat);
        if (screenSpace) {
            // 24 px square, its centre 20 px in from the bottom-right corner
            // (anchor (1, 0) is the right edge and the bottom).
            s->screenSpace = true;
            s->screenAnchor.set(1.f, 0.f);
            s->position.set(-20.f, 20.f, 0.f);
            s->scale.set(24.f, 24.f, 1.f);
        } else {
            s->scale.set(3.f, 3.f, 1.f);// fills the middle of the frame
        }
        return s;
    }

    std::array<int, 3> pixel(const std::vector<unsigned char>& im, int x, int y) {
        const size_t i = (static_cast<size_t>(y) * kW + x) * 3;
        if (im.size() < i + 3) return {-1, -1, -1};
        return {im[i], im[i + 1], im[i + 2]};
    }

    bool isRed(const std::array<int, 3>& p) { return p[0] > 150 && p[1] < 60 && p[2] < 60; }
    bool isBlue(const std::array<int, 3>& p) { return p[2] > 150 && p[0] < 60 && p[1] < 60; }

    std::string str(const std::array<int, 3>& p) {
        return "(" + std::to_string(p[0]) + " " + std::to_string(p[1]) + " " + std::to_string(p[2]) + ")";
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto vk = vktest::openHeadlessVulkan("VulkanSpriteTextureReuse_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    Canvas& canvas    = *vk.canvas;
    VulkanRenderer& r = *vk.renderer;

    Scene scene;
    scene.background = Color(0.02f, 0.02f, 0.02f);
    PerspectiveCamera camera(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0.f, 0.f, 5.f);
    camera.lookAt(Vector3());

    const auto frames = [&](int n) {
        for (int i = 0; i < n; ++i) canvas.animateOnce([&] { r.render(scene, camera); });
    };
    const auto world = [&] { return pixel(r.readRGBPixels(), kW / 2, kH / 2); };
    const auto hud   = [&] { return pixel(r.readRGBPixels(), kW - 20, kH - 20); };

    const auto held = [&](const char* what, uint32_t worldImages, uint32_t hudImages) {
        const auto s = r.tlasStats();
        check(s.spriteTextures == worldImages && s.screenSpriteAtlases == hudImages,
              std::string(what) + ": " + std::to_string(s.spriteTextures) + " world-sprite and " +
                      std::to_string(s.screenSpriteAtlases) + " HUD images held, want " +
                      std::to_string(worldImages) + " and " + std::to_string(hudImages));
    };

    std::printf("red textures\n");
    auto worldSprite = makeSprite(placedTexture(255, 0, 0), false);
    auto hudSprite   = makeSprite(solidTexture(255, 0, 0), true);
    scene.add(worldSprite);
    scene.add(hudSprite);
    frames(8);
    std::weak_ptr<Texture> oldWorldTex = worldSprite->material()->as<SpriteMaterial>()->map;
    check(isRed(world()), "the world sprite is red " + str(world()));
    check(isRed(hud()), "the HUD sprite is red " + str(hud()));
    held("drawn", 1, 1);

    std::printf("both freed, and blue ones made in their place\n");
    scene.remove(*worldSprite);
    scene.remove(*hudSprite);
    worldSprite.reset();
    hudSprite.reset();
    frames(2);
    // The new world texture is built where the old one was: neither its
    // address, its size nor its version tells the two apart.
    check(oldWorldTex.expired(), "the red world texture is gone before the blue one is built in its place");
    check(r.tlasStats().spriteTextures == 0, "and its image is freed");
    worldSprite = makeSprite(placedTexture(0, 0, 255), false);
    hudSprite   = makeSprite(solidTexture(0, 0, 255), true);
    scene.add(worldSprite);
    scene.add(hudSprite);
    frames(8);
    check(isBlue(world()), "the world sprite is blue " + str(world()));
    check(isBlue(hud()), "the HUD sprite is blue " + str(hud()));
    // One each: the red ones' images went when their textures died.
    held("the blue ones drawn", 1, 1);

    std::printf("the world sprite out of the scene for a while, its texture alive\n");
    scene.remove(*worldSprite);
    frames(300);
    // Kept: an effect shown now and then must not upload its texture again
    // every time it comes back.
    held("after 300 frames away", 1, 1);
    scene.add(worldSprite);
    frames(4);
    check(isBlue(world()), "back, and blue " + str(world()));

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
