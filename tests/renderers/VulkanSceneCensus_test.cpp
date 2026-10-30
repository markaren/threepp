// VulkanSceneCensus_test: a frame draws the lights and sprites the scene holds
// at that render(), whichever way the scene came to hold them.
//
// The Vulkan renderer collects a scene's lights, world sprites and screen-space
// sprites apart from its mesh entries, every frame, because these change without
// anything the snapshot of the scene would notice. Sprites were not drawn by any
// other Vulkan test, and rg_ab has none, so a collector that kept last frame's
// list would pass every image comparison in the tree. So, on a dim wall:
//   lamp     white PointLight in a group, hidden from the start, then shown;
//   green    world-space sprite in a group, hidden with it, then shown;
//   blue     screen-space sprite in the same group;
//   red      a sprite and a red lamp added between two frames, then removed and
//            destroyed: the frame after must neither draw nor touch them;
//   cyan     a sprite in a second scene, rendered in turn with the first;
//   yellow   a sprite in a HUD scene, drawn over the first in the same frame.

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "vulkan_test_util.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 256, kH = 128;
    constexpr int kSettleFrames = 16;
    constexpr int kPatch = 3;

    std::shared_ptr<Texture> whiteTexture() {
        auto tex = Texture::create(Image{std::vector<unsigned char>(4 * 4 * 4, 255), 4u, 4u});
        tex->format = Format::RGBA;
        tex->type = Type::UnsignedByte;
        tex->needsUpdate();
        return tex;
    }

    std::shared_ptr<Sprite> makeSprite(const std::shared_ptr<Texture>& map, const Color& c, float size) {
        auto mat = SpriteMaterial::create();
        mat->map = map;
        mat->color = c;
        mat->transparent = true;
        mat->depthWrite = false;
        auto s = Sprite::create(mat);
        s->scale.set(size, size, 1.f);
        return s;
    }

    std::array<double, 3> patchMean(const std::vector<unsigned char>& a, const std::array<int, 2>& p) {
        std::array<double, 3> sum{0, 0, 0};
        int n = 0;
        for (int y = p[1] - kPatch; y <= p[1] + kPatch; ++y) {
            for (int x = p[0] - kPatch; x <= p[0] + kPatch; ++x) {
                for (int c = 0; c < 3; ++c) sum[c] += a[(static_cast<size_t>(y) * kW + x) * 3 + c];
                ++n;
            }
        }
        for (double& v : sum) v /= n;
        return sum;
    }

    // Channel c leads both others by 2x.
    bool leads(const std::array<double, 3>& p, int c) {
        const double o1 = p[(c + 1) % 3], o2 = p[(c + 2) % 3];
        return p[c] > 2.0 * o1 + 4.0 && p[c] > 2.0 * o2 + 4.0;
    }

    bool yellow(const std::array<double, 3>& p) {
        return p[0] > 2.0 * p[2] + 4.0 && p[1] > 2.0 * p[2] + 4.0;
    }

    bool cyan(const std::array<double, 3>& p) {
        return p[1] > 2.0 * p[0] + 4.0 && p[2] > 2.0 * p[0] + 4.0;
    }

    double luma(const std::array<double, 3>& p) {
        return (p[0] + p[1] + p[2]) / 3.0;
    }

}// namespace

int main() {

    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto gpu = vktest::openHeadlessVulkan("VulkanSceneCensus_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;

    renderer.setDenoise(false);
    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    PerspectiveCamera camera(30.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0, 6.f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    const auto map = whiteTexture();

    Scene scene;
    auto wallMat = MeshStandardMaterial::create();
    wallMat->color = Color(1, 1, 1);
    wallMat->metalness = 0.f;
    wallMat->roughness = 1.f;
    scene.add(Mesh::create(PlaneGeometry::create(12.f, 6.f), wallMat));

    auto lampGroup = Group::create();
    lampGroup->visible = false;
    scene.add(lampGroup);
    auto lamp = PointLight::create(Color(1, 1, 1), 0.8f);
    lamp->position.set(-1.5f, -0.9f, 0.7f);
    lampGroup->add(lamp);

    auto spriteGroup = Group::create();
    spriteGroup->visible = false;
    scene.add(spriteGroup);
    auto green = makeSprite(map, Color(0, 1, 0), 0.5f);
    green->position.set(-1.5f, 0.6f, 1.f);
    spriteGroup->add(green);
    auto blue = makeSprite(map, Color(0, 0, 1), 24.f);
    blue->screenSpace = true;
    blue->screenAnchor.set(1.f, 0.f);
    blue->position.set(-20.f, 20.f, 0.f);
    spriteGroup->add(blue);

    Scene other;
    auto cyanSprite = makeSprite(map, Color(0, 1, 1), 0.5f);
    cyanSprite->position.set(0.f, 0.6f, 1.f);
    other.add(cyanSprite);

    Scene hud;
    auto hudCam = OrthographicCamera::create(0.f, static_cast<float>(kW), static_cast<float>(kH), 0.f, 0.1f, 10.f);
    hudCam->position.z = 1.f;
    auto yellowSprite = makeSprite(map, Color(1, 1, 0), 24.f);
    yellowSprite->position.set(kW * 0.5f, kH * 0.5f, 0.f);
    hud.add(yellowSprite);

    const auto px = [&](float x, float y, float z) {
        Vector3 p(x, y, z);
        p.project(camera);
        return std::array<int, 2>{static_cast<int>((p.x * 0.5f + 0.5f) * kW),
                                  static_cast<int>((1.f - (p.y * 0.5f + 0.5f)) * kH)};
    };
    const auto pGreen = px(-1.5f, 0.6f, 1.f), pRed = px(1.5f, 0.6f, 1.f), pCyan = px(0.f, 0.6f, 1.f);
    const auto pLampWall = px(-1.5f, -0.9f, 0.f), pLateWall = px(1.5f, -0.9f, 0.f);
    const std::array<int, 2> pBlue{kW - 20, kH - 20}, pYellow{kW / 2, kH / 2};

    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    try {
        const auto readback = [&] {
            auto out = renderer.readRGBPixels();
            if (out.size() < static_cast<size_t>(kW) * kH * 3) {
                std::printf("readback too small: %zu\n", out.size());
                std::exit(1);
            }
            return out;
        };
        const auto settle = [&](Scene& s) {
            for (int i = 0; i < kSettleFrames; ++i) {
                canvas.animateOnce([&] { renderer.render(s, camera); });
            }
            return readback();
        };
        const auto report = [&](const std::vector<unsigned char>& im, const char* label) {
            const auto g = patchMean(im, pGreen), b = patchMean(im, pBlue), r = patchMean(im, pRed);
            const auto c = patchMean(im, pCyan), y = patchMean(im, pYellow);
            std::printf("%s\n  green (%3.0f %3.0f %3.0f) blue (%3.0f %3.0f %3.0f) red (%3.0f %3.0f %3.0f) "
                        "cyan (%3.0f %3.0f %3.0f) yellow (%3.0f %3.0f %3.0f)\n  lamp wall %.1f, late wall %.1f\n",
                        label, g[0], g[1], g[2], b[0], b[1], b[2], r[0], r[1], r[2], c[0], c[1], c[2],
                        y[0], y[1], y[2], luma(patchMean(im, pLampWall)), luma(patchMean(im, pLateWall)));
        };

        auto im = settle(scene);
        report(im, "lamp and sprites in hidden groups:");
        const double dark = luma(patchMean(im, pLampWall));
        check("no light from the lamp under a hidden group", dark < 60.0);
        check("no sprite under a hidden group", !leads(patchMean(im, pGreen), 1) && !leads(patchMean(im, pBlue), 2));

        lampGroup->visible = true;
        spriteGroup->visible = true;
        im = settle(scene);
        report(im, "groups shown:");
        const double lateBefore = luma(patchMean(im, pLateWall));
        check("lamp lights the wall", luma(patchMean(im, pLampWall)) > dark + 40.0);
        check("green world sprite drawn", leads(patchMean(im, pGreen), 1));
        check("blue screen-space sprite drawn", leads(patchMean(im, pBlue), 2));

        spriteGroup->visible = false;
        im = settle(scene);
        report(im, "sprite group hidden again:");
        check("green gone", !leads(patchMean(im, pGreen), 1));
        check("blue gone", !leads(patchMean(im, pBlue), 2));
        check("lamp still lit", luma(patchMean(im, pLampWall)) > dark + 40.0);
        spriteGroup->visible = true;

        auto red = makeSprite(map, Color(1, 0, 0), 0.5f);
        red->position.set(1.5f, 0.6f, 1.f);
        scene.add(red);
        auto late = PointLight::create(Color(1, 0, 0), 0.8f);
        late->position.set(1.5f, -0.9f, 0.7f);
        scene.add(late);
        im = settle(scene);
        report(im, "red sprite and lamp added between frames:");
        check("added sprite drawn", leads(patchMean(im, pRed), 0));
        check("added lamp lights the wall", leads(patchMean(im, pLateWall), 0));

        scene.remove(*red);
        scene.remove(*late);
        red.reset();
        late.reset();
        im = settle(scene);
        report(im, "both removed and destroyed:");
        check("removed sprite gone", !leads(patchMean(im, pRed), 0));
        check("removed lamp gone", luma(patchMean(im, pLateWall)) < lateBefore + 10.0);

        im = settle(other);
        report(im, "the second scene:");
        check("its sprite drawn", cyan(patchMean(im, pCyan)));
        check("the first scene's sprites not drawn",
              !leads(patchMean(im, pGreen), 1) && !leads(patchMean(im, pBlue), 2));
        check("the first scene's lamp gives no light", luma(patchMean(im, pLampWall)) < dark + 10.0);

        im = settle(scene);
        report(im, "back to the first scene:");
        check("the second scene's sprite not drawn", !cyan(patchMean(im, pCyan)));
        check("green and blue back", leads(patchMean(im, pGreen), 1) && leads(patchMean(im, pBlue), 2));
        check("lamp back", luma(patchMean(im, pLampWall)) > dark + 40.0);

        for (int i = 0; i < kSettleFrames; ++i) {
            canvas.animateOnce([&] {
                renderer.render(scene, camera);
                renderer.render(hud, *hudCam);
            });
        }
        im = readback();
        report(im, "a HUD scene over the first in one frame:");
        check("HUD sprite drawn", yellow(patchMean(im, pYellow)));
        check("the scene under it keeps its sprites and lamp",
              leads(patchMean(im, pGreen), 1) && leads(patchMean(im, pBlue), 2) &&
                      luma(patchMean(im, pLampWall)) > dark + 40.0);

        if (vulkan::validationActive()) {
            check("no validation errors", vulkan::validationErrorCount() == 0u);
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL: the renderer threw: %s\n", e.what());
        ++failures;
    }

    std::printf("scene census: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
