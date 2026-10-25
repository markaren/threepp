// VulkanShadowCacheStale_test — the sun's cached cascades with
// setShadowCacheStaleFrames: a changed static caster set waits up to n frames
// for its redraw, and nothing goes missing meanwhile.
//
// A white ground seen from above, a high sun and small boxes over it, one
// shadow cascade. Each box's shadow footprint is found from settled frames
// with and without it; its "shadow share" in a frame is how far that
// footprint has gone from lit to shadowed (0 lit, 1 shadowed).
//
//   * stale 8, a settled box moves: its new shadow is there at once (drawn
//     over the cache), its old one stays (still in the cache) until the
//     cascade is redrawn, and is gone once the n frames have passed; the new
//     shadow stays through the box settling into the static set again.
//   * stale 8, a hidden box is shown: its shadow is there at once, and stays.
//   * stale 0 (the default): a box that moves leaves no shadow behind.
//
// Run standalone (plain exit-code program) or via CTest. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "vulkan_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 240, kH = 180;
    constexpr int kStale = 8;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    std::vector<float> luma(const std::vector<unsigned char>& px) {
        std::vector<float> out(px.size() / 3);
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = 0.2126f * px[i * 3] + 0.7152f * px[i * 3 + 1] + 0.0722f * px[i * 3 + 2];
        return out;
    }

    // A shadow's footprint: pixels within `radius` of its projected centre
    // (cx, cy) that are clearly darker with the caster than without. The
    // radius keeps the caster itself out: its shadow falls well clear of it.
    struct Footprint {
        std::vector<size_t> px;
        float               lit = 0.f, dark = 0.f;// mean luma over it without / with the shadow
    };
    Footprint footprint(const std::vector<float>& with, const std::vector<float>& without, float cx, float cy,
                        float radius) {
        Footprint f;
        for (int y = 0; y < kH; ++y) {
            for (int x = 0; x < kW; ++x) {
                const size_t i = size_t(y) * kW + x;
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) > radius * radius) continue;
                if (without[i] - with[i] > 40.f) f.px.push_back(i);
            }
        }
        for (size_t i : f.px) {
            f.lit += without[i];
            f.dark += with[i];
        }
        if (!f.px.empty()) {
            f.lit /= float(f.px.size());
            f.dark /= float(f.px.size());
        }
        return f;
    }
    // 0: as lit as without the caster, 1: as dark as its settled shadow.
    float share(const Footprint& f, const std::vector<float>& img) {
        if (f.px.empty() || f.lit - f.dark < 1.f) return -1.f;
        float m = 0.f;
        for (size_t i : f.px) m += img[i];
        m /= float(f.px.size());
        return (f.lit - m) / (f.lit - f.dark);
    }

}// namespace

int main() {
    std::printf("VulkanShadowCacheStale_test\n");
    auto vk = vktest::openHeadlessVulkan("VulkanShadowCacheStale_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping         = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;
    renderer.setShadowMode(VulkanRenderer::ShadowMode::ShadowMaps);
    renderer.setShadowCascades(1);
    renderer.setShadowDistance(30.f);
    check(renderer.shadowCacheStaleFrames() == 0u, "stale frames are 0 by default");

    Scene scene;
    scene.background = Color(0.2f, 0.2f, 0.2f);
    // A low sun: a box kHeight up throws its shadow (-kHeight, -kHeight / 2)
    // away, well clear of the box itself as the camera sees it from above.
    constexpr float kHeight = 3.f;
    auto sun = DirectionalLight::create(Color(1, 1, 1), 3.f);
    sun->position.set(1.f, 1.f, 0.5f);
    scene.add(sun);
    scene.add(HemisphereLight::create(Color(1, 1, 1), Color(1, 1, 1), 0.3f));

    PerspectiveCamera camera(50.f, float(kW) / float(kH), 0.1f, 100.f);
    camera.position.set(0.f, 12.f, 0.01f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    auto groundMat       = MeshStandardMaterial::create();
    groundMat->color     = Color(1, 1, 1);
    groundMat->roughness = 1.f;
    auto ground          = Mesh::create(PlaneGeometry::create(20.f, 20.f), groundMat);
    ground->rotation.x   = -math::PI / 2;
    scene.add(ground);

    auto boxMat   = MeshStandardMaterial::create();
    boxMat->color = Color(0.5f, 0.5f, 0.5f);
    const auto box = [&](float x, float z) {
        auto b = Mesh::create(BoxGeometry::create(1.2f, 0.3f, 1.2f), boxMat);
        b->position.set(x, kHeight, z);
        scene.add(b);
        return b;
    };
    auto mover = box(-3.f, -2.f);
    auto shown = box(3.f, 2.f);

    double     clock  = 0.0;
    const auto frames = [&](int n) {
        for (int i = 0; i < n; ++i) {
            clock += 1.0 / 60.0;
            renderer.setSimTime(clock);
            vk.canvas->animateOnce([&] { renderer.render(scene, camera); });
        }
    };
    const auto settledLuma = [&] {
        frames(60);
        return luma(renderer.readRGBPixels());
    };

    // The footprints: each box's shadow at A (the mover's start), B (where it
    // goes) and C (the shown box), from settled frames.
    const float ax = -3.f, bx = 1.f;
    const auto withA = settledLuma();
    mover->position.x = bx;
    const auto withB = settledLuma();
    shown->visible = false;
    const auto noC = settledLuma();
    shown->visible = true;
    const auto withC = settledLuma();
    // Where a box at (x, z) throws its shadow, on screen.
    const auto shadowAt = [&](float x, float z) {
        Vector3 p(x - kHeight, 0.f, z - 0.5f * kHeight);
        p.project(camera);
        return std::pair<float, float>{(p.x + 1.f) * 0.5f * float(kW), (1.f - p.y) * 0.5f * float(kH)};
    };
    const auto [sax, say] = shadowAt(ax, -2.f);
    const auto [sbx, sby] = shadowAt(bx, -2.f);
    const auto [scx, scy] = shadowAt(3.f, 2.f);
    constexpr float kRadius = 14.f;// px; the shadow's centre is ~50 px from its box's
    const Footprint fa = footprint(withA, withB, sax, say, kRadius), fb = footprint(withB, withA, sbx, sby, kRadius),
                    fc = footprint(withC, noC, scx, scy, kRadius);
    std::printf("footprints: A %zu px, B %zu px, C %zu px\n", fa.px.size(), fb.px.size(), fc.px.size());
    check(fa.px.size() > 100 && fb.px.size() > 100 && fc.px.size() > 100, "each box throws a shadow of its own");

    // ── stale 8: a settled box moves ──
    std::printf("\nstale %d: a settled box moves\n", kStale);
    renderer.setShadowCacheStaleFrames(uint32_t(kStale));
    mover->position.x = ax;
    shown->visible    = false;
    frames(60);
    mover->position.x = bx;
    frames(2);
    {
        const auto img = luma(renderer.readRGBPixels());
        std::printf("  2 frames on: A %.2f, B %.2f\n", share(fa, img), share(fb, img));
        check(share(fb, img) > 0.6f, "its new shadow is there at once");
        check(share(fa, img) > 0.6f, "its old shadow is still in the cache");
    }
    frames(kStale + 6);
    {
        const auto img = luma(renderer.readRGBPixels());
        std::printf("  %d frames on: A %.2f, B %.2f\n", kStale + 8, share(fa, img), share(fb, img));
        check(share(fa, img) < 0.2f, "the old shadow is gone once the cascade is redrawn");
        check(share(fb, img) > 0.8f, "the new shadow stays");
    }
    // 30 frames after the move the box rejoins the static set, without its
    // bit until the cascade is redrawn (up to kStale frames on): drawn over
    // the cache meanwhile.
    frames(34 - (kStale + 8));
    {
        const auto img = luma(renderer.readRGBPixels());
        std::printf("  34 frames on (rejoined, not yet redrawn): A %.2f, B %.2f\n", share(fa, img), share(fb, img));
        check(share(fb, img) > 0.8f, "its shadow stays as it rejoins the static set");
    }
    frames(26);
    {
        const auto img = luma(renderer.readRGBPixels());
        std::printf("  settled again: A %.2f, B %.2f\n", share(fa, img), share(fb, img));
        check(share(fb, img) > 0.8f && share(fa, img) < 0.2f, "settled again: one shadow, where the box is");
    }

    // ── stale 8: a hidden box is shown ──
    std::printf("\nstale %d: a hidden box is shown\n", kStale);
    shown->visible = true;
    frames(2);
    {
        const auto img = luma(renderer.readRGBPixels());
        std::printf("  2 frames on: C %.2f\n", share(fc, img));
        check(share(fc, img) > 0.6f, "its shadow is there at once");
    }
    frames(50);
    {
        const auto img = luma(renderer.readRGBPixels());
        std::printf("  50 frames on: C %.2f\n", share(fc, img));
        check(share(fc, img) > 0.8f, "and stays");
    }

    // ── stale 0: a settled box moves ──
    std::printf("\nstale 0: a settled box moves\n");
    renderer.setShadowCacheStaleFrames(0u);
    frames(60);
    mover->position.x = ax;
    frames(4);
    {
        const auto img = luma(renderer.readRGBPixels());
        std::printf("  4 frames on: A %.2f, B %.2f\n", share(fa, img), share(fb, img));
        check(share(fa, img) > 0.6f, "its new shadow is there at once");
        check(share(fb, img) < 0.3f, "it leaves no shadow behind");
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
