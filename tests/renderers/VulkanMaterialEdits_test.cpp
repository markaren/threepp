// VulkanMaterialEdits_test - a material edited after its first frame shows
// on screen without Material::needsUpdate(), as it does under the GL renderer.
//
// The GL renderer reads a material's uniforms every time it draws it; the
// Vulkan renderer derived a material's GPU record only when version() moved,
// so an edit needed needsUpdate() here and nowhere else. It now reads every
// material of the scene once a frame and needsUpdate()s the changed ones
// itself (VulkanRenderer::Impl::pollMaterialEdits).
//
// The scene: a plane filling the frame, camera straight on, no lights and a
// black environment, so a pixel is the material's colour (unlit) or emission.
// Every edit below is made with NO needsUpdate():
//
//   1. An unlit material's colour.
//   2. A lit material's emissive colour and intensity.
//   3. A map's offset (its uv transform).
//   4. A map swapped for another (the rebuild path).
//   5. A material edited while its mesh is hidden shows the edit when shown.
//   6. needsUpdate() still works (the edit and the call together).
//   7. No validation errors.
//
// ctest runs it twice: as is (ray traced where the GPU can) and with
// THREEPP_VULKAN_NO_RT=1. Plain exit-code program, like the other renderer
// tests here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include "vulkan_test_util.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 160, kH = 120;

    using vktest::check;
    using vktest::failures;
    using vktest::setEnv;
    using vktest::constantEnv;

    // A map whose left half (u < 0.5) is `left` and right half `right`,
    // nearest-filtered so both halves are flat where they are measured.
    std::shared_ptr<Texture> splitMap(unsigned char left, unsigned char right) {
        constexpr int W = 64, H = 4;
        std::vector<unsigned char> data(static_cast<size_t>(W) * H * 4, 255);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                unsigned char* p = &data[(static_cast<size_t>(y) * W + x) * 4];
                p[0] = p[1] = p[2] = x < W / 2 ? left : right;
            }
        auto tex = Texture::create(Image{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H)});
        tex->format = Format::RGBA;
        tex->colorSpace = ColorSpace::Linear;
        tex->magFilter = Filter::Nearest;
        tex->minFilter = Filter::Nearest;
        tex->generateMipmaps = false;
        tex->needsUpdate();
        return tex;
    }

    // The mean RGB (0..255) of a box around a point of the frame, in
    // fractions of its width and height.
    std::array<double, 3> meanRgb(const std::vector<unsigned char>& px, double fx, double fy) {
        std::array<double, 3> s{};
        if (px.size() != std::size_t(kW) * kH * 3) return s;
        const int cx = static_cast<int>(fx * kW), cy = static_cast<int>(fy * kH);
        int n = 0;
        for (int y = cy - 6; y < cy + 6; ++y)
            for (int x = cx - 6; x < cx + 6; ++x) {
                const unsigned char* p = &px[(std::size_t(y) * kW + x) * 3];
                for (int c = 0; c < 3; ++c) s[c] += p[c];
                ++n;
            }
        for (double& c : s) c /= n;
        return s;
    }

    // A few frames, so that the temporal resolve has settled on the edit;
    // the edit itself lands on the first of them.
    std::vector<unsigned char> settle(VulkanRenderer& r, Scene& scene, Camera& cam) {
        for (int i = 0; i < 12; ++i) r.render(scene, cam);
        return r.readRGBPixels();
    }

    void print(const char* what, const std::array<double, 3>& c) {
        std::printf("  %-22s %6.1f %6.1f %6.1f\n", what, c[0], c[1], c[2]);
    }

    bool isRed(const std::array<double, 3>& c) { return c[0] > 150 && c[1] < 60 && c[2] < 60; }
    bool isGreen(const std::array<double, 3>& c) { return c[1] > 150 && c[0] < 60 && c[2] < 60; }
    bool isDark(const std::array<double, 3>& c) { return c[0] < 40 && c[1] < 40 && c[2] < 40; }
    bool isWhite(const std::array<double, 3>& c) { return c[0] > 200 && c[1] > 200 && c[2] > 200; }

}// namespace

int main() {
    std::printf("VulkanMaterialEdits_test\n");

    auto vk = vktest::openHeadlessVulkan("VulkanMaterialEdits_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();
    std::printf("mode: %s\n", renderer.rayTracingSupported() ? "ray traced" : "no ray tracing");

    vktest::pinNativeOutput(renderer);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;
    scene.environment = constantEnv(0.f);
    auto camera = PerspectiveCamera::create(60, float(kW) / float(kH), 0.1f, 100.f);
    camera->position.set(0, 0, 2);
    camera->lookAt(Vector3{0, 0, 0});

    // 6 x 4 m at 2 m: it fills the frame; screen left is u < 0.5.
    auto basic = MeshBasicMaterial::create();
    basic->color = Color(1, 0, 0);
    auto plane = Mesh::create(PlaneGeometry::create(6, 4), basic);
    scene.add(plane);

    // ── 1. An unlit colour ──────────────────────────────────────────────────
    std::printf("\n[1] an unlit material's colour\n");
    {
        const auto before = meanRgb(settle(renderer, scene, *camera), 0.5, 0.5);
        basic->color = Color(0, 1, 0);
        const auto after = meanRgb(settle(renderer, scene, *camera), 0.5, 0.5);
        print("red:", before);
        print("then green:", after);
        check(isRed(before), "the plane draws its colour");
        check(isGreen(after), "the new colour shows without needsUpdate()");
    }

    // ── 2. A lit material's emission ────────────────────────────────────────
    std::printf("\n[2] a lit material's emissive\n");
    auto lit = MeshStandardMaterial::create();
    lit->color = Color(0, 0, 0);
    lit->roughness = 1.f;
    lit->metalness = 0.f;
    lit->emissive = Color(0, 0, 0);
    plane->setMaterial(lit);
    {
        const auto before = meanRgb(settle(renderer, scene, *camera), 0.5, 0.5);
        lit->emissive = Color(1, 0, 0);
        lit->emissiveIntensity = 1.f;
        const auto after = meanRgb(settle(renderer, scene, *camera), 0.5, 0.5);
        print("no emission:", before);
        print("then red emission:", after);
        check(isDark(before), "no light, no emission: black");
        check(isRed(after), "the emission shows without needsUpdate()");
        lit->emissiveIntensity = 0.f;
        const auto off = meanRgb(settle(renderer, scene, *camera), 0.5, 0.5);
        print("then intensity 0:", off);
        check(isDark(off), "and goes with its intensity");
    }

    // ── 3. A map's offset ───────────────────────────────────────────────────
    std::printf("\n[3] a map's offset\n");
    basic->color = Color(1, 1, 1);
    basic->map = splitMap(0, 255);
    plane->setMaterial(basic);
    {
        const auto px = settle(renderer, scene, *camera);
        const auto left = meanRgb(px, 0.25, 0.5), right = meanRgb(px, 0.75, 0.5);
        print("left:", left);
        print("right:", right);
        check(isDark(left) && isWhite(right), "the map's halves show");
        // u + 0.5: the left half of the plane samples the map's white half.
        basic->map->offset.x = 0.5f;
        const auto shifted = meanRgb(settle(renderer, scene, *camera), 0.25, 0.5);
        print("left, offset 0.5:", shifted);
        check(isWhite(shifted), "the offset shows without needsUpdate()");
        basic->map->offset.x = 0.f;
    }

    // ── 4. A map swapped ────────────────────────────────────────────────────
    std::printf("\n[4] a map swapped for another\n");
    {
        basic->map = splitMap(255, 255);
        const auto white = meanRgb(settle(renderer, scene, *camera), 0.25, 0.5);
        print("left, all-white map:", white);
        check(isWhite(white), "the new map shows without needsUpdate()");
        basic->map = nullptr;
        basic->color = Color(0, 1, 0);
        const auto none = meanRgb(settle(renderer, scene, *camera), 0.25, 0.5);
        print("left, no map:", none);
        check(isGreen(none), "and so does taking it away");
    }

    // ── 5. Edited while hidden ──────────────────────────────────────────────
    std::printf("\n[5] a material edited while its mesh is hidden\n");
    {
        plane->visible = false;
        settle(renderer, scene, *camera);
        basic->color = Color(1, 0, 0);
        settle(renderer, scene, *camera);
        plane->visible = true;
        const auto shown = meanRgb(settle(renderer, scene, *camera), 0.5, 0.5);
        print("shown again:", shown);
        check(isRed(shown), "it shows the colour it was given while hidden");
    }

    // ── 6. needsUpdate() as before ──────────────────────────────────────────
    std::printf("\n[6] an edit with needsUpdate()\n");
    {
        basic->color = Color(0, 1, 0);
        basic->needsUpdate();
        const auto after = meanRgb(settle(renderer, scene, *camera), 0.5, 0.5);
        print("green:", after);
        check(isGreen(after), "the edit shows");
    }

    // ── 7. Validation ───────────────────────────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] no validation layer (THREEPP_VULKAN_VALIDATION=1 asks for it): the zero-error\n"
                    "       check is skipped.\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "the edits provoke no validation errors");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
