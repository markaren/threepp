// VulkanAoMap_test - a material's aoMap (glTF occlusionTexture) darkens the
// indirect light of the surface and nothing else.
//
// The deferred renderer uploaded the aoMap and never read it: gbuffer.frag
// now samples its R channel into the UV attachment's .a (as 1 - ao, the
// glTF/three.js ao = 1 + (r - 1) x aoMapIntensity), and the deferred shade
// takes it off the diffuse indirect, the scene ambient and the env specular
// (three.js's computeSpecularOcclusion), never off a light.
//
// The scene: a large plane facing the camera, white, roughness 1, whose aoMap
// is 0 on its left half and 1 on its right; the camera sees the seam in the
// middle of the frame.
//
//   1. Indirect light only (a uniform grey environment, no lights): at
//      intensity 1 the left half is black next to the right; at 0.5 it is
//      half as bright; at 0 the halves match.
//   2. Direct light only (a black environment and a sun): the halves match
//      at intensity 1, so the sun never sees the map.
//   3. A map of 1 everywhere renders as no map at all.
//   4. An unlit MeshBasicMaterial: the map darkens its colour, which three.js
//      treats as indirect diffuse.
//   5. No validation errors.
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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 320, kH = 240;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    void setEnv(const char* k, const char* v) {
#ifdef _WIN32
        _putenv_s(k, v);
#else
        setenv(k, v, /*overwrite*/ 1);
#endif
    }

    std::shared_ptr<Texture> constantEnv(float v) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 1.f);
        for (size_t i = 0; i < data.size(); i += 4) data[i] = data[i + 1] = data[i + 2] = v;
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    // An occlusion map: R = `left` on the left half (u < 0.5), `right` on the
    // right. 64 texels wide, nearest-filtered, so the seam is one texel of a
    // 6 m plane and both halves are flat where they are measured.
    std::shared_ptr<Texture> splitAoMap(unsigned char left, unsigned char right) {
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

    double srgbToLinear(double c) {
        c /= 255.0;
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    // Mean linear luminance of a rectangle of the frame (x0..x1, y0..y1).
    double meanLum(const std::vector<unsigned char>& px, int x0, int x1, int y0, int y1) {
        double s = 0;
        int n = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const unsigned char* p = &px[(std::size_t(y) * kW + x) * 3];
                s += 0.2126 * srgbToLinear(p[0]) + 0.7152 * srgbToLinear(p[1]) + 0.0722 * srgbToLinear(p[2]);
                ++n;
            }
        return n ? s / n : 0.0;
    }

    struct Halves {
        double left = 0, right = 0;
        double ratio() const { return right > 0 ? left / right : -1.0; }
    };

    // The mean of each half, clear of the seam in the middle and of the
    // frame's edges. The histories restart once the change has landed (a new
    // environment is baked inside a frame), so no GI of the previous setup
    // lingers in the accumulators, and then settle.
    Halves renderHalves(VulkanRenderer& r, Scene& scene, Camera& cam) {
        for (int i = 0; i < 4; ++i) r.render(scene, cam);
        r.resetTemporalHistory();
        for (int i = 0; i < 40; ++i) r.render(scene, cam);
        const auto px = r.readRGBPixels();
        Halves h;
        if (px.size() != std::size_t(kW) * kH * 3) return h;
        h.left = meanLum(px, kW / 10, 4 * kW / 10, kH / 5, 4 * kH / 5);
        h.right = meanLum(px, 6 * kW / 10, 9 * kW / 10, kH / 5, 4 * kH / 5);
        return h;
    }

}// namespace

int main() {
    std::printf("VulkanAoMap_test\n");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    auto vk = vktest::openHeadlessVulkan("VulkanAoMap_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();
    std::printf("mode: %s\n", renderer.rayTracingSupported() ? "ray traced" : "no ray tracing");

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;
    auto camera = PerspectiveCamera::create(60, float(kW) / float(kH), 0.1f, 100.f);
    camera->position.set(0, 0, 2);
    camera->lookAt(Vector3{0, 0, 0});

    // 6 x 4 m at 2 m: the 60-degree frame sees 3.1 x 2.3 m of it, the aoMap's
    // seam (u = 0.5, x = 0) down the middle. Screen left is world -x, u < 0.5.
    auto mat = MeshStandardMaterial::create();
    mat->color = Color(1, 1, 1);
    mat->roughness = 1.f;
    mat->metalness = 0.f;
    mat->aoMap = splitAoMap(0, 255);
    mat->aoMapIntensity = 1.f;
    auto plane = Mesh::create(PlaneGeometry::create(6, 4), mat);
    scene.add(plane);

    const auto setIntensity = [&](float k) {
        mat->aoMapIntensity = k;
        mat->needsUpdate();
    };

    // ── 1. Indirect light only ──────────────────────────────────────────────
    std::printf("\n[1] indirect light only (uniform environment, no lights)\n");
    scene.environment = constantEnv(0.5f);
    {
        const Halves h1 = renderHalves(renderer, scene, *camera);
        std::printf("  intensity 1:   left %.4f  right %.4f  ratio %.3f\n", h1.left, h1.right, h1.ratio());
        check(h1.right > 0.2, "the open half is lit by the environment");
        check(h1.ratio() >= 0.0 && h1.ratio() < 0.05, "intensity 1: the occluded half gets no indirect light");

        setIntensity(0.5f);
        const Halves h2 = renderHalves(renderer, scene, *camera);
        std::printf("  intensity 0.5: left %.4f  right %.4f  ratio %.3f\n", h2.left, h2.right, h2.ratio());
        check(std::abs(h2.ratio() - 0.5) < 0.06, "intensity 0.5: the occluded half gets half (ao = 1 + (r - 1) k)");
        check(std::abs(h2.right - h1.right) < 0.01 * h1.right, "the open half does not depend on the intensity");

        setIntensity(0.f);
        const Halves h3 = renderHalves(renderer, scene, *camera);
        std::printf("  intensity 0:   left %.4f  right %.4f  ratio %.3f\n", h3.left, h3.right, h3.ratio());
        check(std::abs(h3.ratio() - 1.0) < 0.02, "intensity 0: the map does nothing");
        setIntensity(1.f);
    }

    // ── 2. Direct light only ────────────────────────────────────────────────
    std::printf("\n[2] direct light only (black environment, one sun)\n");
    // Lambert under it: albedo x intensity x cos / pi, about 0.58.
    auto sun = DirectionalLight::create(Color(1, 1, 1), 2.f);
    sun->position.set(0.5f, 0.8f, 2.f);
    scene.add(sun);
    scene.environment = constantEnv(0.f);
    {
        const Halves h = renderHalves(renderer, scene, *camera);
        std::printf("  intensity 1: left %.4f  right %.4f  ratio %.3f\n", h.left, h.right, h.ratio());
        check(h.right > 0.2, "the sun lights the plane");
        check(std::abs(h.ratio() - 1.0) < 0.02, "the occluded half is lit by the sun as fully as the open one");
    }
    scene.remove(*sun);

    // ── 3. A map of 1 is no map ─────────────────────────────────────────────
    std::printf("\n[3] a map of 1 everywhere renders as no map\n");
    scene.environment = constantEnv(0.5f);
    {
        mat->aoMap = splitAoMap(255, 255);
        mat->needsUpdate();
        const Halves white = renderHalves(renderer, scene, *camera);
        mat->aoMap = nullptr;
        mat->needsUpdate();
        const Halves none = renderHalves(renderer, scene, *camera);
        std::printf("  map of 1: %.4f %.4f   no map: %.4f %.4f\n", white.left, white.right, none.left, none.right);
        check(std::abs(white.left - none.left) < 0.002 && std::abs(white.right - none.right) < 0.002,
              "a map of 1 and no map shade the same");
        mat->aoMap = splitAoMap(0, 255);
        mat->needsUpdate();
    }

    // ── 4. Unlit ────────────────────────────────────────────────────────────
    std::printf("\n[4] an unlit MeshBasicMaterial\n");
    {
        auto basic = MeshBasicMaterial::create();
        basic->color = Color(0.5f, 0.5f, 0.5f);
        basic->aoMap = splitAoMap(0, 255);
        plane->setMaterial(basic);
        const Halves h = renderHalves(renderer, scene, *camera);
        std::printf("  left %.4f  right %.4f  ratio %.3f\n", h.left, h.right, h.ratio());
        check(h.right > 0.1, "the open half shows the colour");
        check(h.ratio() >= 0.0 && h.ratio() < 0.05, "the occluded half is dark, as three.js shades it");
        plane->setMaterial(mat);
    }

    // ── 5. Validation ───────────────────────────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed: the zero-error\n"
                    "       check is skipped.\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "the aoMap path provokes no validation errors");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
