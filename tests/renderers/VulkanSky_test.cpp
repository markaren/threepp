// VulkanSky_test - a threepp::Sky is the environment on the deferred renderer,
// not a mesh.
//
// A Sky is a 1000 m box carrying the Preetham ShaderMaterial the GL renderer
// shades per pixel. The deferred renderer has no ShaderMaterial path, and as a
// mesh the box sealed the scene in a dark room: its inside on every pixel,
// every sun and GI ray stopped by it, every frame black (the report of 18
// October 2026 on examples/lights/directional.cpp). Now the scene gather
// drops the box and bakes the model into the env image (vulkan/SkyBake),
// in place as the sun moves.
//
// The scene is directional.cpp's: the Sky, a sun, a grey plane, a red knot.
//
//   1. No pixel of the frame belongs to the dome: the Ids AOV is 0 (sky)
//      above the horizon, the plane's entry below it, and the frame is lit.
//   2. The sky: blue away from the sun (b > r), brighter toward it than away
//      from it, lit from the model (a texel of the bake agrees with the CPU
//      evaluation of the same direction to within the display quantization).
//   3. Moving the sun moves the sky: after the Sky's sunPosition swings to the
//      opposite azimuth, the bright side swaps, with the same scene (no
//      structural change, the entries unchanged).
//   4. A scene.environment outranks the Sky: with one set, the background is
//      that environment, and the dome still builds no entry.
//   5. Hiding the Sky (visible = false) drops it: the background is the
//      renderer's clear colour again.
//   6. No validation errors.
//
// Plain exit-code program, like the other renderer tests here. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/common/PreethamSky.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include "vulkan_test_util.hpp"

#include <algorithm>
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

    std::shared_ptr<Texture> constantEnv(float r, float g, float b) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 1.f);
        for (size_t i = 0; i < data.size(); i += 4) {
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    // The visible entry id (ids .x) of every pixel.
    std::vector<std::uint32_t> readIds(VulkanRenderer& r) {
        std::vector<std::uint8_t> raw;
        int w = 0, h = 0, bpp = 0;
        if (!r.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, w, h, bpp) || w != kW || h != kH || bpp != 16)
            return {};
        std::vector<std::uint32_t> ids(std::size_t(kW) * kH);
        const auto* p = reinterpret_cast<const std::uint32_t*>(raw.data());
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = p[i * 4];
        return ids;
    }

    struct Rgb {
        double r = 0, g = 0, b = 0;
        double lum() const { return 0.2126 * r + 0.7152 * g + 0.0722 * b; }
    };

    double srgbToLinear(double c) {
        c /= 255.0;
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    // Mean linear colour of a rectangle of the frame (x0..x1, y0..y1).
    Rgb meanRgb(const std::vector<unsigned char>& px, int x0, int x1, int y0, int y1) {
        Rgb m;
        int n = 0;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const unsigned char* p = &px[(std::size_t(y) * kW + x) * 3];
                m.r += srgbToLinear(p[0]);
                m.g += srgbToLinear(p[1]);
                m.b += srgbToLinear(p[2]);
                ++n;
            }
        }
        if (n) {
            m.r /= n;
            m.g /= n;
            m.b /= n;
        }
        return m;
    }

    struct Frame {
        std::vector<unsigned char> rgb;
        std::vector<std::uint32_t> ids;
    };

    Frame renderSettled(VulkanRenderer& r, Scene& scene, Camera& cam, int frames = 12) {
        for (int i = 0; i < frames; ++i) r.render(scene, cam);
        Frame f;
        f.rgb = r.readRGBPixels();
        f.ids = readIds(r);
        return f;
    }

    // The share of pixels in rows y0..y1 whose id is 0 (sky, no entry).
    double skyShare(const std::vector<std::uint32_t>& ids, int y0, int y1) {
        if (ids.empty()) return -1.0;
        int n = 0, sky = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < kW; ++x) {
                ++n;
                if (ids[std::size_t(y) * kW + x] == 0u) ++sky;
            }
        return n ? double(sky) / n : 0.0;
    }

}// namespace

int main() {
    std::printf("VulkanSky_test\n");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    auto vk = vktest::openHeadlessVulkan("VulkanSky_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;
    renderer.shadowMap().enabled = true;

    // directional.cpp's scene. The camera looks level along +z from the
    // origin, 2 m up: the top half of the frame is sky, the bottom the plane.
    Scene scene;
    auto camera = PerspectiveCamera::create(60, float(kW) / float(kH), 0.1f, 1000.f);
    camera->position.set(0, 2, 0);
    camera->lookAt(Vector3{0, 2, 10});

    auto light = DirectionalLight::create();
    light->position.set(150, 50, 150);
    light->castShadow = true;
    scene.add(light);

    auto sky = Sky::create();
    sky->scale.setScalar(1000);
    auto& u = sky->materialAs<ShaderMaterial>()->uniforms;
    u.at("turbidity").value<float>() = 10;
    u.at("rayleigh").value<float>() = 1;
    u.at("mieCoefficient").value<float>() = 0.005f;
    u.at("mieDirectionalG").value<float>() = 0.8f;
    auto& sunPosition = u.at("sunPosition").value<Vector3>();
    // The sun to the camera's RIGHT, 20 degrees up: the right edge of the
    // frame looks 30 degrees from it, the left edge 90 degrees away. (A
    // camera looking along +z has world -x on its right.)
    sunPosition.set(-100, 36.4f, 0);
    scene.add(sky);

    auto plane = Mesh::create(PlaneGeometry::create(100, 100), MeshLambertMaterial::create());
    plane->materialAs<MeshLambertMaterial>()->color = Color::gray;
    plane->materialAs<MeshLambertMaterial>()->side = Side::Double;
    plane->rotateX(math::degToRad(90));
    plane->receiveShadow = true;
    scene.add(plane);

    auto knotMat = MeshStandardMaterial::create();
    knotMat->color = 0xff0000;
    knotMat->roughness = 0.1f;
    knotMat->metalness = 0.1f;
    auto knot = Mesh::create(TorusKnotGeometry::create(0.75f, 0.2f, 128, 64), knotMat);
    knot->position.set(0, 2, 6);
    knot->castShadow = true;
    scene.add(knot);

    // Sky rows: the top quarter of the frame (clear of the horizon and the
    // knot). Left and right thirds: away from and toward the sun.
    const int skyY0 = 4, skyY1 = kH / 4;
    const int third = kW / 3;

    // ── 1. The dome builds no entry, and the frame is lit ───────────────────
    std::printf("\n[1] the dome is dropped, the scene lit\n");
    {
        const Frame f = renderSettled(renderer, scene, *camera);
        check(!f.ids.empty(), "the Ids AOV reads back");
        const double skyTop = skyShare(f.ids, skyY0, skyY1);
        const double skyBottom = skyShare(f.ids, kH - 40, kH - 4);
        std::printf("  sky share: top rows %.3f, bottom rows %.3f\n", skyTop, skyBottom);
        check(skyTop > 0.999, "every pixel above the horizon is sky (no entry of the dome)");
        check(skyBottom < 0.001, "the plane covers the bottom of the frame");
        const Rgb ground = meanRgb(f.rgb, third, 2 * third, kH - 40, kH - 4);
        const Rgb skyMean = meanRgb(f.rgb, 0, kW, skyY0, skyY1);
        std::printf("  mean linear: ground lum %.3f, sky lum %.3f\n", ground.lum(), skyMean.lum());
        check(ground.lum() > 0.02, "the plane is lit (the sun reaches it through where the dome was)");
        check(skyMean.lum() > 0.05, "the sky is lit");
    }

    // ── 2. The sky is the model ─────────────────────────────────────────────
    std::printf("\n[2] the sky is the Preetham model\n");
    Rgb leftA, rightA;
    {
        const Frame f = renderSettled(renderer, scene, *camera, 4);
        leftA = meanRgb(f.rgb, 0, third, skyY0, skyY1);
        rightA = meanRgb(f.rgb, 2 * third, kW, skyY0, skyY1);
        std::printf("  away from the sun (%.3f %.3f %.3f)  toward it (%.3f %.3f %.3f)\n",
                    leftA.r, leftA.g, leftA.b, rightA.r, rightA.g, rightA.b);
        check(leftA.b > leftA.r, "the sky away from the sun is blue");
        check(rightA.lum() > leftA.lum() * 1.15, "the sky toward the sun is brighter");

        // One pixel of the sky against the CPU model: the pixel at the top
        // row's centre looks along the camera's forward axis tilted up by half
        // the vertical FOV; with no tone map and exposure 1, the frame holds
        // sRGB(radiance), so the linear decode is the bake's texel (bilinear
        // between neighbours, which the smooth model keeps within a few %).
        const int px = kW / 2, py = skyY0 + 2;
        const float ndcY = 1.f - 2.f * (py + 0.5f) / float(kH);
        const float tanHalf = std::tan(math::degToRad(30.f));
        float dir[3] = {0.f, ndcY * tanHalf, 1.f};
        const float len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        for (float& d : dir) d /= len;
        float ref[3];
        preethamRadiance(preethamParamsFromSky(*sky), dir, false, ref);
        const Rgb got = meanRgb(f.rgb, px - 1, px + 2, py - 1, py + 2);
        std::printf("  texel (%d,%d): frame (%.4f %.4f %.4f)  model (%.4f %.4f %.4f)\n", px, py, got.r, got.g,
                    got.b, ref[0], ref[1], ref[2]);
        auto close = [](double a, double b) { return std::abs(a - b) <= 0.08 * std::max(a, b) + 0.004; };
        check(close(got.r, ref[0]) && close(got.g, ref[1]) && close(got.b, ref[2]),
              "a sky pixel matches the CPU evaluation of the model (within 8 %)");
    }

    // ── 3. The sun moves the sky in place ───────────────────────────────────
    std::printf("\n[3] the sun moves, the sky follows\n");
    {
        sunPosition.set(100, 36.4f, 0);
        const Frame f = renderSettled(renderer, scene, *camera);
        const Rgb leftB = meanRgb(f.rgb, 0, third, skyY0, skyY1);
        const Rgb rightB = meanRgb(f.rgb, 2 * third, kW, skyY0, skyY1);
        std::printf("  now left (%.3f %.3f %.3f)  right (%.3f %.3f %.3f)\n", leftB.r, leftB.g, leftB.b, rightB.r,
                    rightB.g, rightB.b);
        check(leftB.lum() > rightB.lum() * 1.15, "the bright side of the sky swapped with the sun");
        auto near = [](double a, double b) { return std::abs(a - b) <= 0.06 * std::max(a, b) + 0.004; };
        check(near(leftB.lum(), rightA.lum()) && near(rightB.lum(), leftA.lum()),
              "the mirrored sun gives the mirrored sky");
        check(skyShare(f.ids, skyY0, skyY1) > 0.999, "still no entry above the horizon");
        sunPosition.set(-100, 36.4f, 0);
    }

    // ── 4. A scene.environment outranks the Sky ─────────────────────────────
    std::printf("\n[4] scene.environment wins\n");
    {
        scene.environment = constantEnv(0.2f, 0.5f, 0.1f);
        const Frame f = renderSettled(renderer, scene, *camera);
        const Rgb s = meanRgb(f.rgb, 0, kW, skyY0, skyY1);
        std::printf("  background (%.3f %.3f %.3f)\n", s.r, s.g, s.b);
        auto near = [](double a, double b) { return std::abs(a - b) <= 0.03; };
        check(near(s.r, 0.2) && near(s.g, 0.5) && near(s.b, 0.1), "the background is the environment, not the Sky");
        check(skyShare(f.ids, skyY0, skyY1) > 0.999, "the dome builds no entry either way");
        scene.environment = nullptr;
    }

    // ── 5. A hidden Sky is no sky ───────────────────────────────────────────
    std::printf("\n[5] a hidden Sky is dropped\n");
    {
        sky->visible = false;
        const Frame f = renderSettled(renderer, scene, *camera);
        const Rgb s = meanRgb(f.rgb, 0, kW, skyY0, skyY1);
        std::printf("  background (%.3f %.3f %.3f)\n", s.r, s.g, s.b);
        check(s.lum() < 0.005, "hidden: the background is the clear colour (black)");
        sky->visible = true;
        const Frame g = renderSettled(renderer, scene, *camera);
        const Rgb t = meanRgb(g.rgb, 0, kW, skyY0, skyY1);
        check(t.lum() > 0.05, "shown again: the sky is back");
    }

    // ── 6. Validation ───────────────────────────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed: the zero-error\n"
                    "       check is skipped.\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "the Sky bake provokes no validation errors");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
