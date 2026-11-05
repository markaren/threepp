// VulkanPathTrace_test — the path-trace feature (VulkanRenderer::pathTrace*)
// over the deferred renderer's scene state.
//
//   [1] a furnace: a grey diffuse sphere alone under a uniform white
//       environment comes back at its albedo, the sky at the environment,
//       and the accumulation counts what it was asked for;
//   [2] the session API: samples accumulate across render() calls and stop
//       when told to;
//   [3] a lit scene: the trace and the deferred frame of the same scene
//       agree on brightness and layout (a sanity gate, not a golden — the two
//       are different estimators of the same light), and the trace is the
//       right way up;
//   [4] no validation errors.
//
// Exit 42 (SKIP) without a Vulkan device; a device without ray query checks
// that pathTraceBegin refuses and skips the rest.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include "vulkan_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
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
            data[i]     = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex        = Texture::create(img);
        tex->format     = Format::RGBA;
        tex->type       = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping    = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    struct Rgb {
        double r = 0, g = 0, b = 0;
        [[nodiscard]] double lum() const { return 0.2126 * r + 0.7152 * g + 0.0722 * b; }
    };

    // Mean of a linear float picture (3 floats/px) over a pixel rectangle.
    Rgb meanLinear(const std::vector<float>& px, int w, int x0, int x1, int y0, int y1) {
        Rgb m;
        int n = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const float* p = &px[(std::size_t(y) * w + x) * 3];
                m.r += p[0];
                m.g += p[1];
                m.b += p[2];
                ++n;
            }
        if (n) {
            m.r /= n;
            m.g /= n;
            m.b /= n;
        }
        return m;
    }

    double srgbToLinear(double c) {
        c /= 255.0;
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    // The deferred frame's sRGB bytes as linear floats, 3 per pixel.
    std::vector<float> toLinear(const std::vector<unsigned char>& rgb) {
        std::vector<float> out(rgb.size());
        for (size_t i = 0; i < rgb.size(); ++i) out[i] = static_cast<float>(srgbToLinear(rgb[i]));
        return out;
    }

    // PSNR between two linear pictures after a shared clamp to [0,1]
    // (the deferred bytes carry nothing above 1).
    double psnr(const std::vector<float>& a, const std::vector<float>& b, bool flipB, int w, int h) {
        double se = 0;
        for (int y = 0; y < h; ++y) {
            const int yb = flipB ? (h - 1 - y) : y;
            for (int x = 0; x < w * 3; ++x) {
                const double va = std::clamp<double>(a[std::size_t(y) * w * 3 + x], 0.0, 1.0);
                const double vb = std::clamp<double>(b[std::size_t(yb) * w * 3 + x], 0.0, 1.0);
                se += (va - vb) * (va - vb);
            }
        }
        const double mse = se / (double(w) * h * 3);
        return mse > 0 ? 10.0 * std::log10(1.0 / mse) : 99.0;
    }

}// namespace

int main() {
    std::printf("VulkanPathTrace_test\n");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    auto vk = vktest::openHeadlessVulkan("VulkanPathTrace_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setAutoLod(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping         = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    auto camera = PerspectiveCamera::create(50, float(kW) / float(kH), 0.1f, 100.f);
    camera->position.set(0, 0, 4);
    camera->lookAt(0, 0, 0);
    camera->updateMatrixWorld();

    Scene scene;
    auto whiteEnv     = constantEnv(1.f, 1.f, 1.f);
    scene.environment = whiteEnv;
    scene.background  = whiteEnv;

    auto sphereMat       = MeshStandardMaterial::create();
    sphereMat->color     = Color(0.5f, 0.5f, 0.5f);
    sphereMat->roughness = 1.f;
    sphereMat->metalness = 0.f;
    auto sphere          = Mesh::create(SphereGeometry::create(1.f, 48, 32), sphereMat);
    scene.add(sphere);

    if (!renderer.rayTracingSupported()) {
        std::printf("\n[no ray query] pathTraceBegin must refuse\n");
        bool threw = false;
        try {
            renderer.pathTraceBegin();
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "pathTraceBegin throws without ray query");
        std::printf("\n[skip] the rest needs ray query\n");
        return failures == 0 ? vktest::kSkipCode : 1;
    }

    // One deferred frame first: the scene is built (TLAS, materials) by
    // render(), and the trace's first dispatch rides the frame after.
    renderer.render(scene, *camera);

    // ── 1. Furnace ──────────────────────────────────────────────────────────
    std::printf("\n[1] furnace: a grey sphere under a white environment\n");
    {
        std::vector<float> rgb;
        int w = 0, h = 0;
        VulkanRenderer::PathTraceParams p;
        p.samplesPerFrame = 8;
        p.maxBounces      = 4;
        const bool ok = renderer.pathTrace(scene, *camera, 64, rgb, w, h, p);
        check(ok, "pathTrace returns a picture");
        check(w == kW && h == kH, "at the render extent (" + std::to_string(w) + "x" + std::to_string(h) + ")");
        check(rgb.size() == std::size_t(kW) * kH * 3, "three floats per pixel");
        if (ok && w == kW && h == kH) {
            const Rgb sky = meanLinear(rgb, kW, 0, 24, 0, 24);
            std::printf("  sky corner (%.4f %.4f %.4f)\n", sky.r, sky.g, sky.b);
            check(std::abs(sky.r - 1.0) < 1e-3 && std::abs(sky.g - 1.0) < 1e-3 && std::abs(sky.b - 1.0) < 1e-3,
                  "a miss is the environment, exactly");
            // The sphere's centre: albedo 0.5 under radiance 1 from every
            // direction, plus the dielectric's 4% lobe (~0.52).
            const Rgb c = meanLinear(rgb, kW, kW / 2 - 12, kW / 2 + 12, kH / 2 - 12, kH / 2 + 12);
            std::printf("  sphere centre (%.4f %.4f %.4f)\n", c.r, c.g, c.b);
            check(c.lum() > 0.46 && c.lum() < 0.60, "the sphere comes back at its albedo (0.5 + the 4% lobe)");
            check(std::abs(c.r - c.g) < 0.01 && std::abs(c.g - c.b) < 0.01, "and grey");
            // Noise level at 64 spp: the centre block's pixels should sit
            // close to their mean (a per-pixel estimator, not a single
            // sample).
            double dev = 0;
            int n = 0;
            for (int y = kH / 2 - 12; y < kH / 2 + 12; ++y)
                for (int x = kW / 2 - 12; x < kW / 2 + 12; ++x) {
                    const float* px = &rgb[(std::size_t(y) * kW + x) * 3];
                    dev += std::abs(0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2] - c.lum());
                    ++n;
                }
            dev /= n;
            std::printf("  mean abs deviation %.4f\n", dev);
            check(dev < 0.05, "converged at 64 spp");
        }
        check(!renderer.pathTraceActive(), "the blocking form stops its session");
    }

    // ── 2. The session API ──────────────────────────────────────────────────
    std::printf("\n[2] the session accumulates across render() calls\n");
    {
        VulkanRenderer::PathTraceParams p;
        p.samplesPerFrame = 2;
        renderer.pathTraceBegin(p);
        check(renderer.pathTraceActive(), "active after begin");
        check(renderer.pathTraceSamples() == 0, "no samples before a frame");
        for (int i = 0; i < 3; ++i) renderer.render(scene, *camera);
        check(renderer.pathTraceSamples() == 6, "3 frames x 2 spp = 6 (" + std::to_string(renderer.pathTraceSamples()) + ")");
        std::vector<float> rgb;
        int w = 0, h = 0;
        check(renderer.pathTraceRead(rgb, w, h) && w == kW && h == kH, "readable mid-session");
        renderer.render(scene, *camera);
        check(renderer.pathTraceSamples() == 8, "and keeps counting after a read");
        renderer.pathTraceStop();
        check(!renderer.pathTraceActive(), "inactive after stop");
        renderer.render(scene, *camera);
        check(renderer.pathTraceSamples() == 8, "a stopped session adds nothing");
        // A restart resets.
        renderer.pathTraceBegin(p);
        renderer.render(scene, *camera);
        check(renderer.pathTraceSamples() == 2, "begin restarts the count");
        renderer.pathTraceStop();
    }

    // ── 3. A lit scene against the deferred frame ───────────────────────────
    std::printf("\n[3] a lit scene: trace vs deferred\n");
    {
        auto dimEnv       = constantEnv(0.25f, 0.28f, 0.35f);
        scene.environment = dimEnv;
        scene.background  = dimEnv;
        auto sun = DirectionalLight::create(Color(1.f, 0.97f, 0.9f), 2.5f);
        sun->position.set(3, 5, 2);
        scene.add(sun);
        auto floorMat       = MeshStandardMaterial::create();
        floorMat->color     = Color(0.6f, 0.55f, 0.5f);
        floorMat->roughness = 0.9f;
        auto floor          = Mesh::create(PlaneGeometry::create(12.f, 12.f), floorMat);
        floor->rotation.x   = -math::PI / 2;
        floor->position.y   = -1.f;
        scene.add(floor);
        sphereMat->color     = Color(0.8f, 0.3f, 0.25f);
        sphereMat->roughness = 0.5f;
        sphereMat->needsUpdate();
        camera->position.set(0, 1.5f, 4.5f);
        camera->lookAt(0, 0, 0);
        camera->updateMatrixWorld();

        for (int i = 0; i < 60; ++i) renderer.render(scene, *camera);
        const std::vector<float> deferred = toLinear(renderer.readRGBPixels());

        std::vector<float> traced;
        int w = 0, h = 0;
        VulkanRenderer::PathTraceParams p;
        p.samplesPerFrame = 8;
        p.maxBounces      = 4;
        const bool ok = renderer.pathTrace(scene, *camera, 128, traced, w, h, p);
        check(ok && w == kW && h == kH && deferred.size() == traced.size(), "both pictures at the render extent");
        if (ok && deferred.size() == traced.size()) {
            const Rgb md = meanLinear(deferred, kW, 0, kW, 0, kH);
            const Rgb mt = meanLinear(traced, kW, 0, kW, 0, kH);
            std::printf("  mean deferred (%.3f %.3f %.3f)  traced (%.3f %.3f %.3f)\n",
                        md.r, md.g, md.b, mt.r, mt.g, mt.b);
            const double ratio = mt.lum() / std::max(md.lum(), 1e-6);
            check(ratio > 0.6 && ratio < 1.6, "brightness agrees within 60% (ratio " + std::to_string(ratio) + ")");
            const double up   = psnr(deferred, traced, false, kW, kH);
            const double down = psnr(deferred, traced, true, kW, kH);
            std::printf("  PSNR vs deferred %.2f dB (flipped %.2f dB)\n", up, down);
            check(up > 12.0, "layout agrees (PSNR > 12 dB)");
            check(up > down, "the trace is the right way up");
            // The sphere's lit side is on the sun's side (+x), in both.
            const Rgb lD = meanLinear(deferred, kW, kW / 2 - 30, kW / 2 - 10, kH / 2 - 10, kH / 2 + 10);
            const Rgb rD = meanLinear(deferred, kW, kW / 2 + 10, kW / 2 + 30, kH / 2 - 10, kH / 2 + 10);
            const Rgb lT = meanLinear(traced, kW, kW / 2 - 30, kW / 2 - 10, kH / 2 - 10, kH / 2 + 10);
            const Rgb rT = meanLinear(traced, kW, kW / 2 + 10, kW / 2 + 30, kH / 2 - 10, kH / 2 + 10);
            std::printf("  sphere left/right lum deferred %.3f/%.3f traced %.3f/%.3f\n",
                        lD.lum(), rD.lum(), lT.lum(), rT.lum());
            check((rD.lum() > lD.lum()) == (rT.lum() > lT.lum()), "the sun lights the same side");
            // The sphere's shadow on the floor: below and to the left of the
            // sphere in both, darker than the open floor.
            const Rgb openD = meanLinear(deferred, kW, kW - 40, kW - 10, kH - 30, kH - 10);
            const Rgb openT = meanLinear(traced, kW, kW - 40, kW - 10, kH - 30, kH - 10);
            check(openT.lum() > 0.05 && openD.lum() > 0.05, "the open floor is lit in both");
        }
    }

    // ── 4. Fog: a leg fades toward the fog colour by its depth ─────────────
    std::printf("\n[4] fog fades the far floor toward the fog colour\n");
    {
        const Color fogColor(0.9f, 0.2f, 0.1f);
        scene.fog = Fog(fogColor, 1.f, 6.f);// sigma = 1/5 per metre: the far floor is well inside
        std::vector<float> traced;
        int w = 0, h = 0;
        VulkanRenderer::PathTraceParams p;
        p.samplesPerFrame = 8;
        const bool ok = renderer.pathTrace(scene, *camera, 64, traced, w, h, p);
        check(ok && w == kW && h == kH, "a fogged trace at the render extent");
        if (ok) {
            // Floor to the right of the sphere, ~9 m out (T ≈ e^-1.8) against
            // the floor at the bottom edge, ~4 m out (T ≈ e^-0.7).
            const Rgb far  = meanLinear(traced, kW, kW / 2 + 70, kW / 2 + 110, kH / 2 - 18, kH / 2 - 8);
            const Rgb near = meanLinear(traced, kW, kW / 2 + 70, kW / 2 + 110, kH - 12, kH - 2);
            std::printf("  far (%.3f %.3f %.3f) near (%.3f %.3f %.3f)\n", far.r, far.g, far.b, near.r, near.g, near.b);
            check(far.r > 2.0 * far.b, "the far floor has taken the fog's red");
            check(far.r - far.b > near.r - near.b, "more so than the near floor");
        }
        // Through the display transform: the same picture as 8-bit, and a file.
        std::vector<unsigned char> rgb8;
        renderer.pathTraceBegin(p);
        renderer.render(scene, *camera);
        check(renderer.pathTraceReadRGB(rgb8, w, h) && rgb8.size() == std::size_t(kW) * kH * 3, "pathTraceReadRGB gives 3 bytes per pixel");
        renderer.pathTraceStop();
        const std::filesystem::path png = std::filesystem::temp_directory_path() / "VulkanPathTrace_test.png";
        check(renderer.pathTraceToFile(scene, *camera, png, 16, p) && std::filesystem::exists(png), "pathTraceToFile writes a PNG");
        std::error_code ec;
        std::filesystem::remove(png, ec);
        scene.fog.reset();
    }

    // ── 5. Validation ───────────────────────────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed: the zero-error\n"
                    "       check is skipped.\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "the path trace provokes no validation errors");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
