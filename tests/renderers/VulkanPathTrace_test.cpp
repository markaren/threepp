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
//   [4] fog fades the far floor toward the fog colour;
//   [5] the surface the G-buffer shades: a sphere takes its vertex colours,
//       and a card lit from behind glows through only when translucent;
//   [6] the frame's camera and lamps: a picture of its own extent (larger
//       than the frame, traced in bands), the physical camera's exposure,
//       a thin lens, the lens distortion, and a spot light's beam in clear
//       air;
//   [7] no validation errors.
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

    // ── 5. The surface the G-buffer shades ──────────────────────────────────
    // A sphere whose colour is in its vertices (the GeometryDesc's colour
    // buffer), and a card lit from behind that only a translucent material
    // shows from the front.
    std::printf("\n[5] vertex colours, and a translucent card lit from behind\n");
    {
        Scene s2;
        auto white     = constantEnv(1.f, 1.f, 1.f);
        s2.environment = white;
        s2.background  = white;
        auto geom      = SphereGeometry::create(1.f, 48, 32);
        const int nv   = geom->getAttribute<float>("position")->count();
        std::vector<float> cols(std::size_t(nv) * 3);
        for (int i = 0; i < nv; ++i) {
            cols[std::size_t(i) * 3 + 0] = 1.f;
            cols[std::size_t(i) * 3 + 1] = 0.25f;
            cols[std::size_t(i) * 3 + 2] = 0.25f;
        }
        geom->setAttribute("color", FloatBufferAttribute::create(cols, 3));
        auto ballMat          = MeshStandardMaterial::create();
        ballMat->color        = Color(0.5f, 0.5f, 0.5f);
        ballMat->roughness    = 1.f;
        ballMat->vertexColors = true;
        auto ball             = Mesh::create(geom, ballMat);
        s2.add(ball);
        renderer.render(s2, *camera);
        VulkanRenderer::PathTraceParams p;
        p.samplesPerFrame = 8;
        p.maxBounces      = 4;
        std::vector<float> rgb;
        int w = 0, h = 0;
        bool ok = renderer.pathTrace(s2, *camera, 32, rgb, w, h, p);
        check(ok && w == kW && h == kH, "a trace of the vertex-coloured sphere");
        if (ok && w == kW && h == kH) {
            const Rgb c = meanLinear(rgb, kW, kW / 2 - 10, kW / 2 + 10, kH / 2 - 10, kH / 2 + 10);
            std::printf("  sphere centre (%.3f %.3f %.3f)\n", c.r, c.g, c.b);
            check(c.r > 2.5 * c.g && c.r > 2.5 * c.b, "the sphere takes its vertices' red");
        }
        s2.remove(*ball);

        auto black     = constantEnv(0.f, 0.f, 0.f);
        s2.environment = black;
        s2.background  = black;
        auto back      = DirectionalLight::create(Color(1.f, 1.f, 1.f), 3.f);
        back->position.set(0, 0, -5);// behind the card, lighting its far face
        s2.add(back);
        // The card faces the camera; the light reaches only its back.
        const auto cardLum = [&](float translucency) {
            auto mat               = MeshStandardMaterial::create();
            mat->color             = Color(0.2f, 0.6f, 0.2f);
            mat->roughness         = 1.f;
            mat->side              = Side::Double;
            mat->translucency      = translucency;
            mat->translucencyColor = Color(1.f, 1.f, 1.f);
            auto card              = Mesh::create(PlaneGeometry::create(2.f, 2.f), mat);
            s2.add(card);
            renderer.render(s2, *camera);
            std::vector<float> img;
            int iw = 0, ih = 0;
            Rgb c;
            if (renderer.pathTrace(s2, *camera, 32, img, iw, ih, p) && iw == kW && ih == kH)
                c = meanLinear(img, kW, kW / 2 - 10, kW / 2 + 10, kH / 2 - 10, kH / 2 + 10);
            s2.remove(*card);
            return c;
        };
        const Rgb off = cardLum(0.f);
        const Rgb on  = cardLum(1.f);
        std::printf("  card opaque (%.3f %.3f %.3f) translucent (%.3f %.3f %.3f)\n",
                    off.r, off.g, off.b, on.r, on.g, on.b);
        check(off.lum() < 0.01, "an opaque card is dark on its unlit face");
        check(on.g > 0.05 && on.g > 2.0 * on.r, "a translucent one glows through, in its own green");
    }

    // ── 6. The frame's camera and lamps; a picture of its own extent ────────
    std::printf("\n[6] the frame's camera and lamps; a picture larger than the frame\n");
    {
        VulkanRenderer::PathTraceParams p;
        p.samplesPerFrame = 8;
        p.maxBounces      = 3;

        // A picture twice the frame each way: bands of the frame's size, four
        // render() calls a pass, and the same picture as at the frame's extent.
        {
            VulkanRenderer::PathTraceParams big = p;
            big.width  = 2 * kW;
            big.height = 2 * kH;
            renderer.pathTraceBegin(big);
            renderer.render(scene, *camera);
            check(renderer.pathTraceSamples() == 0 && renderer.pathTraceProgress() > 0.0,
                  "one band in: progress, no whole pass yet");
            for (int i = 0; i < 3; ++i) renderer.render(scene, *camera);
            check(renderer.pathTraceSamples() == 8,
                  "four bands make a pass (" + std::to_string(renderer.pathTraceSamples()) + " spp)");
            renderer.pathTraceStop();
            std::vector<float> large, small;
            int lw = 0, lh = 0, sw = 0, sh = 0;
            const bool okL = renderer.pathTrace(scene, *camera, 64, large, lw, lh, big);
            const bool okS = renderer.pathTrace(scene, *camera, 64, small, sw, sh, p);
            check(okL && lw == 2 * kW && lh == 2 * kH, "pathTrace returns it at its own extent (" +
                                                           std::to_string(lw) + "x" + std::to_string(lh) + ")");
            if (okL && okS && lw == 2 * kW && sw == kW) {
                std::vector<float> down(std::size_t(kW) * kH * 3);
                for (int y = 0; y < kH; ++y)
                    for (int x = 0; x < kW; ++x)
                        for (int c = 0; c < 3; ++c) {
                            float s = 0.f;
                            for (int q = 0; q < 4; ++q)
                                s += large[(std::size_t(2 * y + (q >> 1)) * lw + 2 * x + (q & 1)) * 3 + c];
                            down[(std::size_t(y) * kW + x) * 3 + c] = 0.25f * s;
                        }
                const double db = psnr(small, down, false, kW, kH);
                std::printf("  2x picture, box-filtered down, vs the frame-sized one: %.2f dB\n", db);
                check(db > 28.0, "the same picture, at twice the detail (PSNR > 28 dB)");
            }
        }

        // The frame's exposure: a physical camera's (ISO 200 at f/1, 1 s:
        // EV100 = -1, x1.67) brightens the trace as it brightens the frame.
        {
            const auto meanBytes = [](const std::vector<unsigned char>& b) {
                double s = 0;
                for (auto v : b) s += srgbToLinear(v);
                return b.empty() ? 0.0 : s / double(b.size());
            };
            const auto tracedBytes = [&] {
                std::vector<unsigned char> b;
                int bw = 0, bh = 0;
                renderer.pathTraceBegin(p);
                for (int i = 0; i < 4; ++i) renderer.render(scene, *camera);
                renderer.pathTraceReadRGB(b, bw, bh);
                renderer.pathTraceStop();
                return b;
            };
            const double plain = meanBytes(tracedBytes());
            renderer.setPhysicalCamera(true);
            renderer.setCameraExposure(1.f, 1.f, 200.f);
            for (int i = 0; i < 30; ++i) renderer.render(scene, *camera);
            const double frame = meanBytes(renderer.readRGBPixels());
            const double phys  = meanBytes(tracedBytes());
            renderer.setPhysicalCamera(false);
            std::printf("  mean linear: trace %.3f, under the physical camera %.3f, its frame %.3f\n", plain, phys, frame);
            check(phys > 1.3 * plain, "the physical camera's exposure reaches the trace");
            check(phys > 0.7 * frame && phys < 1.4 * frame, "and agrees with the frame's");
        }

        // A thin lens: focused far in front of the sphere at f/1 its edge
        // spreads over a dozen pixels; focused on it, it stays sharp. The
        // edge is the sphere's red against the floor's grey (r - g), its
        // steepest step over a row-averaged profile across both sides.
        {
            const auto edgeStep = [&](bool dof, float focus) {
                renderer.setDepthOfField(dof);
                renderer.setFocusDistance(focus);
                renderer.setCameraExposure(1.f, 1.f / 125.f, 100.f);
                std::vector<float> img;
                int iw = 0, ih = 0;
                double best = 0;
                if (renderer.pathTrace(scene, *camera, 64, img, iw, ih, p) && iw == kW) {
                    std::vector<double> prof(kW, 0.0);
                    for (int y = kH / 2 - 10; y < kH / 2 + 10; ++y)
                        for (int x = 0; x < kW; ++x) {
                            const float* q = &img[(std::size_t(y) * kW + x) * 3];
                            prof[x] += (q[0] - q[1]) / 20.0;
                        }
                    for (int x = 50; x < kW - 50; ++x) best = std::max(best, std::abs(prof[x + 1] - prof[x]));
                }
                renderer.setDepthOfField(false);
                return best;
            };
            const double pin    = edgeStep(false, 10.f);
            const double blur   = edgeStep(true, 0.4f);
            const double sharp  = edgeStep(true, camera->position.length() - 1.f);
            std::printf("  steepest edge step: pinhole %.3f, focused short %.3f, focused on the sphere %.3f\n",
                        pin, blur, sharp);
            check(blur < 0.5 * pin, "out of focus, the sphere's edge spreads");
            check(sharp > 0.6 * pin, "in focus, it stays sharp");
        }

        // Lens distortion, traced: a grid of dark bars on a white wall, whose
        // straight lines a barrel lens bends most toward the edges. The trace
        // agrees with the frame's warped picture far better with the lens
        // than without it.
        {
            Scene s5;
            auto grey      = constantEnv(0.8f, 0.8f, 0.8f);
            s5.environment = grey;
            s5.background  = grey;
            auto wallMat       = MeshStandardMaterial::create();
            wallMat->color     = Color(0.9f, 0.9f, 0.9f);
            wallMat->roughness = 1.f;
            s5.add(Mesh::create(PlaneGeometry::create(12.f, 9.f), wallMat));
            auto barMat       = MeshStandardMaterial::create();
            barMat->color     = Color(0.02f, 0.02f, 0.02f);
            barMat->roughness = 1.f;
            for (int i = -4; i <= 4; ++i) {
                auto v        = Mesh::create(BoxGeometry::create(0.08f, 9.f, 0.02f), barMat);
                v->position.x = 0.7f * float(i);
                s5.add(v);
                auto hz        = Mesh::create(BoxGeometry::create(12.f, 0.08f, 0.02f), barMat);
                hz->position.y = 0.55f * float(i);
                s5.add(hz);
            }
            auto cam5 = PerspectiveCamera::create(50, float(kW) / float(kH), 0.1f, 100.f);
            cam5->position.set(0, 0, 4);
            cam5->lookAt(0, 0, 0);
            cam5->updateMatrixWorld();
            auto& scene  = s5;
            auto& camera = cam5;
            for (int i = 0; i < 2; ++i) renderer.render(scene, *camera);
            std::vector<float> straight;
            int w = 0, h = 0;
            renderer.pathTrace(scene, *camera, 32, straight, w, h, p);
            LensDistortion lens;
            lens.model = LensModel::BrownConrady;
            lens.k1    = -0.3f;
            lens.k2    = 0.08f;
            renderer.setLensDistortion(lens);
            renderer.setLensOverscan(1.3f);
            for (int i = 0; i < 30; ++i) renderer.render(scene, *camera);
            const std::vector<float> frame = toLinear(renderer.readRGBPixels());
            std::vector<float> bent;
            const bool ok = renderer.pathTrace(scene, *camera, 32, bent, w, h, p);
            renderer.setLensDistortion(LensDistortion{});
            renderer.setLensOverscan(1.f);
            if (ok && bent.size() == frame.size() && straight.size() == frame.size()) {
                const double withLens = psnr(frame, bent, false, kW, kH);
                const double without  = psnr(frame, straight, false, kW, kH);
                std::printf("  vs the frame's barrel-warped picture: traced with the lens %.2f dB, without %.2f dB\n",
                            withLens, without);
                check(withLens > without + 2.0, "the trace bends with the frame's lens");
            } else {
                check(false, "traces with and without the lens");
            }
        }

        // A spot light's beam in clear air (setDeferredVolumetrics): none
        // without the beam density, about the frame's with it, and cut where a
        // slab stands in the cone, which the frame's beam march does not do.
        {
            Scene s4;
            auto black     = constantEnv(0.f, 0.f, 0.f);
            s4.environment = black;
            s4.background  = black;
            auto spot      = SpotLight::create(Color(1.f, 1.f, 1.f), 150.f, 20.f, 0.3f, 0.f, 2.f);
            spot->position.set(0, 4, 0);
            s4.add(spot);
            auto cam4 = PerspectiveCamera::create(50, float(kW) / float(kH), 0.1f, 100.f);
            cam4->position.set(0, 0, 6);
            cam4->lookAt(0, 0, 0);
            cam4->updateMatrixWorld();
            const auto beam = [&](const std::vector<float>& img) {
                return meanLinear(img, kW, kW / 2 - 4, kW / 2 + 4, kH / 2 - 15, kH / 2 + 15);
            };
            const auto traced = [&] {
                std::vector<float> img;
                int iw = 0, ih = 0;
                for (int i = 0; i < 2; ++i) renderer.render(s4, *cam4);
                return renderer.pathTrace(s4, *cam4, 64, img, iw, ih, p) ? beam(img) : Rgb{};
            };
            renderer.setDeferredVolumetrics(0.f);
            const Rgb none = traced();
            renderer.setDeferredVolumetrics(0.15f, 0.3f);
            for (int i = 0; i < 30; ++i) renderer.render(s4, *cam4);
            const Rgb frame = beam(toLinear(renderer.readRGBPixels()));
            const Rgb lit   = traced();
            auto slabMat       = MeshStandardMaterial::create();
            slabMat->roughness = 1.f;
            auto slab          = Mesh::create(BoxGeometry::create(2.f, 0.2f, 2.f), slabMat);
            slab->position.y   = 2.f;
            s4.add(slab);
            const Rgb cut = traced();
            renderer.setDeferredVolumetrics(0.f);
            std::printf("  beam: no density %.4f, frame %.4f, traced %.4f, under a slab %.4f\n",
                        none.lum(), frame.lum(), lit.lum(), cut.lum());
            check(none.lum() < 1e-3, "no beam without a medium");
            check(lit.lum() > 0.01, "the beam is traced in clear air");
            check(lit.lum() > 0.5 * frame.lum() && lit.lum() < 2.0 * frame.lum(), "about as bright as the frame's");
            check(cut.lum() < 0.3 * lit.lum(), "and shadowed by what stands in the cone");
        }

        // A dome port: the camera shut in a closed 0.6 m box sees its black
        // inside; with a dome of 1 m about the lens every ray starts outside
        // the box and sees the white environment.
        {
            Scene s6;
            auto white     = constantEnv(1.f, 1.f, 1.f);
            s6.environment = white;
            s6.background  = white;
            auto boxMat       = MeshStandardMaterial::create();
            boxMat->color     = Color(0.5f, 0.5f, 0.5f);
            boxMat->roughness = 1.f;
            boxMat->side      = Side::Double;
            s6.add(Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), boxMat));
            auto cam6 = PerspectiveCamera::create(50, float(kW) / float(kH), 0.01f, 100.f);
            cam6->position.set(0, 0, 0);
            cam6->lookAt(0, 0, -1);
            cam6->updateMatrixWorld();
            const auto centre = [&](float dome) {
                VulkanRenderer::PathTraceParams q = p;
                q.domePort = dome;
                std::vector<float> img;
                int iw = 0, ih = 0;
                renderer.render(s6, *cam6);
                return renderer.pathTrace(s6, *cam6, 16, img, iw, ih, q) && iw == kW
                               ? meanLinear(img, kW, kW / 2 - 10, kW / 2 + 10, kH / 2 - 10, kH / 2 + 10)
                               : Rgb{};
            };
            const Rgb shut = centre(0.f);
            const Rgb dome = centre(1.f);
            std::printf("  inside a closed box: no dome %.3f, a 1 m dome %.3f\n", shut.lum(), dome.lum());
            check(shut.lum() < 0.02, "shut in the box, the camera sees its dark inside");
            check(dome.lum() > 0.9, "a dome port larger than the box starts every ray outside it");
        }
    }

    // ── 7. Validation ───────────────────────────────────────────────────────
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
