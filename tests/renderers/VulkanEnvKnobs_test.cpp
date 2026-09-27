// VulkanEnvKnobs_test — the Vulkan renderer honours three.js's scene
// environment/background knobs the way GLRenderer does (GLEnvironmentKnobs_test
// is the GL twin):
//
//   1. environmentIntensity = 0 takes a metal sphere's whole environment
//      contribution away (it goes black; the background stays);
//   2. backgroundIntensity = 0.5 halves the LINEAR background;
//   3. environmentRotation = 180 deg about Y moves a bright env feature from
//      one side of a mirror sphere to the other.
//
// Plain exit-code program like VulkanGolden_test; exits 42 (CTest "Skipped")
// without a Vulkan/RT GPU. The environment is synthetic (a dim grey equirect
// with one red patch at +X), the env sun is off, and tone mapping is None, so
// the 8-bit readback decodes to linear through the sRGB curve alone.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kSize = 160;
    constexpr int kFrames = 90;
    constexpr int kSkipCode = 42;

    // Grey 0.2 everywhere, red 4.0 within 25 deg of +X (u = 0.5, v = 0.5 in
    // the renderers' equirect convention: u = 0.5 + atan(z, x) / 2pi).
    std::shared_ptr<Texture> patchEnv() {
        constexpr int W = 256, H = 128;
        std::vector<float> data(static_cast<size_t>(W) * H * 4);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const double phi = (x + 0.5) / W * 2 * math::PI - math::PI;// atan(z, x)
                const double lat = (y + 0.5) / H * math::PI - math::PI / 2;  // asin(y)
                const double dx = std::cos(lat) * std::cos(phi);
                const bool red = dx > std::cos(25.0 * math::PI / 180.0);
                float* p = &data[(static_cast<size_t>(y) * W + x) * 4];
                p[0] = red ? 4.f : 0.2f;
                p[1] = 0.2f;
                p[2] = 0.2f;
                p[3] = 1.f;
            }
        }
        auto tex = Texture::create(Image{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0});
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    double lin(unsigned char c) {
        const double s = c / 255.0;
        return s <= 0.04045 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
    }

    struct Reading {
        double sphere = 0;  // mean linear luminance-ish (r+g+b)/3 over the inner disc
        double redLeft = 0; // mean linear (r - g) over the disc's left half
        double redRight = 0;// ... right half
        double background = 0;// mean linear green over a background corner patch
    };

    Reading measure(const std::vector<unsigned char>& px) {
        Reading r;
        const double c = (kSize - 1) * 0.5;
        // Unit sphere at 3.2, vfov 45: projected radius in pixels.
        const double rad = std::tan(std::asin(1.0 / 3.2)) / std::tan(22.5 * math::PI / 180.0) * (kSize * 0.5);
        int n = 0, nl = 0, nr = 0;
        for (int y = 0; y < kSize; ++y) {
            for (int x = 0; x < kSize; ++x) {
                const double d = std::hypot(x - c, y - c);
                if (d > 0.9 * rad) continue;
                const unsigned char* p = &px[(static_cast<size_t>(y) * kSize + x) * 3];
                const double R = lin(p[0]), G = lin(p[1]), B = lin(p[2]);
                r.sphere += (R + G + B) / 3;
                ++n;
                if (x < c - 2) { r.redLeft += R - G; ++nl; }
                if (x > c + 2) { r.redRight += R - G; ++nr; }
            }
        }
        r.sphere /= n;
        r.redLeft /= nl;
        r.redRight /= nr;
        int nb = 0;
        for (int y = 4; y < 16; ++y) {
            for (int x = 4; x < 16; ++x) {
                r.background += lin(px[(static_cast<size_t>(y) * kSize + x) * 3 + 1]);
                ++nb;
            }
        }
        r.background /= nb;
        return r;
    }

}// namespace

int main() {
    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanEnvKnobs_test").size(kSize, kSize).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.setSharpenStrength(0.f);
    renderer.setEnvSunPolicy(Renderer::EnvSunPolicy::Off);// the red patch is not a sun
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;
    renderer.setClearColor(Color(0.f, 0.f, 0.f));

    const auto env = patchEnv();
    Scene scene;
    scene.background = env;
    scene.environment = env;
    auto mirror = MeshStandardMaterial::create();
    mirror->color = Color(1, 1, 1);
    mirror->metalness = 1.f;
    mirror->roughness = 0.f;
    scene.add(Mesh::create(SphereGeometry::create(1.f, 96, 64), mirror));

    PerspectiveCamera camera(45.f, 1.f, 0.1f, 100.f);
    camera.position.set(0, 0, 3.2f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    struct Case {
        const char* name;
        std::function<void(Scene&)> knobs;
        Reading reading;
    };
    std::vector<Case> cases = {
            {"default", [](Scene&) {}, {}},
            {"environmentIntensity=0", [](Scene& s) { s.environmentIntensity = 0.f; }, {}},
            {"backgroundIntensity=0.5", [](Scene& s) { s.backgroundIntensity = 0.5f; }, {}},
            {"environmentRotation.y=180", [](Scene& s) { s.environmentRotation.set(0, math::PI, 0); }, {}},
    };

    size_t idx = 0;
    int frame = 0;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("  %-64s %s\n", what, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    };

    canvas.animate([&] {
        renderer.render(scene, camera);
        if (++frame < kFrames) return;
        cases[idx].reading = measure(renderer.readRGBPixels());
        const auto& r = cases[idx].reading;
        std::printf("[%s] sphere=%.4f redLeft=%.4f redRight=%.4f background=%.4f\n", cases[idx].name, r.sphere,
                    r.redLeft, r.redRight, r.background);
        frame = 0;
        if (++idx < cases.size()) {
            // Knobs change on the SAME scene: the renderer must notice and wipe
            // its history itself (no resetTemporalHistory here, on purpose).
            scene.environmentIntensity = 1;
            scene.backgroundIntensity = 1;
            scene.environmentRotation.set(0, 0, 0);
            cases[idx].knobs(scene);
            return;
        }

        const auto& base = cases[0].reading;
        const auto& env0 = cases[1].reading;
        const auto& bgHalf = cases[2].reading;
        const auto& rot = cases[3].reading;
        char line[256];
        std::snprintf(line, sizeof line, "environmentIntensity 0: sphere %.4f < 2%% of %.4f", env0.sphere, base.sphere);
        check(env0.sphere < 0.02 * base.sphere && std::abs(env0.background - base.background) < 0.02 * base.background, line);
        std::snprintf(line, sizeof line, "backgroundIntensity 0.5: background ratio %.3f in [0.47, 0.53]",
                      bgHalf.background / base.background);
        check(std::abs(bgHalf.background / base.background - 0.5) < 0.03, line);
        std::snprintf(line, sizeof line, "environmentRotation 180: red right %.3f/%.3f -> left %.3f/%.3f", base.redRight,
                      base.redLeft, rot.redLeft, rot.redRight);
        check(base.redRight > 2 * std::max(base.redLeft, 1e-3) && rot.redLeft > 2 * std::max(rot.redRight, 1e-3) &&
                      std::abs(rot.redLeft - base.redRight) < 0.25 * base.redRight,
              line);
        std::printf("env knobs: %d failed\n", failures);
        std::exit(failures == 0 ? 0 : 1);
    });
    return 0;
}
