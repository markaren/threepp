// VulkanIorOneGlass_test — glass with an index of refraction of exactly 1 is
// still glass.
//
// The deferred renderer carries two very different things as "transmission > 0":
// real glass, and a flat alpha blend (opacity < 1, uploaded as transmission =
// 1 - opacity). It told them apart by the ior, and the blend's marker was ior = 1.
// That is a value a material can really have. glTF sample SunglassesKhronos gives
// the FRONT of its lenses KHR_materials_ior 1.0, transmission 1, a base colour of
// 0.009 and a thin film (KHR_materials_iridescence, film ior 2, 300 nm). It was
// read as "alpha blend at opacity 0" and not drawn: the lenses were clear from the
// front and black only from behind, where the other face has the default ior.
//
// Three quads side by side in a uniform environment, one render:
//
//   tinted     transmission 1, ior 1.0, near-black base colour. ior 1 has no
//              Fresnel reflection of its own and the base colour tints what comes
//              through, so this is DARK. Read as a blend it was the environment.
//   film       the same under the sunglasses' thin film. The film is then the only
//              reflector: darker than the environment, brighter than `tinted`, and
//              COLOURED, because a 300 nm film reflects blue far more than red.
//              The glass path took its Fresnel from the ior alone, so without the
//              film term this reads the same as `tinted`.
//   blend      control: a black alpha blend at opacity 0.25 must still let 3/4 of
//              the environment through. Pins the blend's own marker.
//
// Everything is a ratio against the environment patch, and the thresholds sit far
// from the expected values, so nothing depends on convergence or exact shading.
//
// Plain exit-code program, not Catch2 — same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 384, kH = 128;
    constexpr int kFrames = 90;
    constexpr int kSkipCode = 42;

    std::shared_ptr<Texture> constantEnv(float le) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, le);
        for (size_t i = 3; i < data.size(); i += 4) data[i] = 1.f;
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    double srgbToLinear(double c) {
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    std::array<double, 3> patch(const std::vector<unsigned char>& px, int cx, int cy, int r) {
        std::array<double, 3> sum{0, 0, 0};
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                for (int c = 0; c < 3; ++c) sum[c] += srgbToLinear(px[(static_cast<size_t>(y) * kW + x) * 3 + c] / 255.0);
                ++n;
            }
        }
        for (double& v : sum) v /= n;
        return sum;
    }

    std::shared_ptr<MeshPhysicalMaterial> iorOneGlass() {
        auto m = MeshPhysicalMaterial::create();
        m->color = Color(0.009f, 0.009f, 0.009f);
        m->metalness = 0.f;
        m->roughness = 0.f;
        m->transmission = 1.f;
        m->setIor(1.0f);
        return m;
    }

}// namespace

int main() {

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanIorOneGlass_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    renderer.setDenoise(true);
    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    const auto env = constantEnv(0.5f);

    Scene scene;
    scene.background = env;
    scene.environment = env;

    // 3:1 frame, three unit quads 1.5 apart: each fills the middle of its third.
    PerspectiveCamera camera(30.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0, 3.2f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    const auto tinted = iorOneGlass();

    const auto film = iorOneGlass();
    film->iridescence = 1.f;
    film->iridescenceIOR = 2.f;
    film->iridescenceThicknessNm = 300.f;

    const auto blend = MeshStandardMaterial::create();// lit: the shade's own blend branch
    blend->color = Color(0, 0, 0);
    blend->roughness = 1.f;
    blend->transparent = true;
    blend->opacity = 0.25f;

    const std::array<std::shared_ptr<Material>, 3> mats{tinted, film, blend};
    for (int i = 0; i < 3; ++i) {
        auto quad = Mesh::create(PlaneGeometry::create(1.f, 1.f), mats[i]);
        quad->position.x = (i - 1) * 1.5f;
        scene.add(quad);
    }

    int frame = 0;
    canvas.animate([&] {
        renderer.render(scene, camera);
        if (++frame < kFrames) return;

        const auto px = renderer.readRGBPixels();
        if (px.size() < static_cast<size_t>(kW) * kH * 3) {
            std::printf("readback too small: %zu\n", px.size());
            std::exit(1);
        }

        const auto envPx = patch(px, kW / 2, 6, 3);// above the quads
        const auto rel = [&](int third) {
            auto p = patch(px, kW / 6 + third * (kW / 3), kH / 2, 4);
            for (int c = 0; c < 3; ++c) p[c] /= envPx[c];
            return p;
        };
        const auto t = rel(0), f = rel(1), b = rel(2);

        int failures = 0;
        const auto check = [&](const char* what, bool ok) {
            std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
            if (!ok) ++failures;
        };

        std::printf("of the environment (r g b):\n");
        std::printf("  tinted  %.3f %.3f %.3f\n  film    %.3f %.3f %.3f\n  blend   %.3f %.3f %.3f\n",
                    t[0], t[1], t[2], f[0], f[1], f[2], b[0], b[1], b[2]);

        check("ior 1.0 glass with a black tint is dark (< 0.1)", t[0] < 0.1 && t[1] < 0.1 && t[2] < 0.1);
        check("the film reflects: brighter than the bare glass (blue > 0.1)", f[2] > 0.1);
        check("the film does not turn it into the environment (< 0.8)", f[0] < 0.8 && f[1] < 0.8 && f[2] < 0.8);
        check("the film is coloured: blue at least 1.5x red", f[2] > 1.5 * f[0]);
        check("an alpha blend at opacity 0.25 still passes 3/4 (0.6..0.9)", b[1] > 0.6 && b[1] < 0.9);

        std::printf("ior-one glass: %d failure(s)\n", failures);
        std::exit(failures == 0 ? 0 : 1);
    });
    return 0;
}
