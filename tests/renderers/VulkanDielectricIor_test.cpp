// VulkanDielectricIor_test — how much a dielectric reflects follows its index of
// refraction.
//
// Normal-incidence reflectance is ((n-1)/(n+1))^2: nothing at n = 1, 0.04 at 1.5,
// 0.172 for diamond at 2.42. The deferred shade used the constant 0.04 wherever it
// needed a dielectric F0 (primary shade, shared hit shade, reflection hits) and
// never looked at the material's ior, so glTF sample IORTestGrid's column of black
// spheres, which differ in nothing else, all looked alike, and its ior 1.0 sphere
// had a highlight where the Khronos reference has none.
//
// Three black, smooth, opaque quads facing the camera in a uniform environment. A
// black base has no diffuse term, so the centre of each reads the environment
// times its reflectance at normal incidence, and the three expected values are an
// order of magnitude apart.
//
// Plain exit-code program, not Catch2 — same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
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

    double patch(const std::vector<unsigned char>& px, int cx, int cy, int r) {
        double sum = 0;
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                sum += srgbToLinear(px[(static_cast<size_t>(y) * kW + x) * 3 + 1] / 255.0);
                ++n;
            }
        }
        return sum / n;
    }

}// namespace

int main() {

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanDielectricIor_test").size(kW, kH).vsync(false).headless(true));
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

    constexpr std::array<float, 3> iors{1.0f, 1.5f, 2.42f};
    for (int i = 0; i < 3; ++i) {
        auto m = MeshPhysicalMaterial::create();
        m->color = Color(0, 0, 0);
        m->metalness = 0.f;
        m->roughness = 0.f;
        m->setIor(iors[i]);
        auto quad = Mesh::create(PlaneGeometry::create(1.f, 1.f), m);
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

        const double envPx = patch(px, kW / 2, 6, 3);// above the quads
        std::array<double, 3> got{};
        for (int i = 0; i < 3; ++i) got[i] = patch(px, kW / 6 + i * (kW / 3), kH / 2, 4) / envPx;

        int failures = 0;
        const auto check = [&](const char* what, bool ok) {
            std::printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
            if (!ok) ++failures;
        };

        std::printf("black smooth dielectric, of the environment:\n");
        for (int i = 0; i < 3; ++i) {
            const double r = (iors[i] - 1.0) / (iors[i] + 1.0);
            std::printf("  ior %.2f  %.4f   ((n-1)/(n+1))^2 = %.4f\n", iors[i], got[i], r * r);
        }

        check("ior 1.0 reflects nothing (< 0.01)", got[0] < 0.01);
        check("ior 1.5 reflects 0.04 (0.03..0.05)", got[1] > 0.03 && got[1] < 0.05);
        check("ior 2.42 reflects 0.172 (0.14..0.20)", got[2] > 0.14 && got[2] < 0.20);

        std::printf("dielectric ior: %d failure(s)\n", failures);
        std::exit(failures == 0 ? 0 : 1);
    });
    return 0;
}
