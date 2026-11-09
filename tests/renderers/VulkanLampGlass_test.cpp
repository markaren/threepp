// VulkanLampGlass_test: a lamp is dimmed by the glass it shines through,
// with the denoiser on and with it off.
//
// The shadow rays of the denoised path (shadowRayVisTo) multiply in what the
// glass on the way lets through (glassTransmittance), but the inline lamp
// loops the shade runs with the denoiser off took the opaque test alone: a
// floor under a smoked pane took about a tenth of a point lamp with the
// denoiser on and all of it with setDenoise(false).
//
// A white floor under a point lamp, a smoked pane (thin glass, base colour
// 0.1) between the lamp and the floor's left half. The camera sits below the
// pane, so no view ray crosses it. Two floor patches mirror each other about
// the lamp, one under the pane and one in the open: the ratio of the two is
// what the pane lets through, about 0.1 x (1 - F0)^2 = 0.09. The thresholds sit
// far from it and from 1, the value without the glass.
//
//   VulkanLampGlass_test --dump <dir>   also writes the two renders there as
//                                       lampglass_denoise_{off,on}.ppm
//
// Plain exit-code program, not Catch2, the same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 256, kH = 128;
    constexpr int kFrames = 60;

    double srgbToLinear(double c) {
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    // Mean linear luminance over a (2r+1)^2 patch around the pixel of world point P.
    double patchAt(const std::vector<unsigned char>& px, const PerspectiveCamera& cam, Vector3 P, int r) {
        P.project(cam);
        const int cx = static_cast<int>((P.x * 0.5 + 0.5) * kW);
        const int cy = static_cast<int>((0.5 - P.y * 0.5) * kH);
        double sum = 0;
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                if (x < 0 || y < 0 || x >= kW || y >= kH) continue;
                const size_t i = (static_cast<size_t>(y) * kW + x) * 3;
                sum += 0.2126 * srgbToLinear(px[i] / 255.0) + 0.7152 * srgbToLinear(px[i + 1] / 255.0) +
                       0.0722 * srgbToLinear(px[i + 2] / 255.0);
                ++n;
            }
        }
        return n > 0 ? sum / n : 0.0;
    }

}// namespace

int main(int argc, char** argv) {

    std::string dump;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--dump") == 0) dump = argv[i + 1];

    auto gpu = vktest::openHeadlessVulkan("VulkanLampGlass_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.setClearColor(Color(0.f, 0.f, 0.f));
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;

    PerspectiveCamera camera(40.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0.9f, 3.f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    auto floorMat = MeshStandardMaterial::create();
    floorMat->color = Color(0.8f, 0.8f, 0.8f);
    floorMat->roughness = 1.f;
    floorMat->metalness = 0.f;
    auto floor = Mesh::create(PlaneGeometry::create(10.f, 10.f), floorMat);
    floor->rotation.x = -math::PI / 2;
    scene.add(floor);

    // Smoked glass over x in [-2.5, 0], 1.5 up: thin (thickness 0), base colour 0.1.
    auto paneMat = MeshPhysicalMaterial::create();
    paneMat->color = Color(0.1f, 0.1f, 0.1f);
    paneMat->metalness = 0.f;
    paneMat->roughness = 0.f;
    paneMat->transmission = 1.f;
    paneMat->setIor(1.5f);
    auto pane = Mesh::create(PlaneGeometry::create(2.5f, 3.f), paneMat);
    pane->rotation.x = -math::PI / 2;
    pane->position.set(-1.25f, 1.5f, 0.f);
    scene.add(pane);

    auto lamp = PointLight::create(Color(1, 1, 1), 8.f, 0.f, 2.f);
    lamp->position.set(0, 2.f, 0);
    scene.add(lamp);

    // The two patches: their rays to the lamp cross the pane's height at x = -0.15 and 0.15.
    const Vector3 under{-0.6f, 0.f, 0.f}, open{0.6f, 0.f, 0.f};

    std::array<double, 2> ratio{0, 0};
    std::array<double, 2> openLum{0, 0};
    int pass = 0;// 0: denoiser off, 1: on
    int frame = 0;
    renderer.setDenoise(false);
    canvas.animate([&] {
        renderer.render(scene, camera);
        if (++frame < kFrames) return;

        const auto px = renderer.readRGBPixels();
        if (px.size() < static_cast<size_t>(kW) * kH * 3) {
            std::printf("readback too small: %zu\n", px.size());
            std::exit(1);
        }
        if (!dump.empty()) {
            std::ofstream f(dump + (pass == 0 ? "/lampglass_denoise_off.ppm" : "/lampglass_denoise_on.ppm"),
                            std::ios::binary);
            f << "P6\n" << kW << " " << kH << "\n255\n";
            f.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(kW) * kH * 3);
        }
        openLum[pass] = patchAt(px, camera, open, 3);
        ratio[pass] = patchAt(px, camera, under, 3) / std::max(openLum[pass], 1e-6);
        std::printf("denoiser %-3s  open %.4f  under the pane / open %.3f\n", pass == 0 ? "off" : "on",
                    openLum[pass], ratio[pass]);
        frame = 0;
        if (++pass < 2) {
            renderer.setDenoise(true);
            return;
        }

        int failures = 0;
        const auto check = [&](const char* what, bool ok) {
            std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
            if (!ok) ++failures;
        };
        check("the open floor is lit (> 0.1)", openLum[0] > 0.1 && openLum[1] > 0.1);
        check("denoiser off: the smoked pane passes about a tenth (0.03..0.25)", ratio[0] > 0.03 && ratio[0] < 0.25);
        check("denoiser on: the smoked pane passes about a tenth (0.03..0.25)", ratio[1] > 0.03 && ratio[1] < 0.25);
        check("the two agree within 0.06", std::abs(ratio[0] - ratio[1]) < 0.06);

        std::printf("lamp through glass: %d failure(s)\n", failures);
        std::exit(failures == 0 ? 0 : 1);
    });
    return 0;
}
