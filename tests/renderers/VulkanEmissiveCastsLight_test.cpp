// VulkanEmissiveCastsLight_test: setEmissiveCastsLight must not change how a
// ray hit reads the mesh's triangles.
//
// GeometryDesc.flags tells the ray-hit shaders (reflections, GI, probes, lidar)
// how to decode a mesh's buffers. One of its bits says the index buffer is
// uint16, two indices to a word, which is what every static mesh of at most
// 65536 vertices gets. setEmissiveCastsLight was given the same bit for "glows
// but is not a light", so:
//
//   castsLight = false on a mesh with uint32 indices SET the uint16 bit, and
//       every ray that hit the mesh fetched its triangles two to a word: a
//       reflection of crumpled garbage, reads in bounds;
//   castsLight = true on a mesh with uint16 indices CLEARED it, and every ray
//       that hit the mesh read index words that are two indices fused (up to
//       65535 * 65536 + 65535) and then fetched vertices at those: far outside
//       the buffer, a GPU fault, VK_ERROR_DEVICE_LOST from the next submit. The
//       validation layer sees none of it, the reads go through buffer device
//       addresses.
//
// The second was found on the Norvasundet twin (2026-10-09): a bench that
// switched every emitter to glow-only and back lost the device three times out
// of three.
//
// The scene is a mirror with two spheres behind the camera, so the whole image
// is ray hits: the left half the reflection of a sphere small enough for uint16
// indices, the right half one of a sphere too big for them. Neither sphere is
// emissive, so the flag has nothing legitimate to change and each half must
// stay what it was through false and back to true. The limit is the test's own
// null (the same scene read back twice) with a wide margin.
//
// Plain exit-code program, not Catch2, the same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 384, kH = 192;
    constexpr int kSettleFrames = 60;
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

    // Mean absolute difference, in 8-bit levels, over a square around (cx, cy).
    double patchDiff(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b,
                     int cx, int cy, int r) {
        double sum = 0.0;
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const size_t i = (static_cast<size_t>(y) * kW + x) * 3 + c;
                    sum += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
                    ++n;
                }
            }
        }
        return sum / n;
    }

    // Mean level of a square around (cx, cy), per channel.
    std::array<double, 3> patchMean(const std::vector<unsigned char>& a, int cx, int cy, int r) {
        std::array<double, 3> sum{0, 0, 0};
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                for (int c = 0; c < 3; ++c) sum[c] += a[(static_cast<size_t>(y) * kW + x) * 3 + c];
                ++n;
            }
        }
        for (double& v : sum) v /= n;
        return sum;
    }

}// namespace

int main() {

    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanEmissiveCastsLight_test").size(kW, kH).vsync(false).headless(true));
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

    const auto env = constantEnv(0.25f);

    Scene scene;
    scene.background = env;
    scene.environment = env;

    // The camera looks at the mirror with its back to the spheres: 30 degrees of
    // 2:1 frame at 9 m of mirrored path, each sphere 40 px in radius and 80 px
    // to its side of the centre.
    PerspectiveCamera camera(30.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0, 3.f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    const auto mirrorMat = MeshStandardMaterial::create();
    mirrorMat->color = Color(1, 1, 1);
    mirrorMat->metalness = 1.f;
    mirrorMat->roughness = 0.f;
    scene.add(Mesh::create(PlaneGeometry::create(12.f, 6.f), mirrorMat));

    const auto paint = MeshStandardMaterial::create();
    paint->color = Color(0.8f, 0.45f, 0.2f);
    paint->metalness = 0.f;
    paint->roughness = 0.6f;

    // 49 x 25 = 1225 vertices: uint16 indices. 301 x 301 = 90601: uint32.
    // Their own geometry at every distance: a simplified level would be a
    // second thing that changes under the readbacks.
    auto small = Mesh::create(SphereGeometry::create(1.f, 48, 24), paint);
    small->position.set(-2.f, 0, 6.f);// the image's left: its mirror image keeps its x
    small->autoLod = false;
    scene.add(small);
    auto big = Mesh::create(SphereGeometry::create(1.f, 300, 300), paint);
    big->position.set(2.f, 0, 6.f);
    big->autoLod = false;
    scene.add(big);

    auto sun = DirectionalLight::create(0xffffff, 2.5f);
    sun->position.set(1.f, 6.f, 1.f);// from above and a little before them: a gradient down each
    scene.add(sun);

    const int cxSmall = kW / 2 - 80, cxBig = kW / 2 + 80, cy = kH / 2, r = 22;

    int failures = 0;
    const auto check = [&](const char* what, double got, double limit) {
        const bool ok = got <= limit;
        std::printf("  %-64s %7.3f (limit %.3f) %s\n", what, got, limit, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    try {
        const auto settle = [&](int n) {
            for (int i = 0; i < n; ++i) {
                canvas.animateOnce([&] { renderer.render(scene, camera); });
            }
            return renderer.readRGBPixels();
        };

        const auto ref = settle(kSettleFrames);
        if (ref.size() < static_cast<size_t>(kW) * kH * 3) {
            std::printf("readback too small: %zu\n", ref.size());
            return 1;
        }
        const auto again = settle(kSettleFrames);
        const double nullSmall = patchDiff(again, ref, cxSmall, cy, r);
        const double nullBig = patchDiff(again, ref, cxBig, cy, r);
        // The reflections are there at all: the mirror's corner shows the
        // environment, and neither patch is that colour.
        const auto corner = patchMean(ref, r + 2, r + 2, r);
        const auto apart = [&](int cx) {
            const auto p = patchMean(ref, cx, cy, r);
            return (std::abs(p[0] - corner[0]) + std::abs(p[1] - corner[1]) + std::abs(p[2] - corner[2])) / 3.0;
        };
        const double there = std::min(apart(cxSmall), apart(cxBig));
        std::printf("null (the scene read back twice): small %.3f, big %.3f; reflections against the corner: %.1f\n",
                    nullSmall, nullBig, there);
        if (there < 8.0) {
            std::printf("the mirror shows no spheres: the test has nothing to compare\n");
            return 1;
        }
        const double limSmall = std::max(4.0 * nullSmall, 1.5), limBig = std::max(4.0 * nullBig, 1.5);

        renderer.setEmissiveCastsLight(*small, false);
        renderer.setEmissiveCastsLight(*big, false);
        const auto off = settle(kSettleFrames);
        std::printf("castsLight = false on both:\n");
        check("uint16 sphere's reflection is what it was", patchDiff(off, ref, cxSmall, cy, r), limSmall);
        check("uint32 sphere's reflection is what it was", patchDiff(off, ref, cxBig, cy, r), limBig);

        renderer.setEmissiveCastsLight(*small, true);
        renderer.setEmissiveCastsLight(*big, true);
        const auto on = settle(kSettleFrames);
        std::printf("castsLight = true again:\n");
        check("uint16 sphere's reflection is what it was", patchDiff(on, ref, cxSmall, cy, r), limSmall);
        check("uint32 sphere's reflection is what it was", patchDiff(on, ref, cxBig, cy, r), limBig);
    } catch (const std::exception& e) {
        std::printf("  FAIL: the renderer threw: %s\n", e.what());
        ++failures;
    }

    std::printf("emissive casts light: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
