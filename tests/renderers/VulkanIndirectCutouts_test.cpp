// VulkanIndirectCutouts_test — the indirect rays and alpha-tested cutouts
// (VulkanRenderer::setIndirectOpaqueCutouts).
//
// The GI bounce, the probe sweep and the ambient-occlusion rays traced
// force-opaque, so under a card tree's crown every one of them ended on the
// first card it met, transparent texel or not: a closed roof, where the surface
// shadows, the reflections and the path tracer alpha-test every card. With the
// knob off those rays pass a pure cutout's transparent texels inside their
// traversal, up to kIndirectCutoutPassMax of them.
//
// A roof of one alpha-cutout quad, a checkerboard of solid and transparent
// cells (half and half), over grey ground under a uniform white sky, no lights:
// what the ground under the roof receives is indirect light alone. Read off the
// picture, the ground under the roof against the open ground far from it:
//   default  the roof is closed to the indirect rays, the ground under it dark
//            (measured 0.15 of the open ground on an RTX 5080)
//   knob off the ground under it reads about the roof's open share of the open
//            ground, and well over the default (0.53: the holes plus the bounce
//            off the solid cells' undersides)
// The open ground itself must not move between the two.
//
// Plain exit-code program, not Catch2 — same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available, or when
// the device has no ray query (the indirect rays are not traced).

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "vulkan_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 640, kH = 360;
    constexpr int kFrames = 150;// the 1 spp GI accumulates over the renderer's temporal window

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

    // A green checkerboard whose alpha is 1 on the even cells and 0 on the odd
    // ones: half the quad is a hole, in 8 x 8 cells.
    std::shared_ptr<Texture> checkerCutout() {
        constexpr int W = 64, H = 64, cell = 8;
        std::vector<unsigned char> data(static_cast<size_t>(W) * H * 4, 0);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const bool solid = ((x / cell) + (y / cell)) % 2 == 0;
                unsigned char* p = &data[(static_cast<size_t>(y) * W + x) * 4];
                p[0] = 40;
                p[1] = 140;
                p[2] = 50;
                p[3] = solid ? 255 : 0;
            }
        }
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->colorSpace = ColorSpace::Linear;
        tex->needsUpdate();
        return tex;
    }

    double srgbToLinear(double c) {
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    // The mean linear luma of a (2r+1)^2 patch about a projected world point.
    double patchAt(const std::vector<unsigned char>& px, const Camera& cam, Vector3 p, int r) {
        p.project(cam);
        const int cx = static_cast<int>((p.x + 1.f) * 0.5f * static_cast<float>(kW));
        const int cy = static_cast<int>((1.f - p.y) * 0.5f * static_cast<float>(kH));
        double sum = 0;
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                if (x < 0 || y < 0 || x >= kW || y >= kH) continue;
                const size_t at = (static_cast<size_t>(y) * kW + x) * 3u;
                sum += 0.2126 * srgbToLinear(px[at] / 255.0) + 0.7152 * srgbToLinear(px[at + 1] / 255.0)
                     + 0.0722 * srgbToLinear(px[at + 2] / 255.0);
                ++n;
            }
        }
        return n ? sum / n : 0.0;
    }

}// namespace

int main() {

    auto gpu = vktest::openHeadlessVulkan("VulkanIndirectCutouts_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;
    if (!renderer.rayTracingSupported()) {
        std::printf("[skip] no ray query: the indirect rays are not traced\n");
        return kSkipCode;
    }

    renderer.setDenoise(true);  // the stochastic 1-bounce GI, accumulated
    renderer.setDeferredAO(true);
    renderer.setProbeGI(false); // the gather's own rays decide, not the grid
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

    auto groundMat = MeshStandardMaterial::create();
    groundMat->color = Color(0.5f, 0.5f, 0.5f);
    groundMat->roughness = 1.f;
    auto ground = Mesh::create(PlaneGeometry::create(140.f, 140.f), groundMat);
    ground->rotation.x = -3.14159265f / 2.f;
    scene.add(ground);

    auto roofMat = MeshStandardMaterial::create();
    roofMat->map = checkerCutout();
    roofMat->alphaTest = 0.5f;
    roofMat->roughness = 1.f;
    roofMat->side = Side::Double;
    auto roof = Mesh::create(PlaneGeometry::create(14.f, 14.f), roofMat);
    roof->rotation.x = -3.14159265f / 2.f;
    roof->position.y = 3.f;
    scene.add(roof);

    PerspectiveCamera camera(45.f, static_cast<float>(kW) / kH, 0.1f, 200.f);
    camera.position.set(0.f, 1.3f, 10.f);
    camera.lookAt(Vector3{0.f, 0.f, -4.f});
    camera.updateMatrixWorld();

    const Vector3 under{0.f, 0.f, 0.f};  // the ground in the middle of the roof
    const Vector3 open{0.f, 0.f, -45.f}; // the ground far past its far edge: the roof is a sliver of its sky

    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    check("indirect cutouts default to opaque", renderer.indirectOpaqueCutouts());

    struct Arm {
        bool opaque;
        double under = 0, open = 0;
    };
    Arm arms[2] = {{true}, {false}};
    int arm = 0, frame = 0;
    renderer.setIndirectOpaqueCutouts(arms[0].opaque);

    canvas.animate([&] {
        renderer.render(scene, camera);
        if (++frame < kFrames) return;
        const auto px = renderer.readRGBPixels();
        if (px.size() < static_cast<size_t>(kW) * kH * 3) {
            std::printf("readback too small: %zu\n", px.size());
            std::exit(1);
        }
        arms[arm].under = patchAt(px, camera, under, 6);
        arms[arm].open = patchAt(px, camera, open, 3);
        std::printf("indirect rays %s cutouts: ground under the roof %.4f, open ground %.4f, ratio %.3f\n",
                    arms[arm].opaque ? "take" : "alpha-test", arms[arm].under, arms[arm].open,
                    arms[arm].under / std::max(arms[arm].open, 1e-6));
        if (arm == 0) {
            ++arm;
            frame = 0;
            renderer.setIndirectOpaqueCutouts(arms[1].opaque);
            renderer.resetTemporalHistory();
            return;
        }
        check("the knob reads back off", !renderer.indirectOpaqueCutouts());
        const double r0 = arms[0].under / std::max(arms[0].open, 1e-6);
        const double r1 = arms[1].under / std::max(arms[1].open, 1e-6);
        check("the open ground is the same in both (within 8 %)",
              std::fabs(arms[1].open - arms[0].open) <= 0.08 * std::max(arms[0].open, 1e-6));
        check("default: the roof is closed to the indirect rays (under/open < 0.45)", r0 < 0.45);
        check("knob off: the open share of the sky comes through (under/open 0.30..0.80)", r1 > 0.30 && r1 < 0.80);
        check("knob off: at least 1.4x the default under the roof", r1 >= 1.4 * r0);
        std::printf("indirect cutouts: %d failure(s)\n", failures);
        std::exit(failures == 0 ? 0 : 1);
    });
    return 0;
}
