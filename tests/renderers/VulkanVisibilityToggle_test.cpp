// VulkanVisibilityToggle_test: hiding and showing a mesh at run time is no
// structural scene rebuild, and a hidden mesh is gone from every view of it.
//
// Toggling Object3D::visible (or material()->visible) used to re-expand and
// rebuild the whole scene, a few ms of render() on a game-sized scene each
// way, so the dune-buggy parked objects 400 m underground instead of hiding
// them (2026-10-15). A mesh that had entries now keeps them while hidden
// (MeshEntry::hidden): ray mask 0, culled from the raster, no emitter.
//
// The camera looks at a mirror with its back to three spheres, so they are
// seen only as ray hits (the TLAS masks), and a small quad hangs in front of
// the mirror, seen only by the raster:
//   red    sphere, left   toggled through Object3D::visible, then its material;
//   green  sphere, right  inside a group, hidden through the group;
//   yellow sphere, centre hidden from the start, shown late: its first show is
//                         the one structural rebuild the test allows;
//   blue   quad, top      toggled with the red sphere.
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "vulkan_test_util.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 384, kH = 192;
    constexpr int kSettleFrames = 45;

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

    // Channel c leads both others by 2x.
    bool leads(const std::array<double, 3>& p, int c) {
        const double o1 = p[(c + 1) % 3], o2 = p[(c + 2) % 3];
        return p[c] > 2.0 * o1 + 4.0 && p[c] > 2.0 * o2 + 4.0;
    }

    bool yellow(const std::array<double, 3>& p) {
        return p[0] > 2.0 * p[2] + 4.0 && p[1] > 2.0 * p[2] + 4.0;
    }

    std::shared_ptr<MeshStandardMaterial> paint(const Color& c) {
        auto m = MeshStandardMaterial::create();
        m->color = c;
        m->metalness = 0.f;
        m->roughness = 0.6f;
        return m;
    }

}// namespace

int main() {

    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto gpu = vktest::openHeadlessVulkan("VulkanVisibilityToggle_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;

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

    PerspectiveCamera camera(30.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0, 3.f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    const auto mirrorMat = MeshStandardMaterial::create();
    mirrorMat->color = Color(1, 1, 1);
    mirrorMat->metalness = 1.f;
    mirrorMat->roughness = 0.f;
    scene.add(Mesh::create(PlaneGeometry::create(12.f, 6.f), mirrorMat));

    auto red = Mesh::create(SphereGeometry::create(0.8f, 32, 16), paint(Color(0.9f, 0.05f, 0.05f)));
    red->position.set(-2.f, 0, 6.f);
    scene.add(red);
    auto group = Group::create();
    scene.add(group);
    auto green = Mesh::create(SphereGeometry::create(0.8f, 32, 16), paint(Color(0.05f, 0.9f, 0.05f)));
    green->position.set(2.f, 0, 6.f);
    group->add(green);
    auto yel = Mesh::create(SphereGeometry::create(0.8f, 32, 16), paint(Color(0.9f, 0.8f, 0.05f)));
    yel->position.set(0.f, 0, 6.f);
    yel->visible = false;
    scene.add(yel);
    auto blueMat = paint(Color(0.05f, 0.1f, 0.9f));
    auto quad = Mesh::create(PlaneGeometry::create(0.25f, 0.25f), blueMat);
    quad->position.set(0.f, 0.32f, 1.5f);
    scene.add(quad);

    auto sun = DirectionalLight::create(0xffffff, 2.5f);
    sun->position.set(1.f, 6.f, 4.f);
    scene.add(sun);

    // Screen position of a world point (a mirror image is the point mirrored
    // through z = 0).
    const auto px = [&](float x, float y, float z) {
        Vector3 p(x, y, z);
        p.project(camera);
        return std::array<int, 2>{static_cast<int>((p.x * 0.5f + 0.5f) * kW),
                                  static_cast<int>((1.f - (p.y * 0.5f + 0.5f)) * kH)};
    };
    const auto pRed = px(-2.f, 0, -6.f), pGreen = px(2.f, 0, -6.f), pYel = px(0.f, 0, -6.f);
    const auto pQuad = px(0.f, 0.32f, 1.5f);

    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    try {
        const auto settle = [&] {
            for (int i = 0; i < kSettleFrames; ++i) {
                canvas.animateOnce([&] { renderer.render(scene, camera); });
            }
            auto out = renderer.readRGBPixels();
            if (out.size() < static_cast<size_t>(kW) * kH * 3) {
                std::printf("readback too small: %zu\n", out.size());
                std::exit(1);
            }
            return out;
        };
        const auto at = [&](const std::vector<unsigned char>& im, const std::array<int, 2>& p, int r) {
            return patchMean(im, p[0], p[1], r);
        };
        const auto report = [&](const std::vector<unsigned char>& im, const char* label) {
            const auto r = at(im, pRed, 8), g = at(im, pGreen, 8), y = at(im, pYel, 8), q = at(im, pQuad, 2);
            std::printf("%s\n  red (%3.0f %3.0f %3.0f) green (%3.0f %3.0f %3.0f) yellow (%3.0f %3.0f %3.0f) "
                        "quad (%3.0f %3.0f %3.0f), structural rebuilds %llu\n",
                        label, r[0], r[1], r[2], g[0], g[1], g[2], y[0], y[1], y[2], q[0], q[1], q[2],
                        static_cast<unsigned long long>(renderer.tlasStats().structuralRebuilds));
        };
        const auto structural = [&] { return renderer.tlasStats().structuralRebuilds; };

        auto im = settle();
        report(im, "as built:");
        const auto s0 = structural();
        check("red and green reflected, yellow hidden from the start, quad blue",
              leads(at(im, pRed, 8), 0) && leads(at(im, pGreen, 8), 1) && !yellow(at(im, pYel, 8)) &&
                      leads(at(im, pQuad, 2), 2));

        red->visible = false;
        quad->visible = false;
        im = settle();
        report(im, "red sphere and quad hidden (Object3D::visible):");
        check("red gone from the mirror (TLAS)", !leads(at(im, pRed, 8), 0));
        check("quad gone from the raster", !leads(at(im, pQuad, 2), 2));
        check("no structural rebuild", structural() == s0);

        red->visible = true;
        quad->visible = true;
        im = settle();
        report(im, "shown again:");
        check("red and quad back", leads(at(im, pRed, 8), 0) && leads(at(im, pQuad, 2), 2));
        check("no structural rebuild", structural() == s0);

        red->material()->visible = false;
        im = settle();
        report(im, "red sphere hidden through its material:");
        check("red gone from the mirror", !leads(at(im, pRed, 8), 0));
        check("no structural rebuild", structural() == s0);

        red->material()->visible = true;
        group->visible = false;
        im = settle();
        report(im, "material shown, the green sphere's group hidden:");
        check("red back, green gone", leads(at(im, pRed, 8), 0) && !leads(at(im, pGreen, 8), 1));
        check("no structural rebuild", structural() == s0);

        group->visible = true;
        im = settle();
        report(im, "group shown again:");
        check("green back", leads(at(im, pGreen, 8), 1));
        check("no structural rebuild", structural() == s0);

        const auto y0 = at(im, pYel, 8);
        yel->visible = true;
        im = settle();
        report(im, "yellow shown for the first time:");
        // Only on its way to yellow: a first show is a structural rebuild, and
        // the mirror's history of the empty spot is not rejected there, so the
        // reflection converges over hundreds of frames (as it did before for
        // every toggle). Hiding it again below is in place and immediate.
        const auto y1 = at(im, pYel, 8);
        check("yellow appears (its blue drops from the empty mirror's)", y1[2] < y0[2] - 10.0);
        check("green still there", leads(at(im, pGreen, 8), 1));
        check("one structural rebuild (yellow's first show)", structural() == s0 + 1);

        yel->visible = false;
        im = settle();
        report(im, "yellow hidden again:");
        check("yellow gone", !yellow(at(im, pYel, 8)));
        check("no further structural rebuild", structural() == s0 + 1);
        if (vulkan::validationActive()) {
            check("no validation errors", vulkan::validationErrorCount() == 0u);
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL: the renderer threw: %s\n", e.what());
        ++failures;
    }

    std::printf("visibility toggle: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
