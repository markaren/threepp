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
//                         a slot edit (stable entry slots, VulkanCoreSlots.cpp),
//                         no structural rebuild; with THREEPP_VULKAN_NO_SLOTS=1
//                         it is the one structural rebuild the default allows;
//   blue   quad, top      toggled with the red sphere.
//
// Then five more are added hidden before they were ever rendered, each with
// Object3D::resident set; their first show is in place, no structural rebuild:
//   magenta sphere, lower left   emissive (a lit brake lens), Object3D::visible;
//   cyan    sphere, lower right  material()->visible;
//   orange  sphere, lower centre inside a group hidden from the start;
//   red     quad, top right      Object3D::visible, seen only by the raster;
//   grey    sphere, far behind   beyond the probe grid: its first show re-fits it.
// Beside them, two more spheres, hidden from the start, without the hint:
//   plain   sphere, upper left   never gets it: its first show is a slot edit,
//                                as yellow's was (without slots, the one
//                                structural rebuild);
//   late    sphere, upper right  gets it after it was rendered hidden: setting
//                                it builds the entries (a slot edit; without
//                                slots, the structural one), its first show is
//                                in place.
// Lidar beams check the spheres' TLAS masks directly.
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/helpers/LidarTypes.hpp"
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
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

    // Channels c and the next one both lead the third by 2x (c = 1: cyan,
    // c = 2: magenta).
    bool pairLeads(const std::array<double, 3>& p, int c) {
        const int d = (c + 1) % 3, o = (c + 2) % 3;
        return p[c] > 2.0 * p[o] + 4.0 && p[d] > 2.0 * p[o] + 4.0;
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
        const auto slotEdits = [&] { return renderer.tlasStats().slotEdits; };
        // A first show of a mesh with no entries: a slot edit, or with the
        // slots switched off the structural rebuild it always was.
        const char* noSlots = std::getenv("THREEPP_VULKAN_NO_SLOTS");
        const uint64_t firstShow = (noSlots && *noSlots == '1') ? 1u : 0u;

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
        const auto e0 = slotEdits();
        yel->visible = true;
        im = settle();
        report(im, "yellow shown for the first time:");
        // Only on its way to yellow: a first show adds entries, and the
        // mirror's history of the empty spot is not rejected there, so the
        // reflection converges over hundreds of frames (as it did before for
        // every toggle). Hiding it again below is in place.
        const auto y1 = at(im, pYel, 8);
        check("yellow appears (its blue drops from the empty mirror's)", y1[2] < y0[2] - 10.0);
        check("green still there", leads(at(im, pGreen, 8), 1));
        check(firstShow ? "one structural rebuild (yellow's first show)"
                        : "no structural rebuild, one slot edit (yellow's first show)",
              structural() == s0 + firstShow && (firstShow || slotEdits() == e0 + 1));

        yel->visible = false;
        im = settle();
        report(im, "yellow hidden again:");
        check("yellow gone", !yellow(at(im, pYel, 8)));
        check("no further structural rebuild", structural() == s0 + firstShow);

        // Object3D::resident: a mesh hidden before it was ever rendered has
        // its entries, hidden, from the start. Adding these is structural;
        // showing them for the first time is not.
        auto lensMat = paint(Color(0.9f, 0.05f, 0.9f));
        lensMat->emissive = Color(0.9f, 0.05f, 0.9f);
        lensMat->emissiveIntensity = 1.5f;
        auto lens = Mesh::create(SphereGeometry::create(0.5f, 32, 16), lensMat);
        lens->position.set(-2.f, -1.6f, 6.f);
        lens->visible = false;
        lens->resident = true;
        scene.add(lens);
        auto cyan = Mesh::create(SphereGeometry::create(0.5f, 32, 16), paint(Color(0.05f, 0.8f, 0.9f)));
        cyan->position.set(2.f, -1.6f, 6.f);
        cyan->material()->visible = false;
        cyan->resident = true;
        scene.add(cyan);
        auto hiddenGroup = Group::create();
        hiddenGroup->visible = false;
        scene.add(hiddenGroup);
        auto orange = Mesh::create(SphereGeometry::create(0.5f, 32, 16), paint(Color(0.9f, 0.6f, 0.05f)));
        orange->position.set(0.f, -1.6f, 6.f);
        orange->resident = true;// on the mesh itself: a group's flag is not inherited
        hiddenGroup->add(orange);
        auto quad2 = Mesh::create(PlaneGeometry::create(0.25f, 0.25f), paint(Color(0.9f, 0.05f, 0.05f)));
        quad2->position.set(0.6f, 0.32f, 1.5f);
        quad2->visible = false;
        quad2->resident = true;
        scene.add(quad2);
        // Beyond the probe grid fitted so far (z up to 6.8): shown in place, it
        // must still get the grid re-fitted around it, as a structural show did.
        auto farBall = Mesh::create(SphereGeometry::create(0.5f, 16, 8), paint(Color(0.5f, 0.5f, 0.5f)));
        farBall->position.set(0.f, 0.f, 12.f);
        farBall->visible = false;
        farBall->resident = true;
        scene.add(farBall);
        // Without the hint, beside them: plain never gets it, late gets it once
        // it has been rendered hidden.
        auto plain = Mesh::create(SphereGeometry::create(0.5f, 32, 16), paint(Color(0.05f, 0.1f, 0.9f)));
        plain->position.set(-2.f, 1.6f, 6.f);
        plain->visible = false;
        scene.add(plain);
        auto late = Mesh::create(SphereGeometry::create(0.5f, 32, 16), paint(Color(0.9f, 0.9f, 0.9f)));
        late->position.set(2.f, 1.6f, 6.f);
        late->visible = false;
        scene.add(late);
        const auto pLens = px(-2.f, -1.6f, -6.f), pCyan = px(2.f, -1.6f, -6.f), pOrange = px(0.f, -1.6f, -6.f);
        const auto pQuad2 = px(0.6f, 0.32f, 1.5f);
        const auto reportLower = [&](const std::vector<unsigned char>& im, const char* label) {
            const auto l = at(im, pLens, 6), c = at(im, pCyan, 6), o = at(im, pOrange, 6), q = at(im, pQuad2, 2);
            std::printf("%s\n  lens (%3.0f %3.0f %3.0f) cyan (%3.0f %3.0f %3.0f) orange (%3.0f %3.0f %3.0f) "
                        "quad2 (%3.0f %3.0f %3.0f), structural rebuilds %llu\n",
                        label, l[0], l[1], l[2], c[0], c[1], c[2], o[0], o[1], o[2], q[0], q[1], q[2],
                        static_cast<unsigned long long>(renderer.tlasStats().structuralRebuilds));
        };
        const auto lensLit = [&](const std::vector<unsigned char>& im) { return pairLeads(at(im, pLens, 6), 2); };
        const auto cyanSeen = [&](const std::vector<unsigned char>& im) { return pairLeads(at(im, pCyan, 6), 1); };
        const auto orangeSeen = [&](const std::vector<unsigned char>& im) { return yellow(at(im, pOrange, 6)); };
        const auto quad2Seen = [&](const std::vector<unsigned char>& im) { return leads(at(im, pQuad2, 2), 0); };
        // Lidar beams from the camera to the lens, cyan, orange, plain and
        // late: they read the TLAS masks directly. A hidden mesh's reflection
        // fades out of the mirror through the reflection history (only a ray
        // that hits a mesh just shown or moved resets it), which a mirror
        // patch cannot tell from a mesh still in the TLAS.
        const Vector3 eye(0.f, 0.f, 3.f);
        const std::array<Vector3, 5> centres{Vector3(-2.f, -1.6f, 6.f), Vector3(2.f, -1.6f, 6.f),
                                             Vector3(0.f, -1.6f, 6.f), Vector3(-2.f, 1.6f, 6.f),
                                             Vector3(2.f, 1.6f, 6.f)};
        enum { kLens, kCyan, kOrange, kPlain, kLate };
        std::vector<LidarBeam> beams;
        for (const auto& c : centres) beams.push_back({eye, (c - eye).normalize()});
        const auto lidarHits = [&] {
            std::vector<LidarReturn> rets;
            renderer.scanLidar(beams, rets);
            std::array<bool, 5> hit{};
            for (size_t i = 0; i < hit.size() && i < rets.size(); ++i) {
                const float d = (centres[i] - eye).length() - 0.5f;
                hit[i] = rets[i].returnNo > 0 && std::abs(rets[i].distance - d) < 0.05f;
            }
            std::printf("  lidar: lens %d cyan %d orange %d plain %d late %d\n", hit[kLens], hit[kCyan],
                        hit[kOrange], hit[kPlain], hit[kLate]);
            return hit;
        };
        // Exactly the spheres in `want` return a beam.
        const auto lidarIs = [&](const std::array<bool, 5>& h, std::initializer_list<int> want) {
            std::array<bool, 5> w{};
            for (int i : want) w[i] = true;
            return h == w;
        };
        const auto probeZMax = [&] {
            float o[3], s[3];
            int d[3];
            if (!renderer.probeGrid(o, s, d)) return -1.f;
            return o[2] + (static_cast<float>(d[2]) - 0.5f) * s[2];
        };

        im = settle();
        reportLower(im, "lens, cyan, orange, quad2 and a far sphere added hidden and resident, plain and late without:");
        const auto s1 = structural();
        const float zMax0 = probeZMax();
        check("none of the four seen", !lensLit(im) && !cyanSeen(im) && !orangeSeen(im) && !quad2Seen(im));
        check("red, green and quad unchanged",
              leads(at(im, pRed, 8), 0) && leads(at(im, pGreen, 8), 1) && leads(at(im, pQuad, 2), 2));
        auto h = lidarHits();
        check("no lidar return from the hidden spheres (TLAS mask 0)", lidarIs(h, {}));

        lens->visible = true;
        quad2->visible = true;
        farBall->visible = true;
        im = settle();
        reportLower(im, "lens, quad2 and the far sphere shown for the first time (Object3D::visible):");
        const float zMax1 = probeZMax();
        std::printf("  probe grid z max %.2f -> %.2f\n", zMax0, zMax1);
        check("lens lit in the mirror (TLAS)", lensLit(im));
        check("quad2 drawn (raster)", quad2Seen(im));
        h = lidarHits();
        check("lidar returns from the lens only", lidarIs(h, {kLens}));
        check("probe grid re-fitted around the far sphere", zMax0 < 11.5f && zMax1 >= 12.4f);
        check("no structural rebuild", structural() == s1);

        cyan->material()->visible = true;
        hiddenGroup->visible = true;
        im = settle();
        reportLower(im, "cyan (material) and orange (group) shown for the first time:");
        check("cyan and orange in the mirror", cyanSeen(im) && orangeSeen(im));
        check("lens and quad2 still there", lensLit(im) && quad2Seen(im));
        h = lidarHits();
        check("lidar returns from lens, cyan and orange", lidarIs(h, {kLens, kCyan, kOrange}));
        check("no structural rebuild", structural() == s1);

        // The hint set on a mesh already rendered hidden: the next render
        // builds its entries, hidden (the structural rebuild its first show
        // would have been), and the show after that is in place.
        late->resident = true;
        im = settle();
        reportLower(im, "late: resident set while it is hidden:");
        h = lidarHits();
        check("late still hidden (TLAS mask 0)", !h[kLate]);
        check(firstShow ? "one structural rebuild (late's entries)" : "no structural rebuild (late's entries)",
              structural() == s1 + firstShow);
        late->visible = true;
        im = settle();
        reportLower(im, "late shown for the first time:");
        h = lidarHits();
        check("lidar returns from late", h[kLate]);
        check("no further structural rebuild", structural() == s1 + firstShow);

        // Without the hint a first show is a slot edit (without slots, the
        // structural rebuild).
        plain->visible = true;
        im = settle();
        reportLower(im, "plain (never resident) shown for the first time:");
        h = lidarHits();
        check("lidar returns from plain", h[kPlain]);
        check(firstShow ? "one structural rebuild (plain's first show)"
                        : "no structural rebuild (plain's first show)",
              structural() == s1 + 2u * firstShow);

        lens->visible = false;
        cyan->material()->visible = false;
        hiddenGroup->visible = false;
        quad2->visible = false;
        farBall->visible = false;
        plain->visible = false;
        late->visible = false;
        im = settle();
        reportLower(im, "all hidden again:");
        h = lidarHits();
        check("no lidar return from the five (TLAS mask 0)", lidarIs(h, {}));
        check("quad2 gone from the raster", !quad2Seen(im));
        check("no structural rebuild", structural() == s1 + 2u * firstShow);

        // Clearing the hint keeps the entries a hidden mesh has, as for any
        // mesh that was shown: the re-show stays in place.
        lens->resident = false;
        im = settle();
        lens->visible = true;
        im = settle();
        reportLower(im, "lens: resident cleared while hidden, then shown:");
        h = lidarHits();
        check("lidar returns from the lens only", lidarIs(h, {kLens}));
        check("no structural rebuild", structural() == s1 + 2u * firstShow);
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
