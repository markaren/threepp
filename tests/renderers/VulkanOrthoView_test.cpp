// VulkanOrthoView_test — an OrthographicCamera declared a 3D view
// (setOrthographicSceneRendering(true)) takes the deferred path and keeps the
// NEAREST surface, with the scene background behind everything, exactly as the
// same scene under a PerspectiveCamera does.
//
// The scene is the one that exposed the 2026-09-29 symptom (a top-down
// orthophoto of a standard-material model drawn with the farther surface on
// top and no background): a red MeshStandardMaterial box on a blue ground
// plane, a green patch on the box's lid, a hemisphere light, a grey
// background, the camera straight above. The ground is smaller than the
// orthographic frame so one corner of the image is pure background. Both
// projections frame the ground at the same half-extent, so the same pixels
// sample the same surfaces: centre = green patch, a ring out = red lid,
// further out = blue ground, corner = grey background.
//
// Without the flag an ortho render is the 2D overlay path by design
// (painter's order, no depth, no background): that is not what this test
// checks, so the flag is set explicitly.
//
// Plain exit-code program like VulkanGolden_test; exits 42 (CTest "Skipped")
// without a Vulkan/RT GPU. Pass a directory as argv[1] to also write the two
// frames as PNG for a look.

#include "threepp/threepp.hpp"

#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "vulkan_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kSize = 200;
    constexpr int kFrames = 12;// enough for the temporal history to settle

    // 'r', 'g', 'b' when one channel leads both others by a clear margin,
    // 'k' for black (the overlay route's clear colour), 'y' (grey) when the
    // three sit together above that, '?' otherwise.
    char dominant(const std::vector<unsigned char>& px, int x, int y) {
        const size_t i = (static_cast<size_t>(y) * kSize + x) * 3;
        const int r = px[i], g = px[i + 1], b = px[i + 2];
        if (std::max({r, g, b}) < 24) return 'k';
        if (std::max({r, g, b}) - std::min({r, g, b}) < 24) return 'y';
        if (r > g + 40 && r > b + 40) return 'r';
        if (g > r + 40 && g > b + 40) return 'g';
        if (b > r + 40 && b > g + 40) return 'b';
        return '?';
    }

    std::shared_ptr<MeshStandardMaterial> flat(unsigned hex) {
        auto m = MeshStandardMaterial::create();
        m->color = Color(hex);
        m->roughness = 1.f;
        return m;
    }

}// namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto gpu = vktest::openHeadlessVulkan("VulkanOrthoView_test", kSize, kSize);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;
    renderer.setOrthographicSceneRendering(true);

    const std::string outDir = argc > 1 ? argv[1] : "";

    Scene scene;
    scene.background = Color(0x808080);
    scene.add(HemisphereLight::create(0xffffff, 0xffffff, 1.f));

    auto box = Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), flat(0xff0000));
    box->position.y = 0.3f;
    scene.add(box);

    auto lid = Mesh::create(PlaneGeometry::create(0.2f, 0.2f), flat(0x00ff00));
    lid->rotation.x = -math::PI / 2;
    lid->position.y = 0.601f;
    scene.add(lid);

    auto ground = Mesh::create(PlaneGeometry::create(1.6f, 1.6f), flat(0x0000ff));
    ground->rotation.x = -math::PI / 2;
    scene.add(ground);// added LAST: painter's order would put it on top

    // Straight above, looking down, -Z up the image. The perspective camera's
    // half-height at the ground (10 * tan(10 deg) = 1.763) is wider than the
    // ortho frame's 1.0, so its samples are scaled accordingly below.
    auto ortho = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 100.f);
    auto persp = PerspectiveCamera::create(20.f, 1.f, 0.1f, 100.f);
    for (Camera* cam : {static_cast<Camera*>(ortho.get()), static_cast<Camera*>(persp.get())}) {
        cam->position.set(0, 10, 0);
        cam->up.set(0, 0, -1);
        cam->lookAt(Vector3{0, 0, 0});
        cam->updateMatrixWorld();
    }

    struct Sample {
        const char* what;
        float u, v;// image fraction, 0..1 from the top-left, under the ORTHO frame
        char want;
    };
    const Sample samples[] = {
            {"centre: green lid patch", 0.5f, 0.5f, 'g'},
            {"ring: red box lid", 0.5f + 0.22f / 2, 0.5f, 'r'},
            {"ground past the box", 0.5f + 0.5f / 2, 0.5f, 'b'},
            {"corner: scene background", 0.05f, 0.05f, 'y'},
    };

    struct View {
        const char* name;
        Camera* camera;
        float scale;// ortho-frame fraction -> this camera's frame fraction
        std::vector<unsigned char> px;
    };
    View views[] = {
            {"ortho", ortho.get(), 1.f, {}},
            {"persp", persp.get(), 1.f / (10.f * std::tan(10.f * math::PI / 180.f)), {}},
    };

    size_t idx = 0;
    int frame = 0;
    int failures = 0;
    canvas.animate([&] {
        renderer.setSimTime(frame / 60.0);
        renderer.render(scene, *views[idx].camera);
        if (++frame < kFrames) return;
        views[idx].px = renderer.readRGBPixels();
        if (!outDir.empty())
            renderer.writeFramebuffer(std::filesystem::path(outDir) / (std::string("ortho_view_") + views[idx].name + ".png"));
        frame = 0;
        if (++idx < std::size(views)) return;

        for (const View& view : views) {
            std::printf("[%s]\n", view.name);
            for (const Sample& s : samples) {
                const float u = 0.5f + (s.u - 0.5f) * view.scale;
                const float v = 0.5f + (s.v - 0.5f) * view.scale;
                const int x = static_cast<int>(u * kSize), y = static_cast<int>(v * kSize);
                const size_t i = (static_cast<size_t>(y) * kSize + x) * 3;
                const char got = dominant(view.px, x, y);
                const bool ok = got == s.want;
                std::printf("  %-28s (%3d,%3d) rgb=(%3d,%3d,%3d) want %c got %c  %s\n", s.what, x, y, view.px[i],
                            view.px[i + 1], view.px[i + 2], s.want, got, ok ? "PASS" : "FAIL");
                if (!ok) ++failures;
            }
        }
        std::printf("ortho view: %d failed\n", failures);
        std::exit(failures == 0 ? 0 : 1);
    });
    return 0;
}
