// VulkanInstanceColor_test — InstancedMesh::setColorAt tints each instance.
//
// three.js multiplies the material's diffuse color by the instance color. The
// Vulkan backend read only the material, so every instance of an InstancedMesh
// rendered in the material's color whatever setColorAt said.
//
// Two rows of three white quads in a uniform environment, each row one
// InstancedMesh:
//   top    — colored red/green/blue before the first frame (full scene build),
//   bottom — no instance colors until the first check, then colored (the
//            attribute is created lazily by setColorAt: change detection path).
// After the first check the top row's first instance is recolored green
// (instanceColor()->needsUpdate() on an existing attribute: the patch path).
//
// Plain exit-code program, not Catch2 — same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
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

    constexpr int kW = 384, kH = 256;
    constexpr int kFramesPerPhase = 60;
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

    // Mean RGB of a (2r+1)^2 patch, 0..255.
    std::array<double, 3> patch(const std::vector<unsigned char>& px, int cx, int cy, int r) {
        std::array<double, 3> sum{};
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                for (int c = 0; c < 3; ++c) sum[c] += px[(static_cast<size_t>(y) * kW + x) * 3 + c];
                ++n;
            }
        }
        for (auto& s : sum) s /= n;
        return sum;
    }

    // Index of the dominant channel, or -1 when no channel leads the other two
    // by 2x (grey / white).
    int dominant(const std::array<double, 3>& c) {
        for (int i = 0; i < 3; ++i) {
            const double o1 = c[(i + 1) % 3], o2 = c[(i + 2) % 3];
            if (c[i] > 2.0 * o1 + 4.0 && c[i] > 2.0 * o2 + 4.0) return i;
        }
        return -1;
    }

    std::shared_ptr<InstancedMesh> makeRow(float y) {
        auto mat = MeshStandardMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->roughness = 1.f;
        mat->metalness = 0.f;
        auto mesh = InstancedMesh::create(PlaneGeometry::create(1.f, 1.f), mat, 3);
        Matrix4 m;
        for (size_t i = 0; i < 3; ++i) {
            m.makeTranslation((static_cast<float>(i) - 1.f) * 1.5f, y, 0.f);
            mesh->setMatrixAt(i, m);
        }
        mesh->instanceMatrix()->needsUpdate();
        mesh->computeBoundingSphere();
        return mesh;
    }

    void colorRow(InstancedMesh& mesh, const std::array<Color, 3>& colors) {
        for (size_t i = 0; i < 3; ++i) mesh.setColorAt(i, colors[i]);
        mesh.instanceColor()->needsUpdate();
    }

}// namespace

int main() {

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanInstanceColor_test").size(kW, kH).vsync(false).headless(true));
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

    // 3:2 frame, two rows of three unit quads 1.5 apart.
    PerspectiveCamera camera(40.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0, 4.4f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    const std::array<Color, 3> rgb{Color(1, 0, 0), Color(0, 1, 0), Color(0, 0, 1)};

    auto top = makeRow(0.75f);
    colorRow(*top, rgb);
    scene.add(top);

    auto bottom = makeRow(-0.75f);
    scene.add(bottom);

    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::printf("  %-56s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    // Quad centres in pixels: columns at the thirds' centres, rows found by
    // projecting the row heights.
    const auto sample = [&](const std::vector<unsigned char>& px, int row, int col) {
        Vector3 p((static_cast<float>(col) - 1.f) * 1.5f, row == 0 ? 0.75f : -0.75f, 0.f);
        p.project(camera);
        const int x = static_cast<int>((p.x * 0.5f + 0.5f) * kW);
        const int y = static_cast<int>((1.f - (p.y * 0.5f + 0.5f)) * kH);
        return patch(px, x, y, 4);
    };
    const auto report = [&](const std::vector<unsigned char>& px, const char* label) {
        std::printf("%s\n", label);
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 3; ++col) {
                const auto c = sample(px, row, col);
                std::printf("  %s[%d] = (%5.1f %5.1f %5.1f)\n", row == 0 ? "top   " : "bottom", col, c[0], c[1], c[2]);
            }
        }
    };

    int frame = 0;
    canvas.animate([&] {
        renderer.render(scene, camera);
        ++frame;

        if (frame == kFramesPerPhase) {
            const auto px = renderer.readRGBPixels();
            if (px.size() < static_cast<size_t>(kW) * kH * 3) {
                std::printf("readback too small: %zu\n", px.size());
                std::exit(1);
            }
            report(px, "phase 1 (top colored at build, bottom uncolored):");
            check("top[0] is red", dominant(sample(px, 0, 0)) == 0);
            check("top[1] is green", dominant(sample(px, 0, 1)) == 1);
            check("top[2] is blue", dominant(sample(px, 0, 2)) == 2);
            for (int col = 0; col < 3; ++col) check("bottom is untinted", dominant(sample(px, 1, col)) == -1);

            colorRow(*bottom, rgb);                // lazily creates the attribute
            top->setColorAt(0, Color(0, 1, 0));    // edits an existing one
            top->instanceColor()->needsUpdate();
            return;
        }

        if (frame == 2 * kFramesPerPhase) {
            const auto px = renderer.readRGBPixels();
            report(px, "phase 2 (bottom colored late, top[0] recolored green):");
            check("top[0] recolored green", dominant(sample(px, 0, 0)) == 1);
            check("top[1] still green", dominant(sample(px, 0, 1)) == 1);
            check("top[2] still blue", dominant(sample(px, 0, 2)) == 2);
            check("bottom[0] is red", dominant(sample(px, 1, 0)) == 0);
            check("bottom[1] is green", dominant(sample(px, 1, 1)) == 1);
            check("bottom[2] is blue", dominant(sample(px, 1, 2)) == 2);

            std::printf("instance color: %d failure(s)\n", failures);
            std::exit(failures == 0 ? 0 : 1);
        }
    });
    return 0;
}
