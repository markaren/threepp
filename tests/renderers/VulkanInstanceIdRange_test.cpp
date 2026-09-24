// VulkanInstanceIdRange_test — entry indices above 65,535 keep their material.
//
// The G-buffer ids attachment carries the per-frame entry index + 1 in .x, and
// the deferred shade looks the pixel's material up by it. With 16-bit channels
// the index wrapped past 65,535 entries, so a pixel of entry 66,000 was shaded
// with entry 464's material.
//
// Scene: one InstancedMesh of kGrid red unlit quads filling the frame (one
// entry per instance on this backend), then one green unlit quad in front of
// them in the centre, added last so its entry index is above 65,535. The
// centre must shade green, and the Ids AOV must report the large index.
//
// Plain exit-code program, not Catch2 — same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 384, kH = 256;
    constexpr int kFrames = 30;
    constexpr int kSkipCode = 42;

    constexpr int kCols = 300, kRows = 220;// 66,000 instances
    constexpr float kHalfW = 6.5f, kHalfH = 4.3f;

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

}// namespace

int main() {

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanInstanceIdRange_test").size(kW, kH).vsync(false).headless(true));
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
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;
    scene.background = Color(0, 0, 0);

    PerspectiveCamera camera(50.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0, 10.f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    auto red = MeshBasicMaterial::create();
    red->color = Color(1, 0, 0);
    const size_t count = static_cast<size_t>(kCols) * kRows;
    auto grid = InstancedMesh::create(PlaneGeometry::create(0.03f, 0.03f), red, count);
    Matrix4 m;
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const float x = -kHalfW + 2.f * kHalfW * (static_cast<float>(c) + 0.5f) / kCols;
            const float y = -kHalfH + 2.f * kHalfH * (static_cast<float>(r) + 0.5f) / kRows;
            m.makeTranslation(x, y, 0.f);
            grid->setMatrixAt(static_cast<size_t>(r) * kCols + c, m);
        }
    }
    grid->instanceMatrix()->needsUpdate();
    grid->computeBoundingSphere();
    scene.add(grid);

    auto green = MeshBasicMaterial::create();
    green->color = Color(0, 1, 0);
    auto marker = Mesh::create(PlaneGeometry::create(3.f, 2.f), green);
    marker->position.z = 1.f;
    scene.add(marker);

    for (int i = 0; i < kFrames; ++i) canvas.animateOnce([&] { renderer.render(scene, camera); });

    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::printf("  %-56s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    const auto px = renderer.readRGBPixels();
    if (px.size() < static_cast<size_t>(kW) * kH * 3) {
        std::printf("readback too small: %zu\n", px.size());
        return 1;
    }
    const auto centre = patch(px, kW / 2, kH / 2, 4);
    std::printf("centre = (%5.1f %5.1f %5.1f)\n", centre[0], centre[1], centre[2]);
    check("centre quad shades with its own (green) material",
          centre[1] > 128.0 && centre[0] < 32.0 && centre[2] < 32.0);

    std::vector<uint8_t> ids;
    int w = 0, h = 0, bpp = 0;
    const bool gotIds = renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, ids, w, h, bpp);
    check("ids AOV readback is 4x uint32", gotIds && w == kW && h == kH && bpp == 16);
    if (gotIds && w == kW && h == kH && bpp == 16) {
        uint32_t px4[4];
        std::memcpy(px4, ids.data() + (static_cast<size_t>(kH / 2) * kW + kW / 2) * 16u, sizeof(px4));
        std::printf("centre ids = (%u %u %u %u)\n", px4[0], px4[1], px4[2], px4[3]);
        check("centre entry index is above 16 bits", px4[0] > 65536u);
    }

    std::printf("instance id range: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
