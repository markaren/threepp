// VulkanAutoLod_test: an auto-LOD chain lands with ray query and without it,
// and the raster draws fewer triangles at distance.
//
// Every level of a chain used to be a BLAS build, so without ray query
// (THREEPP_VULKAN_NO_RT=1, lavapipe, a GPU without ray tracing) no chain was
// ever queued: the raster drew every mesh at full detail at any distance,
// though its selection pass picks levels for it exactly as with ray query. A
// level is now its index buffer alone when its record has no acceleration
// structure. One blue sphere of ~9,000 triangles, white environment:
//   near  the camera close enough that the sphere fills most of the frame,
//         waiting for its chain to land;
//   far   the camera 80 m back: a coarser level is selected and the G-buffer
//         draw list holds fewer triangles than the full mesh, and the sphere
//         still draws blue (the level's index buffer is the one it reads).
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "vulkan_test_util.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 384, kH = 256;
    constexpr int kMaxWaitFrames = 400;// for the chain to land
    constexpr int kFarFrames = 20;
    constexpr float kNearZ = 2.2f, kFarZ = 80.f;

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

    std::array<double, 3> centrePatch(const std::vector<unsigned char>& px, int r) {
        std::array<double, 3> sum{};
        int n = 0;
        for (int y = kH / 2 - r; y <= kH / 2 + r; ++y) {
            for (int x = kW / 2 - r; x <= kW / 2 + r; ++x) {
                for (int c = 0; c < 3; ++c) sum[c] += px[(static_cast<size_t>(y) * kW + x) * 3 + c];
                ++n;
            }
        }
        for (auto& s : sum) s /= n;
        return sum;
    }

    bool blue(const std::array<double, 3>& c) {
        return c[2] > 2.0 * c[0] + 4.0 && c[2] > 2.0 * c[1] + 4.0;
    }

}// namespace

int main() {

    auto gpu = vktest::openHeadlessVulkan("VulkanAutoLod_test", kW, kH);
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

    const bool rt = renderer.rayTracingSupported();
    std::printf("VulkanAutoLod_test (%s)\n", rt ? "ray query" : "no ray query");

    const auto env = constantEnv(0.5f);
    Scene scene;
    scene.background = env;
    scene.environment = env;

    auto mat = MeshStandardMaterial::create();
    mat->color = Color(0.05f, 0.1f, 0.9f);
    mat->roughness = 1.f;
    auto ball = Mesh::create(SphereGeometry::create(1.f, 96, 48), mat);
    scene.add(ball);
    const auto fullTriangles = static_cast<uint64_t>(ball->geometry()->getIndex()->count() / 3);

    PerspectiveCamera camera(40.f, static_cast<float>(kW) / kH, 0.1f, 200.f);
    camera.position.set(0, 0, kNearZ);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };
    const auto read = [&] {
        auto px = renderer.readRGBPixels();
        if (px.size() < static_cast<size_t>(kW) * kH * 3) {
            std::printf("readback too small: %zu\n", px.size());
            std::exit(1);
        }
        return px;
    };

    int frame = 0;
    int landedAt = -1;
    uint64_t nearTriangles = 0;
    canvas.animate([&] {
        renderer.render(scene, camera);
        ++frame;

        if (landedAt < 0) {
            const auto lod = renderer.autoLodStats();
            if (lod.chainsReady >= 1u) {
                landedAt = frame;
                nearTriangles = lod.trianglesDrawn;
                std::printf("  chain landed on frame %d; near: %llu of %llu triangles drawn\n", frame,
                            static_cast<unsigned long long>(nearTriangles),
                            static_cast<unsigned long long>(fullTriangles));
                check("the sphere's auto-LOD chain landed", true);
                check("near: the full mesh is drawn", nearTriangles == fullTriangles);
                camera.position.set(0, 0, kFarZ);
                camera.lookAt(Vector3{0, 0, 0});
                camera.updateMatrixWorld();
            } else if (frame >= kMaxWaitFrames) {
                std::printf("  after %d frames: %u chain(s) ready, %u queued\n", frame, lod.chainsReady,
                            lod.chainsQueued);
                check("the sphere's auto-LOD chain landed", false);
                std::exit(1);
            }
            return;
        }

        if (frame == landedAt + kFarFrames) {
            const auto lod = renderer.autoLodStats();
            std::printf("  far: %llu triangles drawn; entries per level:", static_cast<unsigned long long>(lod.trianglesDrawn));
            for (uint32_t n : lod.entriesPerLevel) std::printf(" %u", n);
            std::printf("\n");
            check("far: a coarser level is selected", lod.entriesPerLevel[0] == 0u);
            check("far: the raster draws fewer triangles than the full mesh",
                  lod.trianglesDrawn > 0u && lod.trianglesDrawn < fullTriangles / 2u);
            const auto c = centrePatch(read(), 2);
            std::printf("  far: centre = (%5.1f %5.1f %5.1f)\n", c[0], c[1], c[2]);
            check("far: the sphere still draws, blue", blue(c));
            if (vulkan::validationActive()) {
                check("no validation errors", vulkan::validationErrorCount() == 0u);
            }
            std::printf("auto-LOD: %d failure(s)\n", failures);
            std::exit(failures == 0 ? 0 : 1);
        }
    });
    return 0;
}
