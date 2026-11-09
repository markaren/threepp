// VulkanDepthAov_test: the Depth AOV decoded to planar view-space depth
// (VulkanRenderer::decodeDepthAOV, which Python's read_depth and
// read_aovs_typed use). The renderer stores reverse-Z depth, which is
// hyperbolic in the distance under a perspective projection and linear under
// an orthographic one; decoding an orthographic camera's depth with the
// perspective formula read a ground 100 m below as 0.111 m.
//
// The decode against the projections' own formulas first, then end to end: a
// ground plane 100 m below a camera looking straight down, read back from the
// G-buffer, under both projections. The second case SKIPs without a
// ray-tracing Vulkan device.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <cstring>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinRel;

namespace {

    constexpr float kNear = 0.1f, kFar = 1000.f;

    float decode(float z, const Camera& camera) {
        float d = 0.f;
        VulkanRenderer::decodeDepthAOV(&z, 1, camera, &d);
        return d;
    }

}// namespace

TEST_CASE("Depth AOV: reverse-Z decodes to the planar depth under either projection") {
    const PerspectiveCamera persp(60.f, 1.f, kNear, kFar);
    const OrthographicCamera ortho(-50.f, 50.f, 50.f, -50.f, kNear, kFar);
    for (const float d : {0.5f, 10.f, 100.f, 900.f}) {
        // What reverseZVk makes of each projection at view depth d.
        const float zPersp = kNear * (kFar - d) / (d * (kFar - kNear));
        const float zOrtho = (kFar - d) / (kFar - kNear);
        CHECK_THAT(decode(zPersp, persp), WithinRel(d, 1e-4f));
        CHECK_THAT(decode(zOrtho, ortho), WithinRel(d, 1e-4f));
    }
    // The cleared background reads as the far plane.
    CHECK(decode(0.f, persp) == kFar);
    CHECK(decode(0.f, ortho) == kFar);
}

TEST_CASE("Depth AOV: a ground 100 m below reads 100 m from either camera") {
    constexpr int kW = 64, kH = 64;
    auto gpu = vktest::openHeadlessVulkan("VulkanDepthAov_test", kW, kH);
    if (!gpu || !gpu.renderer->rayTracingSupported()) {
        SKIP("no Vulkan ray tracing device: nothing to render");
    }
    VulkanRenderer& renderer = *gpu.renderer;
    renderer.setOrthographicSceneRendering(true);// the ortho camera takes the deferred path

    Scene scene;
    auto ground = Mesh::create(PlaneGeometry::create(400.f, 400.f), MeshStandardMaterial::create());
    ground->rotation.x = -3.14159265f / 2.f;
    scene.add(ground);
    scene.add(HemisphereLight::create());

    auto centreDepth = [&](Camera& camera) {
        camera.position.set(0.f, 100.f, 0.f);
        camera.lookAt(Vector3(0.f, 0.f, 0.f));
        camera.updateMatrixWorld();
        for (int i = 0; i < 3; ++i) gpu.canvas->animateOnce([&] { renderer.render(scene, camera); });
        std::vector<std::uint8_t> raw;
        int w = 0, h = 0, bpp = 0;
        REQUIRE(renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::Depth, raw, w, h, bpp));
        REQUIRE(bpp == 4);
        std::vector<float> depth(static_cast<std::size_t>(w) * h);
        VulkanRenderer::decodeDepthAOV(reinterpret_cast<const float*>(raw.data()), depth.size(), camera,
                                       depth.data());
        return depth[static_cast<std::size_t>(h / 2) * w + w / 2];
    };

    PerspectiveCamera persp(60.f, 1.f, kNear, kFar);
    OrthographicCamera ortho(-50.f, 50.f, 50.f, -50.f, kNear, kFar);
    // The orthographic frame first: what the AOV holds must be its own.
    const float fromOrtho = centreDepth(ortho);
    const float fromPersp = centreDepth(persp);
    INFO("perspective " << fromPersp << " m, orthographic " << fromOrtho << " m");
    CHECK_THAT(fromPersp, WithinRel(100.f, 1e-3f));
    CHECK_THAT(fromOrtho, WithinRel(100.f, 1e-3f));
}
