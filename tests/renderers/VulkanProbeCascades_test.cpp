// VulkanProbeCascades_test: the probe GI's camera-centred cascades at their
// maximum of four. Each frame's probe update covers every cascade in one
// dispatch, and so does the reset after a window scrolls: at the default sizes
// that is 16384 probes a window, 65536 workgroups in all, one more than a
// device has to accept along one dimension (maxComputeWorkGroupCount[0] is at
// least 65535; NVIDIA and AMD take far more, Intel ANV and lavapipe exactly
// that). The dispatches are two-dimensional, (probes, cascades), so they fit
// any device.
//
// A 2 km ground under a sun, so the scene fit would space its probes far
// apart and the grid becomes four cascades; the camera moves, so the windows
// scroll. The probes must fill and the picture must be lit. The FNV-1a hashes
// of the finest cascade's SH store and of the deferred shade's images (the
// indirect light the cascades feed among them) are printed, to compare two
// builds of the update on one machine: a change to how the work is
// dispatched, and not to the work, leaves them bit for bit the same.

#include <catch2/catch_test_macros.hpp>

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 160, kH = 90;

    template<class T>
    std::uint64_t fnv1a(const std::vector<T>& v) {
        std::uint64_t h = 1469598103934665603ULL;
        const auto* p = reinterpret_cast<const unsigned char*>(v.data());
        for (std::size_t i = 0; i < v.size() * sizeof(T); ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

}// namespace

TEST_CASE("Probe GI: four cascades fill, scroll and light the picture") {
    auto gpu = vktest::openHeadlessVulkan("VulkanProbeCascades_test", kW, kH);
    if (!gpu || !gpu.renderer->rayTracingSupported()) {
        SKIP("no Vulkan ray tracing device: the probes are not traced");
    }
    VulkanRenderer& renderer = *gpu.renderer;
    renderer.setProbeGI(true);
    renderer.setProbeCascades(4);
    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    REQUIRE(renderer.probeCascades() == 4u);

    Scene scene;
    auto groundMat = MeshStandardMaterial::create();
    groundMat->color = Color(0.5f, 0.5f, 0.5f);
    groundMat->roughness = 1.f;
    auto ground = Mesh::create(PlaneGeometry::create(2000.f, 2000.f), groundMat);
    ground->rotation.x = -3.14159265f / 2.f;
    scene.add(ground);
    auto wallMat = MeshStandardMaterial::create();
    wallMat->color = Color(0.8f, 0.3f, 0.2f);
    for (int i = 0; i < 6; ++i) {
        auto wall = Mesh::create(BoxGeometry::create(6.f, 4.f, 1.f), wallMat);
        wall->position.set(-15.f + 6.f * static_cast<float>(i), 2.f, -10.f - 4.f * static_cast<float>(i % 3));
        scene.add(wall);
    }
    auto sun = DirectionalLight::create(Color(0xffffff), 3.f);
    sun->position.set(30.f, 50.f, 20.f);
    scene.add(sun);

    PerspectiveCamera camera(50.f, static_cast<float>(kW) / kH, 0.1f, 3000.f);
    // 40 frames, the camera walking 2 m a frame: the finest window (1 m
    // probes) scrolls every frame, the 4 m one every other.
    for (int f = 0; f < 40; ++f) {
        camera.position.set(2.f * static_cast<float>(f) - 40.f, 2.f, 12.f);
        camera.lookAt(Vector3(camera.position.x, 1.f, -10.f));
        camera.updateMatrixWorld();
        gpu.canvas->animateOnce([&] { renderer.render(scene, camera); });
    }

    float origin[3], spacing[3];
    int dims[3];
    REQUIRE(renderer.probeGrid(origin, spacing, dims));
    CHECK(spacing[0] < 2.f);// the finest cascade, not a scene fit 60 m apart

    std::vector<std::uint8_t> sh;
    REQUIRE(renderer.readProbeShDebug(sh));
    std::size_t lit = 0;
    for (const auto b : sh) lit += b != 0;
    CHECK(lit > sh.size() / 4);

    const auto px = renderer.readRGBPixels();
    REQUIRE(px.size() >= static_cast<std::size_t>(kW) * kH * 3);
    unsigned brightest = 0;
    for (const auto c : px) brightest = std::max<unsigned>(brightest, c);
    CHECK(brightest > 32u);

    std::printf("probe cascades: finest SH store fnv1a %016llx\n", static_cast<unsigned long long>(fnv1a(sh)));
    for (const auto& [name, hash] : renderer.debugHashShadeImages()) {
        std::printf("probe cascades: shade image %s fnv1a %016llx\n", name.c_str(),
                    static_cast<unsigned long long>(hash));
    }
}
