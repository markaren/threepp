// The frame loop of ImagingSonar and MarineRadar on the Vulkan ray tracer:
// what EchoSensors_test cannot reach without a GPU. The scenes are a box or
// two and a few hundred rays a scan. Every case skips (the message the CTest
// SKIP_REGULAR_EXPRESSION matches) without a ray-tracing Vulkan device.

#include <catch2/catch_test_macros.hpp>

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/helpers/ImagingSonar.hpp"
#include "threepp/helpers/MarineRadar.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <cmath>
#include <memory>

using namespace threepp;

namespace {

    constexpr double kDeg = 3.14159265358979323846 / 180.0;

    // One headless renderer for the whole binary. Never torn down: the
    // process exit does that, after Catch2 has reported.
    VulkanRenderer* gpu() {
        static vktest::HeadlessVulkan* vk = new vktest::HeadlessVulkan(
                vktest::openHeadlessVulkan("EchoSensorsVulkan_test", 64, 64));
        if (!*vk || !vk->renderer->rayTracingSupported()) return nullptr;
        return vk->renderer.get();
    }

#define REQUIRE_RT_GPU(r)                                                  \
    VulkanRenderer* r = gpu();                                             \
    if (!r) SKIP("no Vulkan ray tracing device: the sensors cannot trace")

    std::shared_ptr<Mesh> box(float w, float h, float d) {
        return Mesh::create(BoxGeometry::create(w, h, d), MeshStandardMaterial::create());
    }

    // Render so the tracer has the scene's acceleration structure.
    void build(VulkanRenderer& r, Scene& scene) {
        PerspectiveCamera camera(60.f, 1.f, 0.1f, 5000.f);
        camera.position.set(0, 50, 200);
        scene.updateMatrixWorld();
        for (int i = 0; i < 3; ++i) r.render(scene, camera);
    }

    // A small, quick radar: few spokes and rays, a short range.
    MarineRadarModel smallRadar() {
        MarineRadarModel m;
        m.spokes = 64;
        m.azimuthSamples = 1;
        m.elevationSamples = 32;
        m.minRange = 5.f;
        m.maxRange = 300.f;
        m.rangeBins = 128;
        m.rpm = 60.f;// one revolution a second
        return m;
    }

    // Holds the clock at `t` and calls update() until it forms nothing more
    // (every slot the antenna has passed is traced, collected and formed).
    unsigned runAt(MarineRadar& radar, VulkanRenderer& r, double t) {
        radar.setSimTime(t);
        unsigned formed = 0;
        for (int i = 0; i < 40; ++i) formed += radar.update(r);
        return formed;
    }

    // A quiet sonar with a small ray table.
    ImagingSonarModel smallSonar() {
        ImagingSonarModel m;
        m.horizontalFov = 60.f;
        m.beams = 64;
        m.horizontalBeamwidth = 2.f;
        m.verticalBeamwidth = 20.f;
        m.azimuthOversample = 2;
        m.verticalSamples = 8;
        m.minRange = 0.5f;
        m.maxRange = 20.f;
        m.rangeBins = 256;
        m.rangeResolution = 0.05f;
        m.noiseSpectrumLevelDb = -150.f;
        m.volumeBackscatterDb = -300.f;
        m.surfaceMultipath = false;
        return m;
    }

    // Hits to starboard of x = `minX` (world).
    int hitsBeyond(const std::vector<LidarReturn>& returns, float minX) {
        int n = 0;
        for (const auto& h : returns) n += h.returnNo > 0 && h.returnKind == 0 && h.position.x > minX;
        return n;
    }

}// namespace

TEST_CASE("MarineRadar: a new range scale rebuilds the ring instead of writing past it") {
    REQUIRE_RT_GPU(r);
    Scene scene;
    auto target = box(20.f, 20.f, 20.f);
    target->position.set(0, 10, -150);
    scene.add(target);
    build(*r, scene);

    MarineRadar radar(smallRadar());
    radar.position.set(0, 10, 0);
    radar.updateMatrixWorld();

    // Set before the first update: the picture and the ring follow the model.
    radar.model.rangeBins = 512;
    radar.setSimTime(0.0);
    radar.update(*r);
    REQUIRE(radar.image().bins == 512u);

    CHECK(runAt(radar, *r, 0.5) > 0u);
    for (const auto& sp : radar.lastSpokes()) REQUIRE(sp.snrDb.size() == 512u);

    // Mid-sweep, with slots of the old geometry in flight: a range-scale
    // switch, and the ring grows four times over.
    radar.update(*r);
    radar.model.rangeBins = 2048;
    radar.model.maxRange = 600.f;
    CHECK(runAt(radar, *r, 1.0) > 0u);
    CHECK(radar.image().bins == 2048u);
    CHECK(radar.image().maxRange == 600.f);
    for (const auto& sp : radar.lastSpokes()) CHECK(sp.snrDb.size() == 2048u);
}

TEST_CASE("ImagingSonar: new fan geometry re-bakes the ray table at the next ping") {
    REQUIRE_RT_GPU(r);
    Scene scene;
    // Dead ahead, 10 m out: inside any fan.
    auto ahead = box(2.f, 4.f, 1.f);
    ahead->position.set(0.f, 0.f, -10.f);
    scene.add(ahead);
    // 45 deg to starboard, 10 m out: outside a 60 deg fan, inside a 120 deg one.
    auto target = box(1.f, 4.f, 1.f);
    target->position.set(static_cast<float>(10.0 * std::sin(45.0 * kDeg)), 0.f,
                         static_cast<float>(-10.0 * std::cos(45.0 * kDeg)));
    scene.add(target);
    build(*r, scene);

    ImagingSonar sonar(smallSonar());
    sonar.updateMatrixWorld();
    SonarFrame frame;
    sonar.ping(*r, frame);
    REQUIRE(hitsBeyond(sonar.lastReturns(), -1e9f) > 0);// the box ahead
    CHECK(hitsBeyond(sonar.lastReturns(), 5.f) == 0);

    sonar.model.horizontalFov = 120.f;
    sonar.model.verticalSamples = 12;
    sonar.ping(*r, frame);
    CHECK(sonar.rayCount() == 64u * 2u * 12u);
    CHECK(hitsBeyond(sonar.lastReturns(), 5.f) > 0);
}
