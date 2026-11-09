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

#include <algorithm>
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
    // lastSpokes() holds only the last call's, so `spokes` gathers them all.
    unsigned runAt(MarineRadar& radar, VulkanRenderer& r, double t,
                   std::vector<RadarSpoke>* spokes = nullptr) {
        radar.setSimTime(t);
        unsigned formed = 0;
        for (int i = 0; i < 40; ++i) {
            formed += radar.update(r);
            if (spokes) spokes->insert(spokes->end(), radar.lastSpokes().begin(), radar.lastSpokes().end());
        }
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

    std::vector<RadarSpoke> spokes;
    CHECK(runAt(radar, *r, 0.5, &spokes) > 0u);
    REQUIRE(!spokes.empty());
    for (const auto& sp : spokes) REQUIRE(sp.snrDb.size() == 512u);

    // Mid-sweep, with slots of the old geometry in flight: a range-scale
    // switch, and the ring grows four times over.
    radar.update(*r);
    radar.model.rangeBins = 2048;
    radar.model.maxRange = 600.f;
    spokes.clear();
    CHECK(runAt(radar, *r, 1.0, &spokes) > 0u);
    CHECK(radar.image().bins == 2048u);
    CHECK(radar.image().maxRange == 600.f);
    REQUIRE(!spokes.empty());
    for (const auto& sp : spokes) CHECK(sp.snrDb.size() == 2048u);
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

TEST_CASE("MarineRadar: a clock set back by less than a revolution restarts the sweep") {
    REQUIRE_RT_GPU(r);
    Scene scene;
    auto target = box(20.f, 20.f, 20.f);
    target->position.set(0, 10, -150);
    scene.add(target);
    build(*r, scene);

    MarineRadar radar(smallRadar());// one revolution a second
    radar.position.set(0, 10, 0);
    radar.updateMatrixWorld();

    // Two episodes of 0.8 s, the clock reset to zero between them.
    radar.setSimTime(0.0);
    radar.update(*r);
    CHECK(runAt(radar, *r, 0.8) > 0u);

    radar.setSimTime(0.0);
    radar.update(*r);
    CHECK(runAt(radar, *r, 0.4) > 0u);
    // The picture is of the second episode: nothing from later than its clock.
    const auto& image = radar.image();
    double latest = -1.0;
    for (const double t : image.time) latest = std::max(latest, t);
    CHECK(latest >= 0.0);
    CHECK(latest <= 0.4);
}

TEST_CASE("MarineRadar: a clock set back drops the trace in flight instead of waiting for it") {
    REQUIRE_RT_GPU(r);
    Scene scene;
    auto target = box(20.f, 20.f, 20.f);
    target->position.set(0, 10, -150);
    scene.add(target);
    build(*r, scene);

    MarineRadar radar(smallRadar());
    radar.position.set(0, 10, 0);
    radar.updateMatrixWorld();
    radar.setSimTime(0.0);
    radar.update(*r);
    runAt(radar, *r, 0.8);

    // The antenna moves on: the slots ahead of it go out as a new trace.
    radar.setSimTime(0.9);
    radar.update(*r);
    const std::uint64_t collected = radar.scansCollected();

    // An episode reset with that trace still out: it is of the old timeline,
    // so the reset neither reads it back nor waits on its fence.
    radar.setSimTime(0.0);
    radar.update(*r);
    CHECK(radar.scansCollected() == collected);

    // The new episode's own traces are taken delivery of as usual.
    CHECK(runAt(radar, *r, 0.4) > 0u);
    CHECK(radar.scansCollected() > collected);
}

namespace {

    // Spokes of the picture that hold a formed spoke, and the latest one's time.
    unsigned filledSpokes(const RadarPolarImage& img) {
        return static_cast<unsigned>(std::count_if(img.time.begin(), img.time.end(), [](double t) { return t >= 0.0; }));
    }
    double latestSpoke(const RadarPolarImage& img) {
        double latest = -1.0;
        for (const double t : img.time) latest = std::max(latest, t);
        return latest;
    }

}// namespace

TEST_CASE("MarineRadar: a lower rpm mid-run keeps the picture and the sweep") {
    REQUIRE_RT_GPU(r);
    Scene scene;
    auto target = box(20.f, 20.f, 20.f);
    target->position.set(0, 10, -150);
    scene.add(target);
    build(*r, scene);

    MarineRadar radar(smallRadar());// 60 rpm
    radar.position.set(0, 10, 0);
    radar.updateMatrixWorld();
    radar.setSimTime(0.0);
    radar.update(*r);
    runAt(radar, *r, 0.8);
    const unsigned filled = filledSpokes(radar.image());
    REQUIRE(filled > 0u);

    // Half speed from here: the antenna slows, the clock still runs forward.
    radar.model.rpm = 30.f;
    radar.setSimTime(0.85);
    radar.update(*r);
    CHECK(filledSpokes(radar.image()) >= filled);

    // It sweeps on at the new rate: fresh spokes, none later than the clock.
    CHECK(runAt(radar, *r, 1.25) > 0u);
    CHECK(latestSpoke(radar.image()) > 0.85);
    CHECK(latestSpoke(radar.image()) <= 1.25);

    // A spin-down: a lower rpm on every frame. The picture stays.
    double t = 1.25;
    for (int i = 0; i < 10; ++i) {
        t += 0.05;
        radar.model.rpm *= 0.85f;
        radar.setSimTime(t);
        radar.update(*r);
        INFO("frame " << i << " at " << radar.model.rpm << " rpm");
        CHECK(filledSpokes(radar.image()) >= filled);
    }
    CHECK(latestSpoke(radar.image()) <= t);
}

namespace {

    // The strongest echo's bearing (deg) in a frame.
    float peakBearing(const SonarFrame& f) {
        const auto peak = static_cast<std::size_t>(std::max_element(f.levelDb.begin(), f.levelDb.end()) -
                                                   f.levelDb.begin());
        return f.bearingsDeg[peak / f.bins];
    }

    // The strongest SNR (dB) of the picture between two ranges.
    float peakSnrBetween(const RadarPolarImage& img, float r0, float r1) {
        float best = -1e9f;
        const float binW = (img.maxRange - img.minRange) / static_cast<float>(img.bins);
        for (unsigned s = 0; s < img.spokes; ++s) {
            if (img.time[s] < 0.0) continue;
            for (unsigned j = 0; j < img.bins; ++j) {
                const float R = img.minRange + (static_cast<float>(j) + 0.5f) * binW;
                if (R < r0 || R > r1) continue;
                best = std::max(best, img.snrDb[static_cast<std::size_t>(s) * img.bins + j]);
            }
        }
        return best;
    }

}// namespace

TEST_CASE("ImagingSonar: a scaled parent does not pull the fan toward broadside") {
    REQUIRE_RT_GPU(r);
    Scene scene;
    // 20 deg to starboard, 10 m out.
    auto target = box(1.f, 4.f, 1.f);
    target->position.set(static_cast<float>(10.0 * std::sin(20.0 * kDeg)), 0.f,
                         static_cast<float>(-10.0 * std::cos(20.0 * kDeg)));
    scene.add(target);
    build(*r, scene);

    SonarFrame plain, scaled;
    ImagingSonar level(smallSonar());
    level.updateMatrixWorld();
    level.ping(*r, plain);

    // The same sonar, at the same place, under a centimetre root.
    auto root = std::make_shared<Object3D>();
    root->scale.setScalar(0.01f);
    auto sonar = std::make_shared<ImagingSonar>(smallSonar());
    root->add(sonar);
    root->updateMatrixWorld(true);
    sonar->ping(*r, scaled);

    INFO("unscaled peak at " << peakBearing(plain) << " deg, scaled at " << peakBearing(scaled) << " deg");
    CHECK(std::abs(peakBearing(plain) - 20.f) < 3.f);
    CHECK(std::abs(peakBearing(scaled) - peakBearing(plain)) < 0.5f);
}

TEST_CASE("MarineRadar: a scaled parent does not widen the vertical beam") {
    REQUIRE_RT_GPU(r);
    Scene scene;
    // Dead ahead, 100 m out and 16 deg up: on the vertical pattern's skirt.
    auto target = box(20.f, 10.f, 20.f);
    target->position.set(0.f, static_cast<float>(10.0 + 100.0 * std::tan(16.0 * kDeg)), -100.f);
    scene.add(target);
    build(*r, scene);

    auto model = smallRadar();
    model.horizonShare = 0.f;// every ray spread evenly over the vertical fan
    model.aboveHorizonShare = 0.f;

    MarineRadar level(model);
    level.position.set(0, 10, 0);
    level.updateMatrixWorld();
    level.setSimTime(0.0);
    level.update(*r);
    runAt(level, *r, 0.99);

    auto root = std::make_shared<Object3D>();
    root->scale.setScalar(0.01f);
    auto radar = std::make_shared<MarineRadar>(model);
    radar->position.set(0, 1000, 0);// 10 m in the world
    root->add(radar);
    root->updateMatrixWorld(true);
    radar->setSimTime(0.0);
    radar->update(*r);
    runAt(*radar, *r, 0.99);

    const float a = peakSnrBetween(level.image(), 80.f, 120.f);
    const float b = peakSnrBetween(radar->image(), 80.f, 120.f);
    INFO("target SNR " << a << " dB unscaled, " << b << " dB under the scaled root");
    CHECK(a > 10.f);
    CHECK(std::abs(a - b) < 0.5f);
}
