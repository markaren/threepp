// The scan-path equivalences the ranging vision sensors promise on a raster
// backend, pinned as bit-for-bit claims. A static scene scanned with the seed
// reset between scans must produce the SAME cloud through every entry point:
//
//   - scan(cloud) and the RGB-D scan(cloud, colors) share one unprojection,
//     so their clouds are identical and colors pair 1:1 with points;
//   - scanBegin/scanReady/scanCollect is the same scan split in two — on a
//     raster backend the fire IS the scan, and the pair's bookkeeping
//     (pending until collected, nothing owed twice) is part of the contract
//     SensorPlaySession drives both sensors through.
//
// These paths share their skeleton in TracedRasterVisionSensor; this test is
// what notices if a sensor's hook and the skeleton ever drift apart. Bit-exact
// comparisons are legitimate here because the raster path is seeded and
// deterministic — see VisionSensor. Cross-backend claims (never bit-exact)
// live in SensorBackendParity_test.

#include "gl_test_helpers.hpp"

#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/helpers/LidarSensor.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {

    bool sameCloud(const std::vector<Vector3>& a, const std::vector<Vector3>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z) return false;
        }
        return true;
    }

    bool sameCloud(const std::vector<LidarReturn>& a, const std::vector<LidarReturn>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].position.x != b[i].position.x || a[i].position.y != b[i].position.y ||
                a[i].position.z != b[i].position.z || a[i].distance != b[i].distance) return false;
        }
        return true;
    }

    // A few boxes around the origin so every scan has structure to hit, plus a
    // floor so downward beams return too.
    std::shared_ptr<Scene> makeScanScene() {
        auto scene = Scene::create();
        scene->add(HemisphereLight::create(0xffffff, 0x404040, 1.f));

        auto box = [&](float x, float z, int color) {
            auto material = MeshStandardMaterial::create();
            material->color = Color(color);
            auto mesh = Mesh::create(BoxGeometry::create(1, 1, 1), material);
            mesh->position.set(x, 0, z);
            scene->add(mesh);
        };
        box(0, -3, 0xff0000);
        box(2, -4, 0x00ff00);
        box(-2, -5, 0x0000ff);
        scene->add(Mesh::create(BoxGeometry::create(20, 0.2f, 20),
                                MeshStandardMaterial::create()));

        scene->updateMatrixWorld(true);
        return scene;
    }

}// namespace


TEST_CASE("RGB-D and depth-only scans produce the identical cloud", "[sensors]") {

    GLRenderer renderer(glCanvas());
    auto scene = makeScanScene();

    DepthSensor depth(60.f, 96, 72, 0.5f, 20.f);
    depth.position.set(0, 1, 0);
    depth.updateMatrixWorld();

    std::vector<Vector3> cloud, rgbdCloud;
    std::vector<Color> colors;

    depth.resetNoise();
    depth.scan(renderer, *scene, cloud);
    REQUIRE_FALSE(cloud.empty());

    depth.resetNoise();
    depth.scan(renderer, *scene, rgbdCloud, colors);
    REQUIRE(sameCloud(cloud, rgbdCloud));
    REQUIRE(colors.size() == rgbdCloud.size());
}

TEST_CASE("the raster fire/collect pair is the sync scan, delivered the same frame", "[sensors]") {

    GLRenderer renderer(glCanvas());
    auto scene = makeScanScene();

    DepthSensor depth(60.f, 96, 72, 0.5f, 20.f);
    depth.position.set(0, 1, 0);
    depth.updateMatrixWorld();

    std::vector<Vector3> sync, split;
    depth.resetNoise();
    depth.scan(renderer, *scene, sync);
    REQUIRE_FALSE(sync.empty());

    depth.resetNoise();
    REQUIRE(depth.scanBegin(renderer, *scene, split));// raster: cloud in hand
    CHECK(depth.scanPending());                       // ...but a delivery is owed
    CHECK(depth.scanReady(renderer));                 // and can be taken now
    REQUIRE(depth.scanCollect(renderer, split));
    CHECK_FALSE(depth.scanPending());
    CHECK_FALSE(depth.scanCollect(renderer, split));// nothing outstanding twice
    REQUIRE(sameCloud(sync, split));

    // The same split, on the cube-face sensor.
    LidarSensor lidar(64, 0.5f, 20.f);
    lidar.position.set(0, 1, 0);
    lidar.updateMatrixWorld();

    std::vector<LidarReturn> lidarSync, lidarSplit;
    lidar.resetNoise();
    lidar.scan(renderer, *scene, lidarSync);
    REQUIRE_FALSE(lidarSync.empty());

    lidar.resetNoise();
    REQUIRE(lidar.scanBegin(renderer, *scene, lidarSplit));
    REQUIRE(lidar.scanCollect(renderer, lidarSplit));
    REQUIRE(sameCloud(lidarSync, lidarSplit));
}

TEST_CASE("lookAt aims the beams: a sensor beside a cube sees it", "[sensors]") {

    GLRenderer renderer(glCanvas());

    // One cube and nothing else. The sensor images along local -Z, and its
    // usesCameraLookAtConvention() override makes lookAt() turn the BEAMS
    // toward the target; under the plain-Object3D convention (+Z toward the
    // target — what the sensor got before the override) it faces empty space
    // and this cloud comes back empty.
    auto scene = Scene::create();
    scene->add(Mesh::create(BoxGeometry::create(1, 1, 1), MeshStandardMaterial::create()));
    scene->updateMatrixWorld(true);

    DepthSensor sensor(60.f, 64, 48, 0.1f, 20.f);
    sensor.position.set(4, 0, 0);
    sensor.lookAt(0, 0, 0);
    sensor.updateMatrixWorld();

    std::vector<Vector3> cloud;
    sensor.scan(renderer, *scene, cloud);
    REQUIRE_FALSE(cloud.empty());

    // ...and what it sees is the cube: every return on its surface, give or
    // take the default range noise.
    const bool onCube = std::all_of(cloud.begin(), cloud.end(), [](const Vector3& p) {
        return std::abs(p.x) < 0.8f && std::abs(p.y) < 0.8f && std::abs(p.z) < 0.8f;
    });
    REQUIRE(onCube);
}

TEST_CASE("a seed reset replays the identical model-based cloud", "[sensors]") {

    GLRenderer renderer(glCanvas());
    auto scene = makeScanScene();

    LidarSensor lidar(LidarModel::VLP16(), 128, 0.5f, 20.f);
    lidar.position.set(0, 1, 0);
    lidar.updateMatrixWorld();

    std::vector<LidarReturn> first, replayed;
    lidar.resetNoise();
    lidar.scan(renderer, *scene, first);
    REQUIRE_FALSE(first.empty());

    lidar.resetNoise();
    lidar.scan(renderer, *scene, replayed);
    REQUIRE(sameCloud(first, replayed));
}
