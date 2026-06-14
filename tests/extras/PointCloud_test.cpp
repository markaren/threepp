
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/pointcloud/Icp.hpp"
#include "threepp/extras/pointcloud/VoxelGrid.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace threepp;

TEST_CASE("voxelDownsample collapses points sharing a voxel") {

    const std::vector<Vector3> pts = {
            {0, 0, 0}, {0.1f, 0, 0}, {0.2f, 0, 0},// one 0.5 m voxel
            {1.f, 0, 0}, {1.1f, 0, 0},            // another 0.5 m voxel
            {5, 5, 5}};                           // far voxel

    REQUIRE(voxelDownsample(pts, 0.5f).size() == 3);

    // Points spaced wider than the voxel are all preserved.
    std::vector<Vector3> spread;
    for (int i = 0; i < 10; ++i) spread.emplace_back(static_cast<float>(i), 0.f, 0.f);
    REQUIRE(voxelDownsample(spread, 0.5f).size() == 10);
}

TEST_CASE("VoxelGrid nearest returns the closest stored point") {

    VoxelGrid grid(1.0f);// default cap, no spacing filter
    grid.insert({0, 0, 0});
    grid.insert({0.3f, 0, 0});
    grid.insert({5, 0, 0});
    REQUIRE(grid.size() == 3);

    Vector3 out;
    REQUIRE(grid.nearest({0.1f, 0, 0}, 1.0f, out));
    REQUIRE_THAT(out.x, Catch::Matchers::WithinAbs(0.0, 1e-6));

    REQUIRE(grid.nearest({0.25f, 0, 0}, 1.0f, out));
    REQUIRE_THAT(out.x, Catch::Matchers::WithinAbs(0.3, 1e-6));

    REQUIRE_FALSE(grid.nearest({100, 0, 0}, 1.0f, out));
}

TEST_CASE("VoxelGrid honours capacity and minimum spacing") {

    VoxelGrid capped(1.0f, 2, 0.f);
    REQUIRE(capped.insert({0, 0, 0}));
    REQUIRE(capped.insert({0.1f, 0, 0}));
    REQUIRE_FALSE(capped.insert({0.2f, 0, 0}));// voxel full
    REQUIRE(capped.size() == 2);

    VoxelGrid spaced(1.0f, 20, 0.2f);
    REQUIRE(spaced.insert({0, 0, 0}));
    REQUIRE_FALSE(spaced.insert({0.1f, 0, 0}));// within 0.2 m of an existing point
    REQUIRE(spaced.insert({0.5f, 0, 0}));      // far enough apart
    REQUIRE(spaced.size() == 2);
}

TEST_CASE("icpPointToPoint recovers a known rigid transform") {

    VoxelGrid map(0.5f, 20, 0.f);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-10.f, 10.f), uy(0.f, 3.f);

    std::vector<Vector3> mapPts;
    for (int i = 0; i < 4000; ++i) {
        Vector3 p(u(rng), uy(rng), u(rng));
        mapPts.push_back(p);
        map.insert(p);
    }

    // Ground-truth transform (map <- local): 0.2 m translation + 3 deg yaw.
    Quaternion q;
    q.setFromAxisAngle(Vector3(0, 1, 0), 3.f * math::DEG2RAD);
    Matrix4 tKnown;
    tKnown.compose(Vector3(0.2f, 0.05f, -0.15f), q, Vector3(1, 1, 1));
    Matrix4 tInv;
    tInv.copy(tKnown).invert();

    // Source = tKnown^-1 * map points, with realistic range noise.
    std::normal_distribution<float> noise(0.f, 0.01f);
    std::vector<Vector3> src;
    for (const auto& m : mapPts) {
        Vector3 s = m;
        s.applyMatrix4(tInv);
        s.x += noise(rng);
        s.y += noise(rng);
        s.z += noise(rng);
        src.push_back(s);
    }

    Matrix4 t;// seed at identity (offset is small)
    const IcpResult res = icpPointToPoint(src, map, t);
    REQUIRE(res.correspondences > 1000);

    // Compare full pose (rotation + translation) via probe points.
    float maxErr = 0.f;
    for (const auto& pr : {Vector3(5, 1, 5), Vector3(-5, 2, -5), Vector3(5, 0, -5), Vector3(0, 3, 0)}) {
        Vector3 a = pr;
        a.applyMatrix4(t);
        Vector3 b = pr;
        b.applyMatrix4(tKnown);
        maxErr = std::max(maxErr, a.sub(b).length());
    }
    REQUIRE(maxErr < 0.02f);
}
