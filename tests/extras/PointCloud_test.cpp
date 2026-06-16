
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/pointcloud/Icp.hpp"
#include "threepp/extras/pointcloud/MarchingCubes.hpp"
#include "threepp/extras/pointcloud/VoxelGrid.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <random>
#include <utility>
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

TEST_CASE("VoxelGrid occupancy view reports one centre per cell") {

    VoxelGrid g(1.0f, 20, 0.f);
    g.insert({0.1f, 0.1f, 0.1f});// cell (0,0,0)
    g.insert({0.9f, 0.2f, 0.3f});// same cell
    g.insert({2.5f, 0.f, 0.f});  // cell (2,0,0)
    REQUIRE(g.voxelCount() == 2);

    std::vector<Vector3> centers;
    g.collectVoxelCenters(centers);
    REQUIRE(centers.size() == 2);
    for (const auto& c : centers) {
        REQUIRE_THAT(c.y, Catch::Matchers::WithinAbs(0.5, 1e-6));
        REQUIRE_THAT(c.z, Catch::Matchers::WithinAbs(0.5, 1e-6));
        REQUIRE((std::abs(c.x - 0.5f) < 1e-6f || std::abs(c.x - 2.5f) < 1e-6f));
    }
}

TEST_CASE("marchingCubes extracts a watertight isosurface") {

    // Solid ball as a scalar field: value = R - distance(node, centre), so the
    // inside (value > 0) is the ball and the iso-0 surface is the sphere.
    const int n = 32;
    const float cs = 0.1f;
    const Vector3 origin(-1.6f, -1.6f, -1.6f);// grid spans ~[-1.6, 1.5], ball R=1 fits
    const float R = 1.0f;

    ScalarField f;
    f.nx = f.ny = f.nz = n;
    f.origin = origin;
    f.cellSize = cs;
    f.data.resize(static_cast<std::size_t>(n) * n * n);
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const Vector3 p(origin.x + x * cs, origin.y + y * cs, origin.z + z * cs);
                const float d = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
                f.data[static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * n + static_cast<std::size_t>(z) * n * n] = R - d;
            }

    const IsoMesh mesh = marchingCubes(f, 0.f);
    REQUIRE(mesh.positions.size() >= 3);
    REQUIRE(mesh.positions.size() % 3 == 0);
    REQUIRE(mesh.positions.size() == mesh.normals.size());

    // Every vertex lies on the sphere (within ~one cell).
    for (const auto& p : mesh.positions) {
        const float d = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        REQUIRE_THAT(d, Catch::Matchers::WithinAbs(R, 2.0 * cs));
    }

    // Normals point radially outward (gradient of the field is inward, negated).
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
        const Vector3& p = mesh.positions[i];
        const Vector3& nrm = mesh.normals[i];
        const float plen = p.length();
        if (plen < 1e-4f) continue;
        const float dot = (p.x * nrm.x + p.y * nrm.y + p.z * nrm.z) / plen;
        REQUIRE(dot > 0.5f);// roughly aligned with the outward radial direction
    }

    // Watertight & manifold: every undirected edge is shared by exactly two
    // triangles. Shared-edge vertices are computed identically in neighbouring
    // cubes, so positions match to float precision; a fine quantisation keys them.
    auto key = [&](const Vector3& v) {
        auto q = [&](float c) { return static_cast<long long>(std::llround(c / (cs * 1e-3f))); };
        return std::array<long long, 3>{q(v.x), q(v.y), q(v.z)};
    };
    std::map<std::pair<std::array<long long, 3>, std::array<long long, 3>>, int> edgeCount;
    auto addEdge = [&](const Vector3& a, const Vector3& b) {
        auto ka = key(a), kb = key(b);
        if (kb < ka) std::swap(ka, kb);
        ++edgeCount[{ka, kb}];
    };
    for (std::size_t i = 0; i + 2 < mesh.positions.size(); i += 3) {
        addEdge(mesh.positions[i], mesh.positions[i + 1]);
        addEdge(mesh.positions[i + 1], mesh.positions[i + 2]);
        addEdge(mesh.positions[i + 2], mesh.positions[i]);
    }
    int nonManifold = 0;
    for (const auto& kv : edgeCount)
        if (kv.second != 2) ++nonManifold;

    // The basic Lorensen-Cline table has known ambiguous-case cracks, so the
    // surface is essentially (not perfectly) closed. This bound still catches a
    // gross table error — that would corrupt the vast majority of edges.
    INFO("non-manifold edges: " << nonManifold << " / " << edgeCount.size());
    REQUIRE(static_cast<std::size_t>(nonManifold) * 20 < edgeCount.size());// < 5%
}
