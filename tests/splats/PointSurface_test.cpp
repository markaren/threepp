// splats::buildPointSurface: the direct point-cloud collider. No GPU — the
// field and the marching cubes are CPU code, and the checks are geometric.

#include "threepp/math/Matrix4.hpp"
#include "threepp/splats/PointSurface.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

using namespace threepp;
using Catch::Approx;

namespace {

    SplatData fromPoints(const std::vector<Vector3>& pts, float opacity = 1.f) {

        SplatData d;
        d.resize(pts.size(), 0);
        for (size_t i = 0; i < pts.size(); ++i) {
            d.means[i] = pts[i];
            d.scales[i].set(0.01f, 0.01f, 0.01f);
            d.opacities[i] = opacity;
        }
        return d;
    }

    // A sheet in the XZ plane at y = 0, `n` x `n` points at `pitch`.
    std::vector<Vector3> sheet(int n, float pitch) {

        std::vector<Vector3> pts;
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                pts.emplace_back(static_cast<float>(i) * pitch, 0.f, static_cast<float>(k) * pitch);
        return pts;
    }

    // A Fibonacci sphere shell of `n` points at `radius`.
    std::vector<Vector3> shell(int n, float radius) {

        std::vector<Vector3> pts;
        const float golden = 3.14159265f * (3.f - std::sqrt(5.f));
        for (int i = 0; i < n; ++i) {
            const float y = 1.f - 2.f * (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
            const float r = std::sqrt(std::max(0.f, 1.f - y * y));
            const float a = golden * static_cast<float>(i);
            pts.emplace_back(radius * r * std::cos(a), radius * y, radius * r * std::sin(a));
        }
        return pts;
    }

    // Edges shared by other than exactly two triangles: 0 for a closed,
    // manifold surface.
    size_t openEdges(const splats::SurfaceMesh& mesh) {

        std::map<std::pair<uint32_t, uint32_t>, int> count;
        for (size_t t = 0; t < mesh.indices.size(); t += 3) {
            const uint32_t v[3] = {mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]};
            for (int e = 0; e < 3; ++e) {
                const uint32_t a = v[e], b = v[(e + 1) % 3];
                ++count[{std::min(a, b), std::max(a, b)}];
            }
        }
        size_t open = 0;
        for (const auto& [edge, c] : count)
            if (c != 2) ++open;
        return open;
    }

}// namespace


TEST_CASE("PointSurface: a sheet of points becomes a slab that hugs it") {

    const auto data = fromPoints(sheet(40, 0.05f));
    const auto mesh = splats::buildPointSurface(data, Matrix4{});

    REQUIRE_FALSE(mesh.empty());
    // Voxel 2 x spacing; the field radius is one voxel and the cut sits half
    // a voxel from the points.
    CHECK(mesh.stats.voxelSize == Approx(0.1f).margin(1e-4f));
    CHECK(mesh.stats.truncation == Approx(0.1f).margin(1e-4f));
    CHECK(mesh.stats.poses == 0);
    CHECK(mesh.stats.observedVoxels > 300);
    CHECK(mesh.triangleCount() > 1000);

    bool above = false, below = false;
    for (size_t i = 0; i < mesh.positions.size(); i += 3) {
        const float y = mesh.positions[i + 1];
        CHECK(std::abs(y) <= 0.1f + 1e-4f);
        if (y > 0.02f) above = true;
        if (y < -0.02f) below = true;
    }
    CHECK(above);
    CHECK(below);
    // Covers the sheet with the offset rim and no more.
    CHECK(mesh.stats.aabbMin.x == Approx(-0.05f).margin(0.06f));
    CHECK(mesh.stats.aabbMax.x == Approx(1.95f + 0.05f).margin(0.06f));
    CHECK(openEdges(mesh) == 0);
    CHECK(mesh.stats.components == 1);
}

TEST_CASE("PointSurface: a shell of points reconstructs at its radius, closed") {

    const auto data = fromPoints(shell(6000, 1.f));
    const auto mesh = splats::buildPointSurface(data, Matrix4{});

    REQUIRE_FALSE(mesh.empty());
    float lo = 1e9f, hi = 0.f;
    for (size_t i = 0; i < mesh.positions.size(); i += 3) {
        const float r = std::sqrt(mesh.positions[i] * mesh.positions[i] +
                                  mesh.positions[i + 1] * mesh.positions[i + 1] +
                                  mesh.positions[i + 2] * mesh.positions[i + 2]);
        lo = std::min(lo, r);
        hi = std::max(hi, r);
    }
    INFO("radius range " << lo << " .. " << hi << " voxel " << mesh.stats.voxelSize);
    // Inner and outer skins, each within one field radius of the samples.
    CHECK(lo > 1.f - 1.5f * mesh.stats.truncation);
    CHECK(hi < 1.f + 1.5f * mesh.stats.truncation);
    CHECK(hi - lo > 0.5f * mesh.stats.truncation);
    CHECK(openEdges(mesh) == 0);
}

TEST_CASE("PointSurface: the same points give the same mesh, bit for bit") {

    const auto data = fromPoints(shell(3000, 0.7f));
    const auto a = splats::buildPointSurface(data, Matrix4{});
    const auto b = splats::buildPointSurface(data, Matrix4{});

    REQUIRE_FALSE(a.empty());
    CHECK(a.positions == b.positions);
    CHECK(a.indices == b.indices);
}

TEST_CASE("PointSurface: the transform is applied before anything else") {

    const auto data = fromPoints(sheet(20, 0.05f));
    Matrix4 m;
    m.makeTranslation(10.f, -2.f, 3.f);

    const auto at = splats::buildPointSurface(data, m);
    const auto origin = splats::buildPointSurface(data, Matrix4{});
    REQUIRE_FALSE(at.empty());
    CHECK(at.stats.aabbMin.x == Approx(origin.stats.aabbMin.x + 10.f).margin(1e-4f));
    CHECK(at.stats.aabbMin.y == Approx(origin.stats.aabbMin.y - 2.f).margin(1e-4f));
    CHECK(at.stats.aabbMax.z == Approx(origin.stats.aabbMax.z + 3.f).margin(1e-4f));
    CHECK(at.triangleCount() == origin.triangleCount());

    // A scale changes the spacing, and the voxel follows it.
    Matrix4 s;
    s.makeScale(2.f, 2.f, 2.f);
    const auto scaled = splats::buildPointSurface(data, s);
    CHECK(scaled.stats.voxelSize == Approx(origin.stats.voxelSize * 2.f).margin(1e-4f));
}

TEST_CASE("PointSurface: a stray clump is an island and is dropped") {

    auto pts = sheet(30, 0.05f);
    // Five returns three metres away: a bird, a reflection, nothing to stand on.
    for (int i = 0; i < 5; ++i)
        pts.emplace_back(3.f + 0.02f * static_cast<float>(i), 2.f, 3.f);
    const auto data = fromPoints(pts);

    splats::PointSurfaceOptions keep;
    keep.minComponentVoxels = 0;
    const auto all = splats::buildPointSurface(data, Matrix4{}, keep);
    REQUIRE_FALSE(all.empty());
    CHECK(all.stats.components == 2);
    CHECK(all.stats.aabbMax.x > 2.5f);

    const auto filtered = splats::buildPointSurface(data, Matrix4{});
    REQUIRE_FALSE(filtered.empty());
    CHECK(filtered.stats.culledComponents == 1);
    CHECK(filtered.stats.culledTriangles > 0);
    CHECK(filtered.stats.aabbMax.x < 1.7f);
    CHECK(filtered.triangleCount() + filtered.stats.culledTriangles == all.triangleCount());
}

TEST_CASE("PointSurface: options - voxel, isolevel, opacity floor, the voxel ceiling") {

    const auto data = fromPoints(sheet(20, 0.05f));

    SECTION("an explicit voxel is used as given") {

        splats::PointSurfaceOptions o;
        o.voxelSize = 0.2f;
        const auto mesh = splats::buildPointSurface(data, Matrix4{}, o);
        CHECK(mesh.stats.voxelSize == 0.2f);
        CHECK(mesh.stats.truncation == Approx(0.2f));
    }

    SECTION("a higher isolevel hugs the points tighter") {

        splats::PointSurfaceOptions tight;
        tight.isolevel = 0.8f;
        const auto loose = splats::buildPointSurface(data, Matrix4{});
        const auto snug = splats::buildPointSurface(data, Matrix4{}, tight);
        REQUIRE_FALSE(snug.empty());
        CHECK(snug.stats.aabbMax.y < loose.stats.aabbMax.y);
        CHECK(snug.stats.aabbMax.y == Approx(0.02f).margin(0.005f));
    }

    SECTION("the opacity floor removes points") {

        // The first half of the sheet goes faint: a contiguous half, so the
        // spacing of what remains (and the voxel derived from it) stays put
        // and the occupied-voxel count halves.
        auto faint = data;
        for (size_t i = 0; i < faint.count() / 2; ++i) faint.opacities[i] = 0.1f;
        splats::PointSurfaceOptions o;
        o.opacityFloor = 0.5f;
        const auto half = splats::buildPointSurface(faint, Matrix4{}, o);
        const auto full = splats::buildPointSurface(faint, Matrix4{});
        REQUIRE_FALSE(half.empty());
        CHECK(half.stats.voxelSize == Approx(full.stats.voxelSize).margin(1e-5f));
        CHECK(half.stats.observedVoxels * 2 <= full.stats.observedVoxels + 2);
        CHECK(half.stats.observedVoxels < full.stats.observedVoxels);

        o.opacityFloor = 2.f;// nothing passes
        CHECK(splats::buildPointSurface(faint, Matrix4{}, o).empty());
    }

    SECTION("the voxel ceiling refuses rather than allocating") {

        splats::PointSurfaceOptions o;
        o.maxVoxels = 10;
        const auto mesh = splats::buildPointSurface(data, Matrix4{}, o);
        CHECK(mesh.empty());
        CHECK(mesh.stats.refusedBlocks > 10);
    }
}

TEST_CASE("PointSurface: empty and degenerate input") {

    CHECK(splats::buildPointSurface(SplatData{}, Matrix4{}).empty());

    // One point: no spacing to measure, the 5 cm fallback voxel, a closed
    // blob — of 8 cells, so the island filter has to be off to see it.
    splats::PointSurfaceOptions keep;
    keep.minComponentVoxels = 0;
    const auto one = splats::buildPointSurface(fromPoints({Vector3{1.f, 2.f, 3.f}}), Matrix4{}, keep);
    REQUIRE_FALSE(one.empty());
    CHECK(one.stats.voxelSize == Approx(0.05f));
    CHECK(openEdges(one) == 0);
    CHECK(one.stats.aabbMin.x > 0.9f);
    CHECK(one.stats.aabbMax.x < 1.1f);

    // A non-finite point is skipped, not propagated.
    auto d = fromPoints(sheet(10, 0.05f));
    d.means[3].set(std::nanf(""), 0.f, 0.f);
    const auto mesh = splats::buildPointSurface(d, Matrix4{});
    REQUIRE_FALSE(mesh.empty());
    for (float v : mesh.positions) CHECK(std::isfinite(v));
}
