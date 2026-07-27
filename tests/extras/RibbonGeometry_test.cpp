#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RibbonGeometry.hpp"

#include <cmath>

using namespace threepp;

namespace {

    // The two vertices of ring `i`, in the order the builder emits them.
    std::pair<Vector3, Vector3> ring(const BufferGeometry& geometry, unsigned int i) {

        const auto* position = geometry.getAttribute<float>("position");
        const auto a = static_cast<int>(i) * 2;
        return {Vector3(position->getX(a), position->getY(a), position->getZ(a)),
                Vector3(position->getX(a + 1), position->getY(a + 1), position->getZ(a + 1))};
    }

}// namespace


TEST_CASE("a ribbon is two vertices per sample, `width` apart", "[extras]") {

    CatmullRomCurve3 curve({Vector3(0, 0, 0), Vector3(0, 0, 4), Vector3(0, 0, 8)});

    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(3.f, 16, 4.f));
    REQUIRE(geometry != nullptr);

    const auto* position = geometry->getAttribute<float>("position");
    REQUIRE(position != nullptr);
    // divisions + 1 cross-sections, two vertices each.
    CHECK(position->count() == (16 + 1) * 2);
    // Two triangles per span.
    REQUIRE(geometry->getIndex() != nullptr);
    CHECK(geometry->getIndex()->count() == 16 * 6);

    for (unsigned int i = 0; i <= 16; ++i) {
        const auto [left, right] = ring(*geometry, i);
        CHECK(std::abs(left.distanceTo(right) - 3.f) < 1e-4f);
    }

    // Flat curve, flat ribbon: every normal is +Y, which is also what the
    // winding has to agree with or the surface is invisible from above.
    const auto* normal = geometry->getAttribute<float>("normal");
    REQUIRE(normal != nullptr);
    for (int i = 0; i < normal->count(); ++i) {
        CHECK(std::abs(normal->getY(i) - 1.f) < 1e-3f);
    }

    // v runs across, u runs arc length / uvLength: 8 metres at 4 m per tile.
    const auto* uv = geometry->getAttribute<float>("uv");
    REQUIRE(uv != nullptr);
    CHECK(std::abs(uv->getX(0)) < 1e-4f);
    CHECK(std::abs(uv->getY(0)) < 1e-4f);
    CHECK(std::abs(uv->getY(1) - 1.f) < 1e-4f);
    CHECK(std::abs(uv->getX(uv->count() - 1) - 2.f) < 1e-2f);
}

TEST_CASE("a ribbon winds so its faces point along the normal", "[extras]") {

    CatmullRomCurve3 curve({Vector3(0, 0, 0), Vector3(0, 0, 2), Vector3(0, 0, 4)});
    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(2.f, 4, 1.f));

    const auto* position = geometry->getAttribute<float>("position");
    const auto& index = geometry->getIndex()->array();
    REQUIRE(index.size() >= 3);

    // (b - a) x (c - a) is the face normal for counter-clockwise front faces.
    // Reversing the winding leaves a road back-face culled from above.
    for (std::size_t t = 0; t + 2 < index.size(); t += 3) {
        Vector3 a(position->getX(index[t]), position->getY(index[t]), position->getZ(index[t]));
        Vector3 b(position->getX(index[t + 1]), position->getY(index[t + 1]), position->getZ(index[t + 1]));
        Vector3 c(position->getX(index[t + 2]), position->getY(index[t + 2]), position->getZ(index[t + 2]));
        b.sub(a);
        c.sub(a);
        CHECK(b.cross(c).y > 0.f);
    }
}

TEST_CASE("a closed ribbon welds its loop", "[extras]") {

    CatmullRomCurve3 curve({Vector3(-4, 0, -4), Vector3(4, 0, -4),
                            Vector3(4, 0, 4), Vector3(-4, 0, 4)},
                           /*closed*/ true);

    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(2.f, 32, 4.f, /*closed*/ true));

    // The seam ring is a DUPLICATE, not a shared index: it has to carry
    // u = totalLength/uvLength where the first ring carries u = 0. What makes
    // it a weld is that the positions coincide exactly.
    const auto [firstLeft, firstRight] = ring(*geometry, 0);
    const auto [lastLeft, lastRight] = ring(*geometry, 32);
    CHECK(firstLeft.distanceTo(lastLeft) < 1e-4f);
    CHECK(firstRight.distanceTo(lastRight) < 1e-4f);

    const auto* uv = geometry->getAttribute<float>("uv");
    CHECK(uv->getX(0) < uv->getX(uv->count() - 1));
    CHECK(uv->getX(uv->count() - 1) > 1.f);
}

TEST_CASE("a vertical ribbon keeps the side vector it had", "[extras]") {

    // Straight up: cross(up, tangent) is degenerate the whole way, so every
    // sample has to fall back on the previous side vector rather than produce
    // a zero-length cross section.
    CatmullRomCurve3 curve({Vector3(0, 0, 0), Vector3(0, 3, 0), Vector3(0, 6, 0)});

    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(2.f, 8, 1.f));

    const auto* position = geometry->getAttribute<float>("position");
    for (int i = 0; i < position->count(); ++i) {
        CHECK(std::isfinite(position->getX(i)));
        CHECK(std::isfinite(position->getY(i)));
        CHECK(std::isfinite(position->getZ(i)));
    }
    for (unsigned int i = 0; i <= 8; ++i) {
        const auto [left, right] = ring(*geometry, i);
        CHECK(std::abs(left.distanceTo(right) - 2.f) < 1e-4f);
    }
}

TEST_CASE("a climbing ribbon stays level side to side", "[extras]") {

    // A curve that rises as it runs: the two vertices of every cross-section
    // must sit at the SAME height, which is the whole difference between a
    // road and a tube's Frenet frame.
    CatmullRomCurve3 curve({Vector3(0, 0, 0), Vector3(0, 2, 4), Vector3(0, 4, 8)});

    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(5.f, 12, 1.f));

    for (unsigned int i = 0; i <= 12; ++i) {
        const auto [left, right] = ring(*geometry, i);
        CHECK(std::abs(left.y - right.y) < 1e-4f);
        CHECK(std::abs(left.distanceTo(right) - 5.f) < 1e-4f);
    }
}
