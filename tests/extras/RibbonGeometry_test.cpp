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

    // Do the segments p0-p1 and q0-q1 cross, in the XZ plane? Ribbons here are
    // flat, and a cross-section that cuts through its neighbour is the ribbon
    // turning inside out.
    bool crossesXZ(const Vector3& p0, const Vector3& p1, const Vector3& q0, const Vector3& q1) {

        const auto side = [](const Vector3& a, const Vector3& b, const Vector3& c) {
            return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
        };
        const float d0 = side(p0, p1, q0);
        const float d1 = side(p0, p1, q1);
        const float d2 = side(q0, q1, p0);
        const float d3 = side(q0, q1, p1);
        return ((d0 > 0) != (d1 > 0)) && ((d2 > 0) != (d3 > 0));
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

TEST_CASE("a ribbon miters its corners instead of folding", "[extras]") {

    // A hard 90-degree corner, tessellated so the corner IS a sample: two
    // 6 metre spans meeting at the origin, and a ribbon wide enough that the
    // per-sample side vectors this replaced sent each edge overshooting into
    // the next span.
    CatmullRomCurve3 curve({Vector3(-6, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 6)});
    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(6.f, 2, 4.f));

    const auto [left0, right0] = ring(*geometry, 0);
    const auto [left1, right1] = ring(*geometry, 1);
    const auto [left2, right2] = ring(*geometry, 2);

    // The straight spans keep the authored width, measured across the span they
    // belong to: the miter widens the corner only.
    CHECK(std::abs(left0.distanceTo(right0) - 6.f) < 1e-4f);
    CHECK(std::abs(left2.distanceTo(right2) - 6.f) < 1e-4f);
    Vector3 across;
    across.copy(right0).sub(left0).normalize();
    CHECK(std::abs(across.dot(Vector3(1, 0, 0))) < 1e-4f);// span runs along +X
    across.copy(right2).sub(left2).normalize();
    CHECK(std::abs(across.dot(Vector3(0, 0, 1))) < 1e-4f);// span runs along +Z

    // The corner is widened by 1/cos(45 deg) = sqrt(2) — enough to reach where
    // the two spans' offset edges meet — and no more.
    const float cornerWidth = left1.distanceTo(right1);
    CHECK(std::abs(cornerWidth - 6.f * std::sqrt(2.f)) < 1e-3f);
    CHECK(cornerWidth <= 6.f * 2.f + 1e-4f);

    // Neither edge doubles back on itself. This is the fold: with independent
    // per-sample sides the corner ring's inner vertex sits short of where the
    // two offset edges cross, so the edge runs in and straight back out again
    // and the two quads overlap in the notch it leaves.
    const auto edgeTurnsBack = [](const Vector3& a, const Vector3& b, const Vector3& c) {
        Vector3 in, out;
        in.copy(b).sub(a);
        out.copy(c).sub(b);
        return in.dot(out) < -1e-4f;
    };
    CHECK(!edgeTurnsBack(left0, left1, left2));
    CHECK(!edgeTurnsBack(right0, right1, right2));

    // And consecutive cross-sections stay clear of each other, so no quad is
    // built inside out.
    CHECK(!crossesXZ(left0, right0, left1, right1));
    CHECK(!crossesXZ(left1, right1, left2, right2));
}

TEST_CASE("the miter widening is clamped at a hairpin", "[extras]") {

    // A near-reversal: 1/cos(theta/2) runs away here (over 8x), and an
    // unclamped cross-section would spike far outside the ribbon.
    CatmullRomCurve3 curve({Vector3(-4, 0, 0), Vector3(0, 0, 0), Vector3(-4, 0, 1)});
    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(2.f, 2, 1.f));

    const auto* position = geometry->getAttribute<float>("position");
    for (int i = 0; i < position->count(); ++i) {
        CHECK(std::isfinite(position->getX(i)));
        CHECK(std::isfinite(position->getZ(i)));
    }
    for (unsigned int i = 0; i <= 2; ++i) {
        const auto [left, right] = ring(*geometry, i);
        const float across = left.distanceTo(right);
        CHECK(across >= 2.f - 1e-4f);
        CHECK(across <= 2.f * 2.f + 1e-4f);
    }
}

TEST_CASE("a pass near vertical does not fan", "[extras]") {

    // Climbs steeply while its horizontal drift REVERSES (+z going up, -z
    // coming over the top). Near the apex the tangent is close to vertical, so
    // cross(up, tangent) is all noise; renormalizing that noise let the side
    // vector's azimuth spin freely ring to ring, and every quad between two
    // spun rings was built inside out — the fan of flipped spikes this pins.
    CatmullRomCurve3 curve({Vector3(0, 0, 0), Vector3(0, 6, 1), Vector3(0, 12, 0)});
    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(2.f, 24, 4.f));

    Vector3 previous, current;
    for (unsigned int i = 0; i < 24; ++i) {
        const auto [leftA, rightA] = ring(*geometry, i);
        const auto [leftB, rightB] = ring(*geometry, i + 1);
        previous.copy(rightA).sub(leftA).normalize();
        current.copy(rightB).sub(leftB).normalize();
        // Adjacent cross-sections never reverse: a left edge that swaps to the
        // right between two rings is exactly the flipped-winding artefact.
        CHECK(previous.dot(current) > 0.f);
        // And the miter clamp still bounds every ring.
        CHECK(leftB.distanceTo(rightB) <= 2.f * 2.f + 1e-4f);
    }
}

TEST_CASE("a horizontal cusp mirrors instead of fanning", "[extras]") {

    // A near-reversal sampled DENSELY: unlike the two-span hairpin above, the
    // curve's about-turn now falls across many rings, and the recomputed side
    // vector comes back mirrored on the far side. Without sign continuity the
    // rings on the two arms disagree about which edge is left, and the quads
    // spanning the cusp cross.
    CatmullRomCurve3 curve({Vector3(-6, 0, 0), Vector3(0, 0, 0), Vector3(-6, 0, 0.4f)});
    auto geometry = RibbonGeometry::create(curve, RibbonGeometry::Params(2.f, 16, 4.f));

    Vector3 previous, current;
    for (unsigned int i = 0; i < 16; ++i) {
        const auto [leftA, rightA] = ring(*geometry, i);
        const auto [leftB, rightB] = ring(*geometry, i + 1);
        previous.copy(rightA).sub(leftA).normalize();
        current.copy(rightB).sub(leftB).normalize();
        CHECK(previous.dot(current) > 0.f);
        CHECK(leftB.distanceTo(rightB) <= 2.f * 2.f + 1e-4f);
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
