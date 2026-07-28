#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RoadGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace threepp;

namespace {

    // The two vertices of cross-section `i`, in the order the builder emits
    // them: left of travel first, right second.
    std::pair<Vector3, Vector3> ring(const BufferGeometry& geometry, int i) {

        const auto* position = geometry.getAttribute<float>("position");
        const int a = i * 2;
        return {Vector3(position->getX(a), position->getY(a), position->getZ(a)),
                Vector3(position->getX(a + 1), position->getY(a + 1), position->getZ(a + 1))};
    }

    int ringCount(const BufferGeometry& geometry) {

        return geometry.getAttribute<float>("position")->count() / 2;
    }

    // Are cross-sections i and i + 1 the pair a JOINT carries? They sit on the
    // same centreline point — same arc length, hence same u — each in its own
    // piece's frame, so they meet in the middle on purpose: the quad between
    // them is the wedge that covers the kink. Every other neighbouring pair has
    // to stay clear of its neighbour.
    bool atJoint(const BufferGeometry& geometry, int i) {

        const auto* uv = geometry.getAttribute<float>("uv");
        return std::abs(uv->getX(i * 2) - uv->getX(i * 2 + 2)) < 1e-6f;
    }

    // Do the segments p0-p1 and q0-q1 cross, in the XZ plane? A cross-section
    // that cuts through its neighbour is the road turning inside out. Two that
    // MEET at a shared vertex are not that: a pie sector's cross-sections all
    // reach the same point at the arc's centre, on purpose.
    bool crossesXZ(const Vector3& p0, const Vector3& p1, const Vector3& q0, const Vector3& q1) {

        if (p0.distanceTo(q0) < 1e-4f || p1.distanceTo(q1) < 1e-4f) return false;
        if (p0.distanceTo(q1) < 1e-4f || p1.distanceTo(q0) < 1e-4f) return false;

        const auto side = [](const Vector3& a, const Vector3& b, const Vector3& c) {
            return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
        };
        const float d0 = side(p0, p1, q0);
        const float d1 = side(p0, p1, q1);
        const float d2 = side(q0, q1, p0);
        const float d3 = side(q0, q1, p1);
        return ((d0 > 0) != (d1 > 0)) && ((d2 > 0) != (d3 > 0));
    }

    // Is `point` on the road, seen from above? The road is a union of pieces,
    // so covering the full width is a property of the WHOLE mesh rather than of
    // any one cross-section — which is exactly what has to be tested, since the
    // inside of a tight corner is covered by the pieces either side of it.
    bool coveredXZ(const BufferGeometry& geometry, const Vector3& point) {

        const auto* position = geometry.getAttribute<float>("position");
        const auto& index = geometry.getIndex()->array();
        const auto side = [](float ax, float az, float bx, float bz, float cx, float cz) {
            return (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
        };
        for (std::size_t t = 0; t + 2 < index.size(); t += 3) {
            const auto a = static_cast<int>(index[t]);
            const auto b = static_cast<int>(index[t + 1]);
            const auto c = static_cast<int>(index[t + 2]);
            const float d0 = side(position->getX(a), position->getZ(a),
                                  position->getX(b), position->getZ(b), point.x, point.z);
            const float d1 = side(position->getX(b), position->getZ(b),
                                  position->getX(c), position->getZ(c), point.x, point.z);
            const float d2 = side(position->getX(c), position->getZ(c),
                                  position->getX(a), position->getZ(a), point.x, point.z);
            if ((d0 >= -1e-5f && d1 >= -1e-5f && d2 >= -1e-5f) ||
                (d0 <= 1e-5f && d1 <= 1e-5f && d2 <= 1e-5f)) {
                return true;
            }
        }
        return false;
    }

    // No triangle may face down: a degenerate one (zero area) is fine — the
    // joints and the pie sectors make those — but a face the winding has turned
    // over is the fold this construction exists to make impossible.
    void checkNoDownwardFace(const BufferGeometry& geometry) {

        const auto* position = geometry.getAttribute<float>("position");
        const auto& index = geometry.getIndex()->array();
        for (std::size_t t = 0; t + 2 < index.size(); t += 3) {
            const Vector3 a(position->getX(index[t]), position->getY(index[t]), position->getZ(index[t]));
            Vector3 b(position->getX(index[t + 1]), position->getY(index[t + 1]), position->getZ(index[t + 1]));
            Vector3 c(position->getX(index[t + 2]), position->getY(index[t + 2]), position->getZ(index[t + 2]));
            b.sub(a);
            c.sub(a);
            CHECK(b.cross(c).y > -1e-6f);
        }
    }

    void checkMonotoneU(const BufferGeometry& geometry) {

        const auto* uv = geometry.getAttribute<float>("uv");
        for (int i = 2; i < uv->count(); i += 2) {
            CHECK(uv->getX(i) >= uv->getX(i - 2) - 1e-5f);
        }
        for (int i = 0; i + 1 < uv->count(); i += 2) {
            CHECK(std::abs(uv->getY(i)) < 1e-5f);
            CHECK(std::abs(uv->getY(i + 1) - 1.f) < 1e-5f);
        }
    }

    // The editor's default spline, and the L corner that bends tighter than a
    // six metre road's half-width.
    CatmullRomCurve3 defaultSpline() {

        return CatmullRomCurve3({Vector3(-3, 0.5f, 1.5f), Vector3(-1, 0.5f, -1),
                                 Vector3(1, 0.5f, -1), Vector3(3, 0.5f, 1.5f)});
    }

    CatmullRomCurve3 cornerSpline() {

        return CatmullRomCurve3({Vector3(-10, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 10)});
    }

}// namespace


TEST_CASE("a road keeps the authored width through its bends", "[extras]") {

    // The editor's default spline at the default width: its tightest bend has a
    // curvature radius just over the half-width, so every offset fits and the
    // road is FULL WIDTH from end to end. The ribbon this replaced narrowed
    // through those bends — its inner offset was capped at the bend radius —
    // and this is the reversal of that expectation.
    auto curve = defaultSpline();
    const auto path = RoadPath::fromCurve(curve, 72, false);
    auto geometry = RoadGeometry::create(path, 4.f, 4.f);
    REQUIRE(geometry != nullptr);

    const int rings = ringCount(*geometry);
    REQUIRE(rings > 2);

    float narrowest = std::numeric_limits<float>::max();
    float widest = 0.f;
    for (int i = 0; i < rings; ++i) {
        const auto [left, right] = ring(*geometry, i);
        const float across = left.distanceTo(right);
        narrowest = std::min(narrowest, across);
        widest = std::max(widest, across);
        CHECK(std::abs(left.y - right.y) < 1e-4f);// level side to side
        if (i + 1 < rings && !atJoint(*geometry, i)) {
            const auto [nextLeft, nextRight] = ring(*geometry, i + 1);
            CHECK(!crossesXZ(left, right, nextLeft, nextRight));
        }
    }
    // Every cross-section, corners included: the width is the width.
    CHECK(narrowest > 4.f - 1e-3f);
    CHECK(widest < 4.f + 1e-3f);

    checkNoDownwardFace(*geometry);
    checkMonotoneU(*geometry);

    // Flat spline, flat road: every normal points straight up, which is also
    // what the winding has to agree with or the surface is invisible.
    const auto* normal = geometry->getAttribute<float>("normal");
    for (int i = 0; i < normal->count(); ++i) {
        CHECK(std::abs(normal->getY(i) - 1.f) < 1e-3f);
    }
}

TEST_CASE("a road covers the full width its config asks for", "[extras]") {

    // Not a property of any one cross-section: the road is a UNION of pieces,
    // and what the user sees is whether the band `width` wide around their
    // curve is paved. Sampled off the CURVE rather than off the primitives, so
    // it is the authored centreline being checked, not the fit's own idea of it.
    auto curve = defaultSpline();
    const auto path = RoadPath::fromCurve(curve, 72, false);
    auto geometry = RoadGeometry::create(path, 4.f, 4.f);

    const float reach = 2.f * 0.95f;
    for (int i = 3; i <= 97; ++i) {
        const float t = static_cast<float>(i) / 100.f;
        Vector3 point, tangent;
        curve.getPoint(t, point);
        curve.getTangent(t, tangent);
        Vector3 across{tangent.z, 0.f, -tangent.x};
        if (across.length() < 1e-6f) continue;
        across.normalize();

        CHECK(coveredXZ(*geometry, point));
        Vector3 left, right;
        left.copy(point).addScaledVector(across, -reach);
        right.copy(point).addScaledVector(across, reach);
        CHECK(coveredXZ(*geometry, left));
        CHECK(coveredXZ(*geometry, right));
    }
}

TEST_CASE("a bend tighter than the half-width becomes a pie sector", "[extras]") {

    // The corner that used to fold: a six metre road through a bend whose
    // radius is about two. The inner offset cannot reach past the centre of the
    // bend without running backward, so the annulus collapses to a SECTOR whose
    // inner vertices sit on that centre — full outer reach, no fold, and the
    // inside of the corner covered by the pieces either side of it.
    auto curve = cornerSpline();
    const auto path = RoadPath::fromCurve(curve, 48, false);
    auto geometry = RoadGeometry::create(path, 6.f, 4.f);

    bool tightBend = false;
    bool reachesCentre = false;
    const int rings = ringCount(*geometry);
    for (const auto& primitive : path.primitives()) {
        if (primitive.kind != RoadPrimitive::Kind::Arc || primitive.radius >= 3.f) continue;
        tightBend = true;
        for (int i = 0; i < rings; ++i) {
            const auto [left, right] = ring(*geometry, i);
            for (const auto& vertex : {left, right}) {
                const float dx = vertex.x - primitive.center.x;
                const float dz = vertex.z - primitive.center.z;
                if (std::sqrt(dx * dx + dz * dz) < 1e-3f) reachesCentre = true;
            }
        }
    }
    CHECK(tightBend);
    CHECK(reachesCentre);

    // ...and the mesh is still a surface: no fold, no cross-section cutting
    // through its neighbour, no face turned over.
    for (int i = 0; i + 1 < rings; ++i) {
        if (atJoint(*geometry, i)) continue;
        const auto [left, right] = ring(*geometry, i);
        const auto [nextLeft, nextRight] = ring(*geometry, i + 1);
        CHECK(!crossesXZ(left, right, nextLeft, nextRight));
    }
    checkNoDownwardFace(*geometry);
    checkMonotoneU(*geometry);

    // The OUTER half of the band is paved the whole way round whatever the bend
    // does — it is the inner one a tight corner cannot reach past its centre,
    // and the neighbouring pieces that cover it there.
    for (int i = 3; i <= 97; ++i) {
        const float t = static_cast<float>(i) / 100.f;
        Vector3 point, tangent;
        curve.getPoint(t, point);
        curve.getTangent(t, tangent);
        Vector3 across{tangent.z, 0.f, -tangent.x};
        if (across.length() < 1e-6f) continue;
        across.normalize();
        // The corner turns one way the whole way through, so the outer side is
        // the one away from (0, 0, 0).
        const float sign = (point.x * across.x + point.z * across.z) >= 0.f ? 1.f : -1.f;
        Vector3 outer;
        outer.copy(point).addScaledVector(across, sign * 3.f * 0.95f);
        CHECK(coveredXZ(*geometry, point));
        CHECK(coveredXZ(*geometry, outer));
    }
}

TEST_CASE("a closed road welds its loop", "[extras]") {

    CatmullRomCurve3 curve({Vector3(-4, 0, -4), Vector3(4, 0, -4),
                            Vector3(4, 0, 4), Vector3(-4, 0, 4)},
                           /*closed*/ true);
    const auto path = RoadPath::fromCurve(curve, 96, true);
    auto geometry = RoadGeometry::create(path, 2.f, 4.f);

    const int rings = ringCount(*geometry);
    const auto [firstLeft, firstRight] = ring(*geometry, 0);
    const auto [lastLeft, lastRight] = ring(*geometry, rings - 1);

    // The seam ring is a DUPLICATE, not a shared index: it carries
    // u = totalLength/uvLength where the first ring carries u = 0. What makes
    // it a weld is that the positions coincide.
    CHECK(firstLeft.distanceTo(lastLeft) < 1e-3f);
    CHECK(firstRight.distanceTo(lastRight) < 1e-3f);

    const auto* uv = geometry->getAttribute<float>("uv");
    CHECK(uv->getX(0) < uv->getX(uv->count() - 1));
    CHECK(uv->getX(uv->count() - 1) > 1.f);
    checkNoDownwardFace(*geometry);
}

TEST_CASE("a climbing road rises with the grade", "[extras]") {

    // A road that climbs as it runs: the two vertices of every cross-section
    // sit at the SAME height (a road banks nowhere), and the normals tip with
    // the slope rather than staying at +Y.
    CatmullRomCurve3 curve({Vector3(0, 0, 0), Vector3(0, 2, 4), Vector3(0, 4, 8)});
    const auto path = RoadPath::fromCurve(curve, 48, false);
    auto geometry = RoadGeometry::create(path, 5.f, 1.f);

    const int rings = ringCount(*geometry);
    for (int i = 0; i < rings; ++i) {
        const auto [left, right] = ring(*geometry, i);
        CHECK(std::abs(left.y - right.y) < 1e-4f);
        CHECK(std::abs(left.distanceTo(right) - 5.f) < 1e-2f);
    }

    const auto* normal = geometry->getAttribute<float>("normal");
    bool tilted = false;
    for (int i = 0; i < normal->count(); ++i) {
        CHECK(normal->getY(i) > 0.f);
        if (normal->getY(i) < 0.99f) tilted = true;
    }
    CHECK(tilted);
    checkNoDownwardFace(*geometry);
}
