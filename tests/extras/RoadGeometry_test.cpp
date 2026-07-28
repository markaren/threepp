#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RoadGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kStationAngle = 0.0174533f;// one degree, RoadGeometry's default

    // One cross-section of the surface, read back the way it was laid down:
    // station i owns vertices 2i and 2i + 1.
    struct Ring {
        Vector3 left;
        Vector3 right;

        [[nodiscard]] Vector3 centre() const {
            Vector3 out;
            out.copy(left).add(right).multiplyScalar(0.5f);
            return out;
        }
        [[nodiscard]] float width() const { return left.distanceTo(right); }
    };

    std::vector<Ring> ringsOf(const BufferGeometry& geometry) {

        std::vector<Ring> out;
        const auto* position = geometry.getAttribute<float>("position");
        if (!position) return out;
        for (int i = 0; i + 1 < position->count(); i += 2) {
            out.push_back({{position->getX(i), position->getY(i), position->getZ(i)},
                           {position->getX(i + 1), position->getY(i + 1), position->getZ(i + 1)}});
        }
        return out;
    }

    float angleBetween(const Vector3& a, const Vector3& b) {

        return std::acos(std::clamp(a.dot(b), -1.f, 1.f));
    }

    // Do the two cross-sections cross each other, seen from above? That is the
    // pinch the old offset roads answered with a fan, and the minimum-radius
    // floor is supposed to make it impossible rather than survivable.
    bool crossesXZ(const Ring& a, const Ring& b) {

        const auto side = [](const Vector3& p, const Vector3& q, const Vector3& r) {
            return (q.x - p.x) * (r.z - p.z) - (q.z - p.z) * (r.x - p.x);
        };
        const float d0 = side(a.left, a.right, b.left);
        const float d1 = side(a.left, a.right, b.right);
        const float d2 = side(b.left, b.right, a.left);
        const float d3 = side(b.left, b.right, a.right);
        return ((d0 > 0.f) != (d1 > 0.f)) && ((d2 > 0.f) != (d3 > 0.f));
    }

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
        REQUIRE(uv != nullptr);
        for (int i = 2; i < uv->count(); i += 2) {
            CHECK(uv->getX(i) >= uv->getX(i - 2) - 1e-5f);
        }
        // v says which edge a vertex is on, and nothing else does.
        for (int i = 0; i + 1 < uv->count(); i += 2) {
            CHECK(uv->getY(i) == 0.f);
            CHECK(uv->getY(i + 1) == 1.f);
        }
    }

    // The invariant the whole rebuild exists for: exactly `width` across at
    // EVERY station, level side to side, with no two cross-sections crossing.
    // Note the direction — the old road was allowed to narrow through a bend
    // and this one is not.
    void checkFullWidth(const BufferGeometry& geometry, float width) {

        const auto rings = ringsOf(geometry);
        REQUIRE(rings.size() > 4);
        for (std::size_t i = 0; i < rings.size(); ++i) {
            CHECK(std::abs(rings[i].width() - width) < 1e-3f);
            CHECK(std::abs(rings[i].left.y - rings[i].right.y) < 1e-4f);
            if (i > 0) CHECK_FALSE(crossesXZ(rings[i - 1], rings[i]));
        }
    }

    // Angle between consecutive cross-sections, in plan and in pitch. The
    // stations are chosen to hold both under the threshold; measuring them is
    // how "smooth" stops being a claim.
    void checkStationBreaks(const BufferGeometry& geometry, float limit) {

        const auto rings = ringsOf(geometry);
        REQUIRE(rings.size() > 4);
        std::vector<Vector3> steps;
        for (std::size_t i = 0; i + 1 < rings.size(); ++i) {
            Vector3 step;
            step.copy(rings[i + 1].centre()).sub(rings[i].centre());
            if (step.length() < 1e-6f) continue;
            steps.push_back(step.normalize());
        }
        REQUIRE(steps.size() > 3);
        for (std::size_t i = 1; i < steps.size(); ++i) {
            const Vector3 planA(steps[i - 1].x, 0.f, steps[i - 1].z);
            const Vector3 planB(steps[i].x, 0.f, steps[i].z);
            if (planA.length() > 1e-5f && planB.length() > 1e-5f) {
                Vector3 a(planA), b(planB);
                a.normalize();
                b.normalize();
                CHECK(angleBetween(a, b) < limit);
            }
            const float pitchA = std::asin(std::clamp(steps[i - 1].y, -1.f, 1.f));
            const float pitchB = std::asin(std::clamp(steps[i].y, -1.f, 1.f));
            CHECK(std::abs(pitchB - pitchA) < limit);
        }
    }

    CatmullRomCurve3 defaultSpline() {

        return CatmullRomCurve3({Vector3(-3, 0.5f, 1.5f), Vector3(-1, 0.5f, -1),
                                 Vector3(1, 0.5f, -1), Vector3(3, 0.5f, 1.5f)});
    }

    CatmullRomCurve3 cornerSpline() {

        return CatmullRomCurve3({Vector3(-10, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 10)});
    }

    // A climb into a crest and down the other side, with the crest INSIDE a
    // bend: the case the user rejected every previous road for, because a
    // piecewise-linear profile leaves a crease exactly there.
    CatmullRomCurve3 crestInTurnSpline() {

        return CatmullRomCurve3({Vector3(-12, 0, -6), Vector3(-4, 1.5f, -2),
                                 Vector3(0, 3, 2), Vector3(4, 1.5f, 6), Vector3(12, 0, 8)});
    }

}// namespace


TEST_CASE("a road's alignment is tangent-continuous by construction", "[extras]") {

    // Not "within a fitting tolerance": a biarc MEETS the tangents it was built
    // from, and its two arcs meet each other, so a joint has no angle at all.
    // Anything above rounding here is a bug in the construction.
    for (const bool wide : {false, true}) {
        auto curve = defaultSpline();
        auto geometry = RoadGeometry::create(curve, wide ? 4.f : 2.f, 24, 4.f);
        const auto& alignment = geometry->alignment();
        REQUIRE(alignment.plan().pieces().size() > 4);
        CHECK(alignment.plan().maxAngleBreak(false) < 1e-3f);
        CHECK(alignment.profile().maxAngleBreak(false) < 1e-3f);
    }

    auto graded = crestInTurnSpline();
    auto geometry = RoadGeometry::create(graded, 5.f, 24, 4.f);
    CHECK(geometry->alignment().plan().maxAngleBreak(false) < 1e-3f);
    CHECK(geometry->alignment().profile().maxAngleBreak(false) < 1e-3f);
}

TEST_CASE("no bend in a road is tighter than its own half-width", "[extras]") {

    // The floor is what removes the apex, the pinch and the fan: an inner edge
    // that reaches the centre of curvature is all three at once. The editor's
    // DEFAULT spline is the case that matters — its tightest curvature radius
    // is about 1.35 m and the default road is 4 m wide, so it cannot be built
    // at all without bending a little.
    auto curve = defaultSpline();
    auto geometry = RoadGeometry::create(curve, 4.f, 24, 4.f);
    const auto& report = geometry->alignment().report();

    CHECK(report.planMinRadius >= 2.1f - 1e-3f);
    CHECK(report.bendsRelaxed > 0);
    CHECK(report.seedsRelaxed > 0);
    // ...and what that cost is bounded: the curve the user drew still runs on
    // the road, which is the only promise a relaxed alignment makes.
    CHECK(report.deviation < 2.f);

    // A bend far tighter than the road is wide: same story, bigger floor.
    auto corner = cornerSpline();
    auto wide = RoadGeometry::create(corner, 6.f, 24, 4.f);
    CHECK(wide->alignment().report().planMinRadius >= 3.15f - 1e-3f);
    CHECK(wide->alignment().report().deviation < 3.f);

    checkFullWidth(*geometry, 4.f);
    checkFullWidth(*wide, 6.f);
    checkNoDownwardFace(*geometry);
    checkNoDownwardFace(*wide);
}

TEST_CASE("a road follows the curve it was fitted to, within tolerance", "[extras]") {

    // A road wide enough that no bend hits the floor follows the drawing, and
    // the seeding refines until it does: `fit` is measured, not assumed.
    auto curve = defaultSpline();
    auto geometry = RoadGeometry::create(curve, 1.f, 8, 4.f);
    const auto& report = geometry->alignment().report();

    CHECK(report.planMinRadius > 0.5f);
    CHECK(report.bendsRelaxed == 0);
    CHECK(report.seedsRelaxed == 0);
    CHECK(report.fit <= 0.02f);
    // With nothing relaxed, the 3D deviation is the fit plus what the profile
    // chain does with the elevation — a flat spline, so nothing.
    CHECK(report.deviation <= 0.03f);
    CHECK(report.seeds >= 8);
}

TEST_CASE("a road is full width at every station", "[extras]") {

    // The reversal of the old expectation. The trimmed road was allowed to
    // narrow through a bend because its inner offset had crossed itself; this
    // one has no bend tight enough for that to happen, so a cross-section that
    // is not full width is a defect rather than a case.
    for (const float width : {1.f, 4.f, 6.f}) {
        auto curve = defaultSpline();
        auto geometry = RoadGeometry::create(curve, width, 24, 4.f);
        checkFullWidth(*geometry, width);
        checkStationBreaks(*geometry, kStationAngle * 2.f);
        checkNoDownwardFace(*geometry);
        checkMonotoneU(*geometry);
    }

    auto corner = cornerSpline();
    auto sharp = RoadGeometry::create(corner, 6.f, 24, 4.f);
    checkFullWidth(*sharp, 6.f);
    checkStationBreaks(*sharp, kStationAngle * 2.f);
}

TEST_CASE("a graded road has no crease where its grade changes", "[extras]") {

    // Elevation is a G1 chain of its own, so a grade change is a vertical curve
    // and not the crease a piecewise-linear profile leaves at every join. The
    // crest here is inside a bend, which is where the previous roads invented a
    // step: they averaged the two crossing segments' heights at a trim corner.
    auto curve = crestInTurnSpline();
    auto geometry = RoadGeometry::create(curve, 5.f, 24, 4.f);
    const auto& report = geometry->alignment().report();

    // Untouched by the vertical floor: what the user drew has a gentler crest
    // than the floor bounds, so the road climbs to the heights it was given.
    CHECK(report.profileMinRadius >= 4.f - 1e-2f);
    CHECK(report.planMinRadius >= 2.625f - 1e-3f);

    checkFullWidth(*geometry, 5.f);
    checkStationBreaks(*geometry, kStationAngle * 2.f);
    checkNoDownwardFace(*geometry);

    // The road really does climb and really does come back down, and both edges
    // are at the same height the whole way — a road banks nowhere.
    const auto rings = ringsOf(*geometry);
    float highest = -1e9f;
    for (const auto& ring : rings) highest = std::max(highest, ring.centre().y);
    CHECK(highest > 1.f);
    CHECK(rings.back().centre().y < highest - 0.5f);

    const auto* normal = geometry->getAttribute<float>("normal");
    REQUIRE(normal != nullptr);
    bool tilted = false;
    for (int i = 0; i < normal->count(); ++i) {
        CHECK(normal->getY(i) > 0.f);
        if (normal->getY(i) < 0.99f) tilted = true;
    }
    CHECK(tilted);
}

TEST_CASE("a climbing road rises with the grade", "[extras]") {

    CatmullRomCurve3 curve({Vector3(0, 0, 0), Vector3(0, 2, 4), Vector3(0, 4, 8)});
    auto geometry = RoadGeometry::create(curve, 5.f, 48, 1.f);

    checkFullWidth(*geometry, 5.f);
    checkNoDownwardFace(*geometry);

    const auto rings = ringsOf(*geometry);
    CHECK(rings.front().centre().y < rings.back().centre().y - 3.f);
}

TEST_CASE("a closed road welds its loop", "[extras]") {

    CatmullRomCurve3 curve({Vector3(-4, 0, -4), Vector3(4, 0, -4),
                            Vector3(4, 0, 4), Vector3(-4, 0, 4)},
                           /*closed*/ true);
    auto geometry = RoadGeometry::create(curve, 2.f, 48, 4.f, /*closed*/ true);

    const auto rings = ringsOf(*geometry);
    REQUIRE(rings.size() > 8);
    // The seam ring is a DUPLICATE, not a shared index: it carries
    // u = totalLength/uvLength where the first carries u = 0. What makes it a
    // weld is that the positions coincide.
    CHECK(rings.front().left.distanceTo(rings.back().left) < 1e-3f);
    CHECK(rings.front().right.distanceTo(rings.back().right) < 1e-3f);

    const auto* uv = geometry->getAttribute<float>("uv");
    CHECK(uv->getX(0) == 0.f);
    CHECK(uv->getX(uv->count() - 1) > 1.f);

    CHECK(geometry->alignment().plan().maxAngleBreak(true) < 1e-3f);
    checkFullWidth(*geometry, 2.f);
    checkNoDownwardFace(*geometry);
    checkMonotoneU(*geometry);
}

TEST_CASE("the collider is one convex hull per station interval", "[extras]") {

    auto curve = defaultSpline();
    auto geometry = RoadGeometry::create(curve, 4.f, 24, 4.f);
    const auto rings = ringsOf(*geometry);
    const auto hulls = RoadGeometry::hulls(*geometry, 0.25f);

    REQUIRE(hulls.size() == rings.size() - 1);
    REQUIRE(!hulls.empty());

    for (const auto& hull : hulls) {
        Vector3 centroid;
        for (const auto& point : hull) centroid.add(point);
        centroid.multiplyScalar(1.f / 8.f);

        for (int corner = 0; corner < 4; ++corner) {
            // The underside is the top pushed down the cross-section's own
            // normal: a real thickness, and downward.
            Vector3 drop;
            drop.copy(hull[corner]).sub(hull[corner + 4]);
            CHECK(std::abs(drop.length() - 0.25f) < 1e-4f);
            CHECK(drop.y > 0.f);
        }

        // Convex, in the sense that matters to a cooker: every one of the eight
        // points is a VERTEX of the hull rather than swallowed inside it. A
        // point that is not extreme in the direction it lies from the centroid
        // is inside something.
        for (const auto& point : hull) {
            Vector3 direction;
            direction.copy(point).sub(centroid);
            REQUIRE(direction.length() > 1e-5f);
            direction.normalize();
            const float reach = point.dot(direction);
            for (const auto& other : hull) {
                if (&other == &point) continue;
                CHECK(other.dot(direction) <= reach + 1e-5f);
            }
        }
    }

    // Consecutive hulls share a whole joint cross-section, top and bottom, so
    // there is no seam between them for a body to catch on.
    for (std::size_t i = 0; i + 1 < hulls.size(); ++i) {
        for (int corner = 0; corner < 4; ++corner) {
            const int mine = corner < 2 ? corner + 2 : corner + 4;
            const int theirs = corner < 2 ? corner : corner + 2;
            CHECK(hulls[i][static_cast<std::size_t>(mine)].distanceTo(
                          hulls[i + 1][static_cast<std::size_t>(theirs)]) < 1e-5f);
        }
    }
}

TEST_CASE("a road's collider is bounded by its stations, not by its samples", "[extras]") {

    // `divisions` seeds the alignment; it does not tessellate the road. Asking
    // for eight times the samples must not cook eight times the shapes — that
    // is the difference between a collider that follows the road's SHAPE and
    // one that follows how finely somebody sampled it.
    auto curve = defaultSpline();
    auto coarse = RoadGeometry::create(curve, 4.f, 12, 4.f);
    auto fine = RoadGeometry::create(curve, 4.f, 96, 4.f);

    const auto few = RoadGeometry::hulls(*coarse, 0.25f).size();
    const auto many = RoadGeometry::hulls(*fine, 0.25f).size();
    REQUIRE(few > 8);
    // Eight times the samples, well under twice the shapes — and the residual
    // difference is the alignments themselves differing a little, not the
    // tessellation following the sampling.
    CHECK(static_cast<float>(many) < static_cast<float>(few) * 2.f);
    CHECK(static_cast<float>(few) < static_cast<float>(many) * 2.f);

    // And the shapes are bounded outright: a degree per station over a road
    // that turns less than four right angles cannot need many hundreds.
    CHECK(many < 500);
}

TEST_CASE("the same road builds the same bytes", "[extras]") {

    // The document carries the generated mesh, so a rebuild that wandered by an
    // ulp would make every save after a reload a diff.
    auto curve = cornerSpline();
    auto a = RoadGeometry::create(curve, 6.f, 48, 4.f);
    auto b = RoadGeometry::create(curve, 6.f, 48, 4.f);

    const auto& positionsA = a->getAttribute<float>("position")->array();
    const auto& positionsB = b->getAttribute<float>("position")->array();
    REQUIRE(positionsA.size() == positionsB.size());
    bool identical = true;
    for (std::size_t i = 0; i < positionsA.size(); ++i) {
        if (positionsA[i] != positionsB[i]) identical = false;
    }
    CHECK(identical);
    CHECK(a->getIndex()->array() == b->getIndex()->array());
}

TEST_CASE("a curve with no length builds no road", "[extras]") {

    CatmullRomCurve3 curve({Vector3(1, 1, 1), Vector3(1, 1, 1), Vector3(1, 1, 1)});
    auto geometry = RoadGeometry::create(curve, 4.f, 24, 4.f);
    CHECK(geometry->getAttribute<float>("position") == nullptr);
    CHECK(RoadGeometry::hulls(*geometry, 0.25f).empty());
}
