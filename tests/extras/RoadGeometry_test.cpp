#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RoadGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace threepp;

namespace {

    // The road lays its whole left edge down and then its whole right edge, and
    // v says which is which. Trimming takes vertices off one edge and not the
    // other, so the two are no longer the same length — reading them back is
    // reading two polylines, not a list of cross-sections.
    std::vector<Vector3> edgeChain(const BufferGeometry& geometry, float v) {

        const auto* position = geometry.getAttribute<float>("position");
        const auto* uv = geometry.getAttribute<float>("uv");
        std::vector<Vector3> out;
        for (int i = 0; i < position->count(); ++i) {
            if (std::abs(uv->getY(i) - v) > 1e-5f) continue;
            out.emplace_back(position->getX(i), position->getY(i), position->getZ(i));
        }
        return out;
    }

    std::vector<float> edgeU(const BufferGeometry& geometry, float v) {

        const auto* uv = geometry.getAttribute<float>("uv");
        std::vector<float> out;
        for (int i = 0; i < uv->count(); ++i) {
            if (std::abs(uv->getY(i) - v) > 1e-5f) continue;
            out.push_back(uv->getX(i));
        }
        return out;
    }

    // Turn angles, in degrees, at every interior vertex of a polyline. On a
    // smooth edge these are bounded by how coarsely the curve was sampled; a
    // TRIM corner is the one place an edge really has a corner, and it stands
    // far clear of that.
    std::vector<float> turnAngles(const std::vector<Vector3>& chain) {

        std::vector<float> out;
        for (std::size_t i = 1; i + 1 < chain.size(); ++i) {
            Vector3 in, next;
            in.copy(chain[i]).sub(chain[i - 1]);
            next.copy(chain[i + 1]).sub(chain[i]);
            if (in.length() < 1e-6f || next.length() < 1e-6f) continue;
            in.normalize();
            next.normalize();
            out.push_back(std::acos(std::clamp(in.dot(next), -1.f, 1.f)) * 180.f / 3.14159265f);
        }
        return out;
    }

    // A corner sharper than any sampling artefact: the trim points, and nothing
    // else. Each test also asserts how smooth the rest of the edge is, so this
    // threshold separates rather than tolerates — the smooth edges below turn
    // under ten degrees a vertex and the trim corners over thirty.
    constexpr float kCornerDegrees = 20.f;

    int sharpCount(const std::vector<Vector3>& chain) {

        int count = 0;
        for (const float angle : turnAngles(chain)) {
            if (angle > kCornerDegrees) ++count;
        }
        return count;
    }

    float smoothest(const std::vector<Vector3>& chain) {

        float worst = 0.f;
        for (const float angle : turnAngles(chain)) {
            if (angle <= kCornerDegrees) worst = std::max(worst, angle);
        }
        return worst;
    }

    // Is `point` on the road, seen from above? Full width is a property of the
    // SURFACE, not of any one cross-section — the region around a trim is a fan
    // rather than a strip — so it is asked of the triangles.
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

        for (const float v : {0.f, 1.f}) {
            const auto u = edgeU(geometry, v);
            for (std::size_t i = 1; i < u.size(); ++i) {
                CHECK(u[i] >= u[i - 1] - 1e-5f);
            }
        }
    }

    // Every centreline sample, and both offsets of it, have to be paved — a
    // road that narrowed through a bend is one that stopped covering its own
    // offsets. `reach` is a shade under the half-width so a point does not fail
    // for landing exactly on the boundary.
    void checkCovers(const BufferGeometry& geometry, const Curve3& curve, float half,
                     bool innerToo) {

        const float reach = half * 0.95f;
        for (int i = 3; i <= 97; ++i) {
            const float t = static_cast<float>(i) / 100.f;
            Vector3 point, tangent;
            curve.getPoint(t, point);
            curve.getTangent(t, tangent);
            Vector3 across{tangent.z, 0.f, -tangent.x};
            if (across.length() < 1e-6f) continue;
            across.normalize();

            CHECK(coveredXZ(geometry, point));
            Vector3 offset;
            offset.copy(point).addScaledVector(across, reach);
            const bool right = coveredXZ(geometry, offset);
            offset.copy(point).addScaledVector(across, -reach);
            const bool left = coveredXZ(geometry, offset);
            if (innerToo) {
                CHECK(left);
                CHECK(right);
            } else {
                // Inside a bend tighter than the half-width the offset has no
                // region left to cover; the outer side always has.
                CHECK((left || right));
            }
        }
    }

    CatmullRomCurve3 defaultSpline() {

        return CatmullRomCurve3({Vector3(-3, 0.5f, 1.5f), Vector3(-1, 0.5f, -1),
                                 Vector3(1, 0.5f, -1), Vector3(3, 0.5f, 1.5f)});
    }

    CatmullRomCurve3 cornerSpline() {

        return CatmullRomCurve3({Vector3(-10, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 10)});
    }

}// namespace


TEST_CASE("a road is smooth and full width where its offsets fit", "[extras]") {

    // The editor's default spline, narrow enough that its tightest bend (a
    // curvature radius of about 1.35 m) clears the half-width: neither offset
    // ties a loop, there is nothing to trim, and the road is exactly as wide as
    // it was asked to be from end to end.
    auto curve = defaultSpline();
    auto geometry = RoadGeometry::create(curve, 2.f, 72, 4.f);
    REQUIRE(geometry != nullptr);

    const auto left = edgeChain(*geometry, 0.f);
    const auto right = edgeChain(*geometry, 1.f);
    REQUIRE(left.size() > 8);

    // No trim, so the two edges still run vertex for vertex, exactly `width`
    // apart and level side to side.
    REQUIRE(left.size() == right.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        // Never narrower than the width, and no wider than the miter join at
        // this sampling makes a cross-section through a bend: the offset of a
        // polyline is its mitered one, so a corner spans width/cos(theta/2).
        const float across = left[i].distanceTo(right[i]);
        CHECK(across >= 2.f - 1e-4f);
        CHECK(across <= 2.f * 1.01f);
        CHECK(std::abs(left[i].y - right[i].y) < 1e-4f);
    }

    CHECK(sharpCount(left) == 0);
    CHECK(sharpCount(right) == 0);
    // ...and "smooth" is not merely "under the separator": the edges turn no
    // faster than the sampling of the curve they came from.
    CHECK(smoothest(left) < 8.f);
    CHECK(smoothest(right) < 8.f);

    checkNoDownwardFace(*geometry);
    checkMonotoneU(*geometry);
    checkCovers(*geometry, curve, 1.f, /*innerToo*/ true);

    // Flat spline, flat road: every normal points straight up, which is also
    // what the winding has to agree with or the surface is invisible.
    const auto* normal = geometry->getAttribute<float>("normal");
    REQUIRE(normal != nullptr);
    for (int i = 0; i < normal->count(); ++i) {
        CHECK(std::abs(normal->getY(i) - 1.f) < 1e-3f);
    }
}

TEST_CASE("a road wider than its own bends trims one corner per bend", "[extras]") {

    // The same default spline at the default width. Both of its bends are
    // tighter than the two metre half-width, so the inner offset ties a
    // swallowtail in each — and cutting them leaves that edge with exactly two
    // corners and nothing else. The outer edge never loops: it keeps every one
    // of its vertices and stays as smooth as the sampling.
    auto curve = defaultSpline();
    auto geometry = RoadGeometry::create(curve, 4.f, 72, 4.f);

    const auto left = edgeChain(*geometry, 0.f);
    const auto right = edgeChain(*geometry, 1.f);

    CHECK(sharpCount(left) == 2);
    CHECK(sharpCount(right) == 0);
    CHECK(smoothest(left) < 8.f);
    CHECK(smoothest(right) < 8.f);
    CHECK(right.size() == 73);
    CHECK(left.size() < right.size());

    checkNoDownwardFace(*geometry);
    checkMonotoneU(*geometry);
    // The centreline and the outer half are paved the whole way; inside a bend
    // tighter than the half-width there is no inner offset left to pave.
    checkCovers(*geometry, curve, 2.f, /*innerToo*/ false);

    // Full width where it starts and where it ends, neither of which is a bend.
    CHECK(std::abs(left.front().distanceTo(right.front()) - 4.f) < 1e-3f);
    CHECK(std::abs(left.back().distanceTo(right.back()) - 4.f) < 1e-3f);
}

TEST_CASE("a bend tighter than the half-width is trimmed to one corner", "[extras]") {

    // The corner that used to fold: a six metre road through a bend of about
    // two. The inner offset runs past the centre of curvature and crosses
    // itself — a swallowtail — and cutting that loop out leaves the inner edge
    // with ONE corner, exactly where the swept region has one. The outer edge
    // never loops and stays smooth.
    auto curve = cornerSpline();
    auto geometry = RoadGeometry::create(curve, 6.f, 48, 4.f);

    const auto left = edgeChain(*geometry, 0.f);
    const auto right = edgeChain(*geometry, 1.f);

    // The turn is toward the left edge, so that is the one that loops.
    CHECK(sharpCount(left) == 1);
    CHECK(sharpCount(right) == 0);
    CHECK(smoothest(left) < 20.f);
    CHECK(smoothest(right) < 20.f);
    // The trim took vertices off the inner edge and none off the outer.
    CHECK(left.size() < right.size());

    checkNoDownwardFace(*geometry);
    checkMonotoneU(*geometry);
    checkCovers(*geometry, curve, 3.f, /*innerToo*/ false);

    // The road is still exactly as wide as it was asked to be where it starts
    // and where it ends, neither of which is in the bend.
    CHECK(std::abs(left.front().distanceTo(right.front()) - 6.f) < 1e-3f);
    CHECK(std::abs(left.back().distanceTo(right.back()) - 6.f) < 1e-3f);
}

TEST_CASE("a closed road welds its loop", "[extras]") {

    CatmullRomCurve3 curve({Vector3(-4, 0, -4), Vector3(4, 0, -4),
                            Vector3(4, 0, 4), Vector3(-4, 0, 4)},
                           /*closed*/ true);
    auto geometry = RoadGeometry::create(curve, 2.f, 96, 4.f, /*closed*/ true);

    const auto left = edgeChain(*geometry, 0.f);
    const auto right = edgeChain(*geometry, 1.f);

    // The seam vertex is a DUPLICATE, not a shared index: it carries
    // u = totalLength/uvLength where the first carries u = 0. What makes it a
    // weld is that the positions coincide.
    CHECK(left.front().distanceTo(left.back()) < 1e-4f);
    CHECK(right.front().distanceTo(right.back()) < 1e-4f);

    const auto u = edgeU(*geometry, 0.f);
    CHECK(u.front() < u.back());
    CHECK(u.back() > 1.f);
    CHECK(sharpCount(left) == 0);
    CHECK(sharpCount(right) == 0);
    checkNoDownwardFace(*geometry);
}

TEST_CASE("a climbing road rises with the grade", "[extras]") {

    // A road that climbs as it runs: the two edges sit at the SAME height (a
    // road banks nowhere) and the normals tip with the slope.
    CatmullRomCurve3 curve({Vector3(0, 0, 0), Vector3(0, 2, 4), Vector3(0, 4, 8)});
    auto geometry = RoadGeometry::create(curve, 5.f, 48, 1.f);

    const auto left = edgeChain(*geometry, 0.f);
    const auto right = edgeChain(*geometry, 1.f);
    REQUIRE(left.size() == right.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        CHECK(std::abs(left[i].y - right[i].y) < 1e-4f);
        CHECK(std::abs(left[i].distanceTo(right[i]) - 5.f) < 1e-3f);
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

TEST_CASE("the collider solid is closed", "[extras]") {

    // What a static collider is cooked from: the surface, its underside, and a
    // wall along every boundary edge. Closed means every edge is shared by
    // exactly two triangles — an opening is a way into a solid a body has no
    // business being inside of, and PhysX will happily let it in.
    const auto check = [](const std::shared_ptr<RoadGeometry>& surface) {
        auto solid = RoadGeometry::solid(*surface, 0.25f);
        const auto* position = solid->getAttribute<float>("position");
        REQUIRE(position != nullptr);
        REQUIRE(solid->getIndex() != nullptr);
        const auto& index = solid->getIndex()->array();
        REQUIRE(index.size() >= 12);

        std::unordered_map<std::uint64_t, int> edges;
        for (std::size_t t = 0; t + 2 < index.size(); t += 3) {
            for (int e = 0; e < 3; ++e) {
                const auto a = index[t + static_cast<std::size_t>(e)];
                const auto b = index[t + static_cast<std::size_t>((e + 1) % 3)];
                const auto key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) |
                                 static_cast<std::uint64_t>(std::max(a, b));
                ++edges[key];
            }
        }
        int open = 0;
        for (const auto& [key, count] : edges) {
            if (count != 2) ++open;
        }
        CHECK(open == 0);

        // The top of the solid is the surface itself, and the underside a real
        // 0.25 m below it.
        surface->computeBoundingBox();
        REQUIRE(solid->boundingBox.has_value());
        REQUIRE(surface->boundingBox.has_value());
        CHECK(std::abs(solid->boundingBox->max().y - surface->boundingBox->max().y) < 1e-5f);
        CHECK(std::abs(solid->boundingBox->min().y -
                       (surface->boundingBox->min().y - 0.25f)) < 1e-5f);
    };

    auto smooth = defaultSpline();
    check(RoadGeometry::create(smooth, 4.f, 72, 4.f));
    // Trimmed, so the surface has a fan and a corner in it.
    auto corner = cornerSpline();
    check(RoadGeometry::create(corner, 6.f, 48, 4.f));
    // Closed, so the seam duplicates have to weld or the solid grows a wall
    // straight across the road.
    CatmullRomCurve3 loop({Vector3(-4, 0, -4), Vector3(4, 0, -4),
                           Vector3(4, 0, 4), Vector3(-4, 0, 4)},
                          /*closed*/ true);
    check(RoadGeometry::create(loop, 2.f, 96, 4.f, /*closed*/ true));
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
