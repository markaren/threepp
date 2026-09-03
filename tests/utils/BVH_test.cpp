
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/curves/LineCurve.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/TubeGeometry.hpp"
#include "threepp/utils/BVH.hpp"
#include "threepp/utils/TriangleIntersect.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    // A tube swept from p0 to p1, the shape colnav lifts a predicted track into:
    // x/z are position, y is time.
    std::shared_ptr<TubeGeometry> tube(const Vector3& p0, const Vector3& p1, float radius) {
        auto path = std::make_shared<LineCurve3>(p0, p1);
        return TubeGeometry::create(path, 16, radius, 8, false);
    }

    BVH buildTube(const Vector3& p0, const Vector3& p1, float radius, int leafSize = 4) {
        BVH bvh(leafSize);
        bvh.build(*tube(p0, p1, radius));
        return bvh;
    }

    // Squared distance from a point to the nearest triangle of a geometry,
    // computed by brute force — the oracle for "this point is on that surface".
    float distanceSqToSurface(const BufferGeometry& g, const Vector3& p) {
        const auto* pos = g.getAttribute<float>("position");
        const auto* idx = g.getIndex();
        REQUIRE(pos);
        REQUIRE(idx);

        float best = std::numeric_limits<float>::infinity();
        Vector3 closest;
        for (int i = 0; i < idx->count(); i += 3) {
            const auto a = idx->getX(i), b = idx->getX(i + 1), c = idx->getX(i + 2);
            const Triangle tri(
                    Vector3(pos->getX(a), pos->getY(a), pos->getZ(a)),
                    Vector3(pos->getX(b), pos->getY(b), pos->getZ(b)),
                    Vector3(pos->getX(c), pos->getY(c), pos->getZ(c)));
            tri.closestPointToPoint(p, closest);
            best = std::min(best, closest.distanceToSquared(p));
        }
        return best;
    }

    std::vector<float> boxPositions(float cx, float cy, float cz, float size) {
        const auto geometry = BoxGeometry::create(size, size, size);
        geometry->translate(cx, cy, cz);
        const auto* pos = geometry->getAttribute<float>("position");
        return {pos->array().begin(), pos->array().end()};
    }

    std::vector<unsigned int> boxIndices(float size) {
        const auto geometry = BoxGeometry::create(size, size, size);
        const auto& a = geometry->getIndex()->array();
        return {a.begin(), a.end()};
    }

}// namespace


TEST_CASE("triangle-triangle predicates", "[BVH]") {

    SECTION("crossing triangles intersect on the shared segment") {
        const Triangle a({-1, 0, 0}, {1, 0, 0}, {0, 0, 1});
        const Triangle b({0, -1, 0.25f}, {0, 1, 0.25f}, {0, 0, 0.75f});

        Vector3 p;
        REQUIRE(detail::triTriIntersectionPoint(a, b, p));
        // The shared segment lies in both planes: y = 0 (a) and x = 0 (b).
        CHECK_THAT(p.x, WithinAbs(0.f, 1e-4f));
        CHECK_THAT(p.y, WithinAbs(0.f, 1e-4f));
    }

    SECTION("triangles whose boxes overlap but whose planes do not touch") {
        // Two slanted triangles in bounding boxes that overlap heavily; the
        // surfaces stay apart. This is the case an AABB leaf test gets wrong.
        const Triangle a({0, 0, 0}, {1, 1, 0}, {0, 0, 1});
        const Triangle b({0.4f, 0, 0}, {1.4f, 1, 0}, {0.4f, 0, 1});

        CHECK_FALSE(detail::triTriOverlap(a, b));
        CHECK(detail::triTriDistanceSq(a, b) > 0.f);
    }

    SECTION("coplanar overlap is reported") {
        const Triangle a({0, 0, 0}, {2, 0, 0}, {0, 2, 0});
        const Triangle b({0.5f, 0.5f, 0}, {2.5f, 0.5f, 0}, {0.5f, 2.5f, 0});
        CHECK(detail::triTriOverlap(a, b));
        CHECK_THAT(detail::triTriDistanceSq(a, b), WithinAbs(0.f, 1e-6f));
    }

    SECTION("coplanar but disjoint is not") {
        const Triangle a({0, 0, 0}, {1, 0, 0}, {0, 1, 0});
        const Triangle b({5, 0, 0}, {6, 0, 0}, {5, 1, 0});
        CHECK_FALSE(detail::triTriOverlap(a, b));
        CHECK_THAT(std::sqrt(detail::triTriDistanceSq(a, b)), WithinAbs(4.f, 1e-4f));
    }

    SECTION("parallel triangles are exactly their separation apart") {
        const Triangle a({0, 0, 0}, {1, 0, 0}, {0, 1, 0});
        const Triangle b({0, 0, 3}, {1, 0, 3}, {0, 1, 3});
        CHECK_FALSE(detail::triTriOverlap(a, b));
        CHECK_THAT(std::sqrt(detail::triTriDistanceSq(a, b)), WithinAbs(3.f, 1e-4f));
    }

    SECTION("degenerate triangles intersect nothing") {
        const Triangle degenerate({0, 0, 0}, {1, 0, 0}, {2, 0, 0});
        const Triangle t({-1, -1, 0}, {3, -1, 0}, {1, 2, 0});
        CHECK_FALSE(detail::triTriOverlap(degenerate, t));
    }

    SECTION("box distance") {
        const Box3 a({0, 0, 0}, {1, 1, 1});
        const Box3 b({4, 0, 0}, {5, 1, 1});
        CHECK_THAT(std::sqrt(detail::boxDistanceSq(a, b)), WithinAbs(3.f, 1e-5f));
        CHECK_THAT(detail::boxDistanceSq(a, a), WithinAbs(0.f, 1e-6f));
    }
}

TEST_CASE("BVH pair queries on swept tubes", "[BVH]") {

    // x, y=time, z — the colnav layout.
    const auto own = buildTube({0, 0, 0}, {100, 60, 100}, 5.f);
    const auto target = buildTube({100, 0, 0}, {0, 60, 100}, 5.f);
    const auto clear = buildTube({0, 0, 500}, {100, 60, 600}, 5.f);
    // Runs alongside `own`, offset 12 m perpendicular to it (the offset is
    // orthogonal to the sweep direction, so the centrelines stay exactly 12 m
    // apart and the surfaces exactly 2 m apart). Their node boxes overlap almost
    // everywhere; the surfaces never do.
    constexpr float offset = 8.485281f;// 12 / sqrt(2)
    const auto parallelTube = buildTube({offset, 0, -offset}, {100 + offset, 60, 100 - offset}, 5.f);

    SECTION("crossing tubes intersect") {
        CHECK(BVH::intersects(own, target));
    }

    SECTION("intersection points lie on both surfaces") {
        const auto results = BVH::intersect(own, Matrix4(), target, Matrix4(), true);
        REQUIRE_FALSE(results.empty());

        const auto ownGeometry = tube({0, 0, 0}, {100, 60, 100}, 5.f);
        const auto targetGeometry = tube({100, 0, 0}, {0, 60, 100}, 5.f);

        float meanTime = 0.f;
        for (const auto& r : results) {
            CHECK(r.idxA >= 0);
            CHECK(r.idxB >= 0);
            CHECK(distanceSqToSurface(*ownGeometry, r.position) < 1e-2f);
            CHECK(distanceSqToSurface(*targetGeometry, r.position) < 1e-2f);
            meanTime += r.position.y;
        }
        meanTime /= static_cast<float>(results.size());
        // The tracks cross near the middle of the 0..60 time span.
        CHECK(meanTime > 20.f);
        CHECK(meanTime < 40.f);
    }

    SECTION("a near miss whose boxes overlap is not a hit") {
        CHECK_FALSE(BVH::intersects(own, parallelTube));
        CHECK(BVH::intersect(own, Matrix4(), parallelTube, Matrix4(), true).empty());

        const float d = BVH::distance(own, parallelTube);
        CHECK(d > 0.f);
        CHECK(d < 3.f);// 12 m centre-to-centre minus two 5 m radii
    }

    SECTION("a far tube is neither a hit nor near") {
        CHECK_FALSE(BVH::intersects(own, clear));
        CHECK(BVH::distance(own, clear) > 300.f);
    }

    SECTION("a distance cutoff rejects anything beyond it") {
        CHECK(std::isinf(BVH::distance(own, clear, Matrix4(), Matrix4(), 50.f)));
        // The true gap is 2 m, so a 10 m cutoff still measures it exactly.
        CHECK_THAT(BVH::distance(own, parallelTube, Matrix4(), Matrix4(), 10.f),
                   WithinAbs(BVH::distance(own, parallelTube), 1e-4f));
        CHECK(std::isinf(BVH::distance(own, parallelTube, Matrix4(), Matrix4(), 1.f)));
    }

    SECTION("intersecting meshes are zero distance apart") {
        CHECK_THAT(BVH::distance(own, target), WithinAbs(0.f, 1e-5f));
        CHECK_THAT(BVH::distance(own, own), WithinAbs(0.f, 1e-5f));
    }
}

TEST_CASE("BVH built from raw arrays", "[BVH]") {

    const auto indices = boxIndices(2.f);

    BVH a(1);
    a.build(boxPositions(0, 0, 0, 2.f), indices);

    BVH b(1);
    b.build(boxPositions(5, 0, 0, 2.f), indices);

    BVH overlapping(1);
    overlapping.build(boxPositions(1, 0, 0, 2.f), indices);

    CHECK(a.triangleCount() == indices.size() / 3);
    CHECK_THAT(a.boundingBox().min().x, WithinAbs(-1.f, 1e-5f));

    SECTION("separated boxes are their gap apart") {
        CHECK_FALSE(BVH::intersects(a, b));
        // Cubes of side 2 centred 5 apart: 5 - 1 - 1 = 3.
        CHECK_THAT(BVH::distance(a, b), WithinAbs(3.f, 1e-4f));
    }

    SECTION("overlapping boxes intersect") {
        CHECK(BVH::intersects(a, overlapping));
        CHECK_THAT(BVH::distance(a, overlapping), WithinAbs(0.f, 1e-5f));
    }

    SECTION("a transform moves the mesh") {
        Matrix4 m;
        m.makeTranslation(5, 0, 0);
        // `a` shifted onto `b`.
        CHECK(BVH::intersects(a, b, m, Matrix4()));
        CHECK_THAT(BVH::distance(a, b, m, Matrix4()), WithinAbs(0.f, 1e-5f));
    }

    SECTION("an empty BVH is infinitely far from everything") {
        const BVH empty;
        CHECK_FALSE(BVH::intersects(a, empty));
        CHECK(std::isinf(BVH::distance(a, empty)));
    }

    SECTION("a non-indexed soup builds the same tree") {
        const auto soupGeometry = BoxGeometry::create(2.f, 2.f, 2.f)->toNonIndexed();
        const auto* pos = soupGeometry->getAttribute<float>("position");
        BVH soup(1);
        soup.build({pos->array().begin(), pos->array().end()});
        CHECK(soup.triangleCount() == a.triangleCount());
        CHECK(BVH::intersects(soup, overlapping));
    }
}
