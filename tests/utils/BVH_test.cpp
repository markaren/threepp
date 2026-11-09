
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/curves/LineCurve.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/TubeGeometry.hpp"
#include "threepp/math/Euler.hpp"
#include "threepp/math/Sphere.hpp"
#include "threepp/utils/BVH.hpp"
#include "threepp/utils/TriangleIntersect.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

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

namespace {

    // Deterministic across compilers: std::*_distribution is implementation-defined.
    struct Rng {
        std::uint64_t state;

        float next() {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<float>(state >> 40) / static_cast<float>(1u << 24);
        }

        float range(float lo, float hi) {
            return lo + (hi - lo) * next();
        }

        Vector3 unitVector() {
            while (true) {
                Vector3 v(range(-1, 1), range(-1, 1), range(-1, 1));
                const float len = v.length();
                if (len > 1e-3f && len <= 1.f) return v.divideScalar(len);
            }
        }
    };

    struct Mesh {
        std::vector<float> positions;
        std::vector<unsigned int> indices;

        [[nodiscard]] std::size_t triangleCount() const {
            return indices.size() / 3;
        }

        [[nodiscard]] Triangle triangle(std::size_t i) const {
            const auto v = [this](unsigned int k) {
                return Vector3(positions[k * 3], positions[k * 3 + 1], positions[k * 3 + 2]);
            };
            return {v(indices[i * 3]), v(indices[i * 3 + 1]), v(indices[i * 3 + 2])};
        }
    };

    // A lumpy sphere: about 2 * rings * segments triangles, the radius wobbled
    // by a few lobes and a little per-vertex noise.
    Mesh deformedSphere(int segments, int rings, const Vector3& center, float radius, std::uint64_t seed) {
        Mesh mesh;
        Rng rng{seed};
        constexpr float pi = 3.14159265358979f;
        for (int r = 0; r <= rings; ++r) {
            const float theta = pi * static_cast<float>(r) / static_cast<float>(rings);
            for (int s = 0; s <= segments; ++s) {
                const float phi = 2.f * pi * static_cast<float>(s % segments) / static_cast<float>(segments);
                const float wobble = 1.f + 0.15f * std::sin(5.f * theta) * std::cos(3.f * phi) + rng.range(-0.01f, 0.01f);
                const float rr = radius * wobble;
                mesh.positions.push_back(center.x + rr * std::sin(theta) * std::cos(phi));
                mesh.positions.push_back(center.y + rr * std::cos(theta));
                mesh.positions.push_back(center.z + rr * std::sin(theta) * std::sin(phi));
            }
        }
        const auto vertex = [segments](int r, int s) {
            return static_cast<unsigned int>(r * (segments + 1) + s);
        };
        for (int r = 0; r < rings; ++r) {
            for (int s = 0; s < segments; ++s) {
                const auto a = vertex(r, s), b = vertex(r + 1, s), c = vertex(r + 1, s + 1), d = vertex(r, s + 1);
                if (r != 0) mesh.indices.insert(mesh.indices.end(), {a, b, d});
                if (r != rings - 1) mesh.indices.insert(mesh.indices.end(), {b, c, d});
            }
        }
        return mesh;
    }

    constexpr float rayEps = 1e-4f;// the segment-endpoint epsilon in BVH.cpp

    bool sameDistance(float a, float b) {
        return std::abs(a - b) <= std::max(1e-6f, 1e-5f * std::max(std::abs(a), std::abs(b)));
    }

}// namespace

TEST_CASE("BVH raycast matches a brute-force loop", "[BVH]") {

    const auto mesh = deformedSphere(160, 157, {0.1f, -0.2f, 0.05f}, 1.f, 1234);
    REQUIRE(mesh.triangleCount() > 49000);

    BVH bvh;
    bvh.build(mesh.positions, mesh.indices);
    REQUIRE(bvh.triangleCount() == mesh.triangleCount());

    std::vector<Triangle> tris;
    tris.reserve(mesh.triangleCount());
    for (std::size_t i = 0; i < mesh.triangleCount(); ++i) tris.push_back(mesh.triangle(i));

    const float maxDistances[] = {std::numeric_limits<float>::infinity(), 3.f, 1.2f, 0.4f};

    Rng rng{42};
    int hits = 0, mismatches = 0, anyMismatches = 0;
    constexpr int rayCount = 10000;
    std::vector<float> triDistance(tris.size());
    for (int r = 0; r < rayCount; ++r) {
        const Vector3 origin(rng.range(-2.f, 2.f), rng.range(-2.f, 2.f), rng.range(-2.f, 2.f));
        Vector3 direction = rng.unitVector();
        // Every 16th ray runs along an axis, so the slab test sees zero direction components.
        if (r % 16 == 0) {
            const int axis = r / 16 % 3;
            direction.set(0, 0, 0);
            direction[axis] = (r / 48) % 2 ? 1.f : -1.f;
        }
        const Ray ray(origin, direction);

        // Nearest hit at or beyond rayEps; a shorter limit keeps it or finds nothing.
        float bruteDistance = std::numeric_limits<float>::infinity();
        Vector3 point;
        std::ranges::fill(triDistance, std::numeric_limits<float>::infinity());
        for (std::size_t i = 0; i < tris.size(); ++i) {
            if (!ray.intersectTriangle(tris[i].a(), tris[i].b(), tris[i].c(), false, point)) continue;
            const float d = point.distanceTo(origin);
            if (d < rayEps) continue;
            triDistance[i] = d;
            bruteDistance = std::min(bruteDistance, d);
        }

        for (const float maxDistance : maxDistances) {
            const bool expectHit = std::isfinite(bruteDistance) && bruteDistance <= maxDistance - rayEps;
            const auto hit = bvh.raycast(ray, maxDistance);
            if (bvh.raycastAny(ray, maxDistance) != expectHit) ++anyMismatches;
            if (hit.has_value() != expectHit) {
                ++mismatches;
                continue;
            }
            if (!hit) continue;
            ++hits;
            // A tie on a shared edge may pick either triangle: both must be hit at that distance.
            Vector3 along;
            const bool ok = hit->triangleIndex >= 0 && hit->triangleIndex < static_cast<int>(tris.size()) &&
                            sameDistance(hit->distance, bruteDistance) &&
                            sameDistance(triDistance[hit->triangleIndex], bruteDistance) &&
                            hit->normal.dot(direction) <= 0.f &&
                            std::abs(hit->normal.length() - 1.f) < 1e-4f &&
                            hit->point.distanceTo(ray.at(hit->distance, along)) < 1e-4f;
            if (!ok) ++mismatches;
        }
    }

    CHECK(hits > rayCount / 3);// a fair share of the rays hit
    CHECK(mismatches == 0);
    CHECK(anyMismatches == 0);
}

TEST_CASE("BVH box and sphere queries match a brute-force loop", "[BVH]") {

    const auto mesh = deformedSphere(120, 100, {0, 0, 0}, 1.f, 99);
    BVH bvh;
    bvh.build(mesh.positions, mesh.indices);

    Matrix4 rotated;
    rotated.makeRotationFromEuler(Euler(0.3f, -0.7f, 0.2f)).setPosition(0.4f, -0.1f, 0.25f);
    Matrix4 scaled;
    scaled.makeScale(1.5f, 0.75f, 1.f).premultiply(rotated);
    const Matrix4 matrices[] = {Matrix4(), rotated, scaled};

    Rng rng{7};
    int boxHits = 0, sphereHits = 0;
    for (const auto& m : matrices) {
        for (int q = 0; q < 200; ++q) {
            const Vector3 c(rng.range(-1.5f, 1.5f), rng.range(-1.5f, 1.5f), rng.range(-1.5f, 1.5f));
            const Vector3 half(rng.range(0.01f, 0.4f), rng.range(0.01f, 0.4f), rng.range(0.01f, 0.4f));
            const Box3 box(Vector3(c).sub(half), Vector3(c).add(half));
            const Sphere sphere(c, rng.range(0.01f, 0.5f));

            std::vector<int> expectBox, expectSphere;
            Box3 triBox;
            Vector3 a, b, cc, closest;
            for (std::size_t i = 0; i < mesh.triangleCount(); ++i) {
                const auto tri = mesh.triangle(i);
                triBox.makeEmpty();
                triBox.expandByPoint(tri.a());
                triBox.expandByPoint(tri.b());
                triBox.expandByPoint(tri.c());
                triBox.applyMatrix4(m);
                if (box.intersectsBox(triBox)) expectBox.push_back(static_cast<int>(i));

                const Triangle world(a.copy(tri.a()).applyMatrix4(m), b.copy(tri.b()).applyMatrix4(m), cc.copy(tri.c()).applyMatrix4(m));
                world.closestPointToPoint(sphere.center, closest);
                if (closest.distanceToSquared(sphere.center) <= sphere.radius * sphere.radius) {
                    expectSphere.push_back(static_cast<int>(i));
                }
            }

            auto gotBox = bvh.intersect(box, m);
            auto gotSphere = bvh.intersect(sphere, m);
            std::ranges::sort(gotBox);
            std::ranges::sort(gotSphere);
            CHECK(gotBox == expectBox);
            CHECK(gotSphere == expectSphere);
            boxHits += !expectBox.empty();
            sphereHits += !expectSphere.empty();
        }
    }
    // The random shapes both touch and miss the mesh.
    CHECK(boxHits > 50);
    CHECK(boxHits < 550);
    CHECK(sphereHits > 50);
    CHECK(sphereHits < 550);
}

TEST_CASE("BVH mesh-mesh queries on a procedural pair", "[BVH]") {

    const auto meshA = deformedSphere(64, 48, {0, 0, 0}, 1.f, 5);
    const auto meshB = deformedSphere(48, 40, {1.4f, 0.3f, -0.2f}, 0.8f, 6);
    BVH a, b;
    a.build(meshA.positions, meshA.indices);
    b.build(meshB.positions, meshB.indices);

    Matrix4 apart;
    apart.makeRotationFromEuler(Euler(0.5f, 0.1f, -0.4f)).setPosition(1.6f, 0.4f, 0.f);

    CHECK(BVH::intersects(a, b));
    CHECK_FALSE(BVH::intersects(a, b, Matrix4(), apart));

    const auto pairs = BVH::intersect(a, Matrix4(), b, Matrix4(), true);
    const auto coarse = BVH::intersect(a, Matrix4(), b, Matrix4(), false);
    const float gap = BVH::distance(a, b, Matrix4(), apart);

    // Expected values from the recursive median-split BVH this one replaced.
    CHECK(pairs.size() == 216u);
    std::int64_t pairSum = 0;// order-free fingerprint of which original triangle pairs were reported
    for (const auto& p : pairs) pairSum += static_cast<std::int64_t>(p.idxA) * 100003 + p.idxB;
    CHECK(pairSum == 56253238848);
    for (const auto& p : pairs) {
        REQUIRE(p.idxA >= 0);
        REQUIRE(p.idxB >= 0);
        REQUIRE(p.idxA < static_cast<int>(meshA.triangleCount()));
        REQUIRE(p.idxB < static_cast<int>(meshB.triangleCount()));
        // The reported indices are the original triangles, and they do cross.
        CHECK(detail::triTriOverlap(meshA.triangle(p.idxA), meshB.triangle(p.idxB)));
    }
    // One entry per overlapping leaf pair: the count follows the tree's shape.
    CHECK_FALSE(coarse.empty());

    CHECK_THAT(BVH::distance(a, b), WithinAbs(0.f, 1e-6f));
    CHECK_THAT(gap, WithinRel(0.987624466f, 1e-5f));
}

namespace {

    // A flat grid of unit quads in the y = 0 plane, x and z in [0, cells]. The
    // vertices are whole numbers, so a ray test against it is exact arithmetic.
    Mesh floorGrid(int cells) {
        Mesh mesh;
        for (int z = 0; z <= cells; ++z) {
            for (int x = 0; x <= cells; ++x) {
                mesh.positions.insert(mesh.positions.end(), {static_cast<float>(x), 0.f, static_cast<float>(z)});
            }
        }
        const auto vertex = [cells](int x, int z) {
            return static_cast<unsigned int>(z * (cells + 1) + x);
        };
        for (int z = 0; z < cells; ++z) {
            for (int x = 0; x < cells; ++x) {
                const auto a = vertex(x, z), b = vertex(x + 1, z), c = vertex(x + 1, z + 1), d = vertex(x, z + 1);
                mesh.indices.insert(mesh.indices.end(), {a, d, b, b, d, c});
            }
        }
        return mesh;
    }

    bool onTriangle(const Triangle& tri, const Vector3& p) {
        Vector3 closest;
        tri.closestPointToPoint(p, closest);
        return closest.distanceToSquared(p) < 1e-10f;
    }

}// namespace

TEST_CASE("BVH leaves out triangles with a non-finite vertex", "[BVH]") {

    // A no-data cell in a DEM or point cloud: one vertex with a NaN or infinite coordinate.
    const float badValues[] = {std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()};
    const Ray down({0, 1, 0}, {0, -1, 0});

    SECTION("a NaN or infinite coordinate") {
        for (const float bad : badValues) {
            // A mesh big enough to be split: 32 floor triangles, and one bad
            // triangle placed in the middle of the source order, so every
            // triangle after it would be misnumbered if the bad one were
            // skipped rather than left out of the tree. The bad value is tried
            // as a height, then as an x the build bins on.
            for (const int axis : {1, 0}) {
                for (const int leafSize : {1, 8}) {
                    CAPTURE(bad, axis, leafSize);
                    auto mesh = floorGrid(4);
                    const auto badVertex = static_cast<unsigned int>(mesh.positions.size() / 3);
                    mesh.positions.insert(mesh.positions.end(), {2.5f, 0.f, 1.5f});
                    mesh.positions[badVertex * 3 + axis] = bad;
                    constexpr std::size_t badTriangle = 10;
                    mesh.indices.insert(mesh.indices.begin() + badTriangle * 3, {0u, 1u, badVertex});

                    BVH bvh(leafSize);
                    bvh.build(mesh.positions, mesh.indices);
                    CHECK(bvh.triangleCount() == mesh.triangleCount() - 1);

                    // Straight down onto the inside of each floor triangle: the
                    // hit is that triangle, numbered as in the source.
                    int found = 0;
                    for (int z = 0; z < 4; ++z) {
                        for (int x = 0; x < 4; ++x) {
                            for (const auto& offset : {Vector3(0.25f, 1, 0.25f), Vector3(0.75f, 1, 0.75f)}) {
                                const Ray ray(Vector3(static_cast<float>(x), 0, static_cast<float>(z)).add(offset), {0, -1, 0});
                                const auto hit = bvh.raycast(ray);
                                REQUIRE(hit);
                                CHECK(hit->distance == 1.f);
                                REQUIRE(hit->triangleIndex >= 0);
                                REQUIRE(hit->triangleIndex < static_cast<int>(mesh.triangleCount()));
                                CHECK(hit->triangleIndex != static_cast<int>(badTriangle));
                                CHECK(onTriangle(mesh.triangle(hit->triangleIndex), hit->point));
                                CHECK(bvh.raycastAny(ray, 2.f));
                                ++found;
                            }
                        }
                    }
                    CHECK(found == 32);

                    // Off the floor nothing is hit; a box over everything finds the 32 good triangles.
                    CHECK_FALSE(bvh.raycast(Ray({6, 1, 6}, {0, -1, 0})));
                    auto all = bvh.intersect(Box3({-1, -1, -1}, {5, 1, 5}));
                    std::ranges::sort(all);
                    CHECK(all.size() == mesh.triangleCount() - 1);
                    CHECK(std::ranges::find(all, static_cast<int>(badTriangle)) == all.end());
                }
            }

            // A single leaf: one floor quad and the bad triangle reaching out
            // to x = z = 3. A ray down at (2.5, 2.5) misses the quad and must
            // not report the bad one.
            CAPTURE(bad);
            Mesh mesh;
            mesh.positions = {0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 3, bad, 3};
            mesh.indices = {0, 3, 1, 1, 3, 2, 1, 2, 4};
            BVH bvh;
            bvh.build(mesh.positions, mesh.indices);
            CHECK(bvh.triangleCount() == 2);

            const Ray miss({2.5f, 1, 2.5f}, {0, -1, 0});
            CHECK_FALSE(bvh.raycast(miss));
            CHECK_FALSE(bvh.raycastAny(miss, 10.f));
            const auto hit = bvh.raycast(Ray({0.5f, 1, 0.25f}, {0, -1, 0}));
            REQUIRE(hit);
            CHECK(hit->distance == 1.f);
        }
    }

    SECTION("only bad triangles build an empty tree") {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        BVH bvh;
        bvh.build({0, 0, 0, 1, 0, 0, nan, 0, 1});
        CHECK(bvh.triangleCount() == 0);
        CHECK_FALSE(bvh.raycast(down));
        CHECK(bvh.intersect(Box3({-10, -10, -10}, {10, 10, 10})).empty());
    }

    SECTION("a triangle too large for float edges never gives a NaN hit") {
        // The vertices are finite, but their differences overflow to infinity,
        // so the ray-triangle test returns NaN for a ray through one. Their
        // centroid z sums overflow to -inf too, next to four small triangles
        // with finite centroids, so the build bins an infinite centroid range.
        constexpr float big = 3e38f;
        std::vector<float> positions;
        for (int i = 0; i < 12; ++i) {
            const float y = -0.5f - static_cast<float>(i) * 0.1f;
            positions.insert(positions.end(), {-big, y, -big, big, y, -big, 0, y, big});
        }
        for (int i = 0; i < 4; ++i) {
            const float x = 10.f + static_cast<float>(i);
            positions.insert(positions.end(), {x, 0, 0, x, 0, 1, x + 1, 0, 0});
        }
        BVH bvh(1);
        bvh.build(positions);
        CHECK(bvh.triangleCount() == 16);

        const auto hit = bvh.raycast(down);
        if (hit) {
            CHECK(std::isfinite(hit->distance));
            CHECK(std::isfinite(hit->point.y));
        }
        CHECK(bvh.raycastAny(down, 10.f) == hit.has_value());

        const auto small = bvh.raycast(Ray({10.25f, 1, 0.25f}, {0, -1, 0}));
        REQUIRE(small);
        CHECK(small->distance == 1.f);
        CHECK(small->triangleIndex == 12);
    }
}
