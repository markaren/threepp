
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/Box3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Ray.hpp"
#include "threepp/math/Sphere.hpp"

#include <cmath>

using namespace threepp;

namespace {

    constexpr float eps = 0.0001f;

    Vector3 zero3{0, 0, 0};
    Vector3 one3{1, 1, 1};
    Vector3 two3{2, 2, 2};


}// namespace

TEST_CASE("Instancing") {

    Ray a;
    CHECK(a.origin == (zero3));
    CHECK(a.direction == (Vector3(0, 0, -1)));

    Ray b(two3, one3);
    CHECK(b.origin == (two3));
    CHECK(b.direction == (one3));
}

TEST_CASE("set") {

    Ray a;

    a.set(one3, one3);
    CHECK(a.origin == (one3));
    CHECK(a.direction == (one3));
}

TEST_CASE("at") {

    Ray a(one3, Vector3(0, 0, 1));
    Vector3 point;

    a.at(0, point);
    CHECK(point.equals(one3));
    a.at(-1, point);
    CHECK(point == (Vector3(1, 1, 0)));
    a.at(1, point);
    CHECK(point == (Vector3(1, 1, 2)));
}

TEST_CASE("lookAt") {

    Ray a(two3, one3);
    Vector3 target = one3;
    Vector3 expected = target.sub(two3).normalize();

    a.lookAt(target);
    CHECK(a.direction == (expected));
}

TEST_CASE("closestPointToPoint") {

    Ray a(one3, Vector3(0, 0, 1));
    Vector3 point;

    // behind the ray
    a.closestPointToPoint(zero3, point);
    CHECK(point.equals(one3));

    // front of the ray
    a.closestPointToPoint(Vector3(0, 0, 50), point);
    CHECK(point == (Vector3(1, 1, 50)));

    // exactly on the ray
    a.closestPointToPoint(one3, point);
    CHECK(point == (one3));
}

TEST_CASE("distanceToPoint") {

    Ray a(one3, Vector3(0, 0, 1));

    // behind the ray
    float b = a.distanceToPoint(zero3);
    CHECK_THAT(b, Catch::Matchers::WithinRel(std::sqrt(3.f)));

    // front of the ray
    float c = a.distanceToPoint(Vector3(0, 0, 50));
    CHECK_THAT(c, Catch::Matchers::WithinRel(std::sqrt(2.f)));

    // exactly on the ray
    float d = a.distanceToPoint(one3);
    CHECK_THAT(d, Catch::Matchers::WithinRel(0.f));
}

TEST_CASE("distanceSqToPoint") {

    Ray a(one3, Vector3(0, 0, 1));

    // behind the ray
    float b = a.distanceSqToPoint(zero3);
    CHECK_THAT(b, Catch::Matchers::WithinRel(3.f));

    // front of the ray
    float c = a.distanceSqToPoint(Vector3(0, 0, 50));
    CHECK_THAT(c, Catch::Matchers::WithinRel(2.f));

    // exactly on the ray
    float d = a.distanceSqToPoint(one3);
    CHECK_THAT(d, Catch::Matchers::WithinRel(0.f));
}

TEST_CASE("distanceSqToSegment") {

    Ray a(one3, Vector3(0, 0, 1));
    Vector3 ptOnLine;
    Vector3 ptOnSegment;

    {
        //segment in front of the ray
        auto v0 = Vector3(3, 5, 50);
        auto v1 = Vector3(50, 50, 50);// just a far away point
        auto distSqr = a.distanceSqToSegment(v0, v1, &ptOnLine, &ptOnSegment);

        CHECK(ptOnSegment.distanceTo(v0) < 0.0001f);
        CHECK(ptOnLine.distanceTo(Vector3(1, 1, 50)) < 0.0001f);
        // ((3-1) * (3-1) + (5-1) * (5-1) = 4 + 16 = 20
        CHECK(std::abs(distSqr - 20) < 0.0001f);
    }

    {
        //segment behind the ray
        auto v0 = Vector3(-50, -50, -50);// just a far away point
        auto v1 = Vector3(-3, -5, -4);
        auto distSqr = a.distanceSqToSegment(v0, v1, &ptOnLine, &ptOnSegment);

        CHECK(ptOnSegment.distanceTo(v1) < 0.0001f);
        CHECK(ptOnLine.distanceTo(one3) < 0.0001f);
        // ((-3-1) * (-3-1) + (-5-1) * (-5-1) + (-4-1) + (-4-1) = 16 + 36 + 25 = 77
        CHECK(std::abs(distSqr - 77) < 0.0001f);
    }

    {
        //exact intersection between the ray and the segment
        auto v0 = Vector3(-50, -50, -50);
        auto v1 = Vector3(50, 50, 50);
        auto distSqr = a.distanceSqToSegment(v0, v1, &ptOnLine, &ptOnSegment);

        CHECK(ptOnSegment.distanceTo(one3) < 0.0001f);
        CHECK(ptOnLine.distanceTo(one3) < 0.0001f);
        CHECK(distSqr < 0.0001f);
    }
}

TEST_CASE("intersectSphere") {
    const float TOL = 0.0001f;
    Vector3 point;

    // ray a0 origin located at ( 0, 0, 0 ) and points outward in negative-z direction
    auto a0 = Ray(zero3, Vector3(0, 0, -1));
    // ray a1 origin located at ( 1, 1, 1 ) and points left in negative-x direction
    auto a1 = Ray(one3, Vector3(-1, 0, 0));

    // sphere (radius of 2) located behind ray a0, should result in null
    auto b = Sphere(Vector3(0, 0, 3), 2);
    a0.intersectSphere(b, point);
    CHECK(point.isNan());

    // sphere (radius of 2) located in front of, but too far right of ray a0, should result in null
    b = Sphere(Vector3(3, 0, -1), 2);
    a0.intersectSphere(b, point);
    CHECK(point.isNan());

    // sphere (radius of 2) located below ray a1, should result in null
    b = Sphere(Vector3(1, -2, 1), 2);
    a1.intersectSphere(b, point);
    CHECK(point.isNan());

    // sphere (radius of 1) located to the left of ray a1, should result in intersection at 0, 1, 1
    b = Sphere(Vector3(-1, 1, 1), 1);
    a1.intersectSphere(b, point);
    CHECK(point.distanceTo(Vector3(0, 1, 1)) < TOL);

    // sphere (radius of 1) located in front of ray a0, should result in intersection at 0, 0, -1
    b = Sphere(Vector3(0, 0, -2), 1);
    a0.intersectSphere(b, point);
    CHECK(point.distanceTo(Vector3(0, 0, -1)) < TOL);

    // sphere (radius of 2) located in front & right of ray a0, should result in intersection at 0, 0, -1, or left-most edge of sphere
    b = Sphere(Vector3(2, 0, -1), 2);
    a0.intersectSphere(b, point);
    CHECK(point.distanceTo(Vector3(0, 0, -1)) < TOL);

    // same situation as above, but move the sphere a fraction more to the right, and ray a0 should now just miss
    b = Sphere(Vector3(2.01, 0, -1), 2);
    a0.intersectSphere(b, point);
    CHECK(point.isNan());

    // following QUnit.tests are for situations where the ray origin is inside the sphere

    // sphere (radius of 1) center located at ray a0 origin / sphere surrounds the ray origin, so the first intersect point 0, 0, 1,
    // is behind ray a0.  Therefore, second exit point on back of sphere will be returned: 0, 0, -1
    // thus keeping the intersection point always in front of the ray.
    b = Sphere(zero3, 1);
    a0.intersectSphere(b, point);
    CHECK(point.distanceTo(Vector3(0, 0, -1)) < TOL);

    // sphere (radius of 4) center located behind ray a0 origin / sphere surrounds the ray origin, so the first intersect point 0, 0, 5,
    // is behind ray a0.  Therefore, second exit point on back of sphere will be returned: 0, 0, -3
    // thus keeping the intersection point always in front of the ray.
    b = Sphere(Vector3(0, 0, 1), 4);
    a0.intersectSphere(b, point);
    CHECK(point.distanceTo(Vector3(0, 0, -3)) < TOL);

    // sphere (radius of 4) center located in front of ray a0 origin / sphere surrounds the ray origin, so the first intersect point 0, 0, 3,
    // is behind ray a0.  Therefore, second exit point on back of sphere will be returned: 0, 0, -5
    // thus keeping the intersection point always in front of the ray.
    b = Sphere(Vector3(0, 0, -1), 4);
    a0.intersectSphere(b, point);
    CHECK(point.distanceTo(Vector3(0, 0, -5)) < TOL);
}

TEST_CASE("intersectsSphere") {
    Ray a(one3.clone(), Vector3(0, 0, 1));
    Sphere b(zero3, 0.5);
    Sphere c(zero3, 1.5);
    Sphere d(one3, 0.1);
    Sphere e(two3, 0.1);
    Sphere f(two3, 1);

    CHECK(!a.intersectsSphere(b));
    CHECK(!a.intersectsSphere(c));
    CHECK(a.intersectsSphere(d));
    CHECK(!a.intersectsSphere(e));
    CHECK(!a.intersectsSphere(f));
}

TEST_CASE("intersectPlane") {
    Ray a(one3, Vector3(0, 0, 1));
    Vector3 point;

    // parallel plane behind
    auto b = Plane().setFromNormalAndCoplanarPoint(Vector3(0, 0, 1), Vector3(1, 1, -1));
    a.intersectPlane(b, point);
    CHECK(point.isNan());

    // parallel plane coincident with origin
    auto c = Plane().setFromNormalAndCoplanarPoint(Vector3(0, 0, 1), Vector3(1, 1, 0));
    a.intersectPlane(c, point);
    CHECK(point.isNan());

    // parallel plane infront
    auto d = Plane().setFromNormalAndCoplanarPoint(Vector3(0, 0, 1), Vector3(1, 1, 1));
    a.intersectPlane(d, point);
    CHECK(point.equals(a.origin));

    // perpendical ray that overlaps exactly
    auto e = Plane().setFromNormalAndCoplanarPoint(Vector3(1, 0, 0), one3);
    a.intersectPlane(e, point);
    CHECK(point.equals(a.origin));

    // perpendical ray that doesn't overlap
    auto f = Plane().setFromNormalAndCoplanarPoint(Vector3(1, 0, 0), zero3);
    a.intersectPlane(f, point);
    CHECK(point.isNan());
}

TEST_CASE("intersectsPlane") {
    Ray a(one3, Vector3(0, 0, 1));

    // parallel plane in front of the ray
    auto b = Plane().setFromNormalAndCoplanarPoint(Vector3(0, 0, 1), one3.clone().sub(Vector3(0, 0, -1)));
    CHECK(a.intersectsPlane(b));

    // parallel plane coincident with origin
    auto c = Plane().setFromNormalAndCoplanarPoint(Vector3(0, 0, 1), one3.clone().sub(Vector3(0, 0, 0)));
    CHECK(a.intersectsPlane(c));

    // parallel plane behind the ray
    auto d = Plane().setFromNormalAndCoplanarPoint(Vector3(0, 0, 1), one3.clone().sub(Vector3(0, 0, 1)));
    CHECK(!a.intersectsPlane(d));

    // perpendical ray that overlaps exactly
    auto e = Plane().setFromNormalAndCoplanarPoint(Vector3(1, 0, 0), one3);
    CHECK(a.intersectsPlane(e));

    // perpendical ray that doesn't overlap
    auto f = Plane().setFromNormalAndCoplanarPoint(Vector3(1, 0, 0), zero3);
    CHECK(!a.intersectsPlane(f));
}

TEST_CASE("intersectBox") {
    const float TOL = 0.0001f;

    Box3 box(Vector3(-1, -1, -1), Vector3(1, 1, 1));
    Vector3 point;

    Ray a(Vector3(-2, 0, 0), Vector3(1, 0, 0));
    //ray should intersect box at -1,0,0
    CHECK(a.intersectsBox(box));
    a.intersectBox(box, point);
    CHECK(point.distanceTo(Vector3(-1, 0, 0)) < TOL);

    Ray b(Vector3(-2, 0, 0), Vector3(-1, 0, 0));
    //ray is point away from box, it should not intersect
    CHECK(!b.intersectsBox(box));
    b.intersectBox(box, point);
    CHECK(point.isNan());

    Ray c(Vector3(0, 0, 0), Vector3(1, 0, 0));
    // ray is inside box, should return exit point
    CHECK(c.intersectsBox(box));
    c.intersectBox(box, point);
    CHECK(point.distanceTo(Vector3(1, 0, 0)) < TOL);

    Ray d(Vector3(0, 2, 1), Vector3(0, -1, -1).normalize());
    //tilted ray should intersect box at 0,1,0
    CHECK(d.intersectsBox(box));
    d.intersectBox(box, point);
    CHECK(point.distanceTo(Vector3(0, 1, 0)) < TOL);

    Ray e(Vector3(1, -2, 1), Vector3(0, 1, 0).normalize());
    //handle case where ray is coplanar with one of the boxes side - box in front of ray
    CHECK(e.intersectsBox(box));
    e.intersectBox(box, point);
    CHECK(point.distanceTo(Vector3(1, -1, 1)) < TOL);

    Ray f(Vector3(1, -2, 0), Vector3(0, -1, 0).normalize());
    //handle case where ray is coplanar with one of the boxes side - box behind ray
    CHECK(!f.intersectsBox(box));
    f.intersectBox(box, point);
    CHECK(point.isNan());
}

TEST_CASE("intersectTriangle") {
    Ray ray;
    Vector3 a(1, 1, 0);
    Vector3 b(0, 1, 1);
    Vector3 c(1, 0, 1);
    Vector3 point;

    // DdN == 0
    ray.set(ray.origin, zero3);
    ray.intersectTriangle(a, b, c, false, point);
    CHECK(point.isNan());

    // DdN > 0, backfaceCulling = true
    ray.set(ray.origin, one3);
    ray.intersectTriangle(a, b, c, true, point);
    CHECK(point.isNan());

    // DdN > 0
    ray.set(ray.origin, one3);
    ray.intersectTriangle(a, b, c, false, point);
    CHECK(std::abs(point.x - 2.f / 3) <= eps);
    CHECK(std::abs(point.y - 2.f / 3) <= eps);
    CHECK(std::abs(point.z - 2.f / 3) <= eps);

    // DdN > 0, DdQxE2 < 0
    b.multiplyScalar(-1);
    ray.intersectTriangle(a, b, c, false, point);
    CHECK(point.isNan());

    // DdN > 0, DdE1xQ < 0
    a.multiplyScalar(-1);
    ray.intersectTriangle(a, b, c, false, point);
    CHECK(point.isNan());

    // DdN > 0, DdQxE2 + DdE1xQ > DdN
    b.multiplyScalar(-1);
    ray.intersectTriangle(a, b, c, false, point);
    CHECK(point.isNan());

    // DdN < 0, QdN < 0
    a.multiplyScalar(-1);
    b.multiplyScalar(-1);
    ray.direction.multiplyScalar(-1);
    ray.intersectTriangle(a, b, c, false, point);
    CHECK(point.isNan());
}

TEST_CASE("applyMatrix4") {
    Ray a(one3, {0, 0, 1});
    Matrix4 m;

    CHECK(a.clone().applyMatrix4(m).equals(a));

    a = Ray(zero3.clone(), {0, 0, 1});
    m.makeRotationZ(math::PI);
    CHECK(a.clone().applyMatrix4(m).equals(a));

    m.makeRotationX(math::PI);
    auto b = a.clone();
    b.direction.negate();
    auto a2 = a.clone().applyMatrix4(m);
    CHECK(a2.origin.distanceTo(b.origin) < 0.0001);
    CHECK(a2.direction.distanceTo(b.direction) < 0.0001);

    a.origin = Vector3(0, 0, 1);
    b.origin = Vector3(0, 0, -1);
    a2 = a.clone().applyMatrix4(m);
    CHECK(a2.origin.distanceTo(b.origin) < 0.0001);
    CHECK(a2.direction.distanceTo(b.direction) < 0.0001);
}
