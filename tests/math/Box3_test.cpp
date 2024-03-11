
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Sphere.hpp"
#include "threepp/math/Triangle.hpp"

#include <cmath>

using namespace threepp;

namespace {

    Vector3 negInf3{-Infinity<float>, -Infinity<float>, -Infinity<float>};
    Vector3 posInf3{Infinity<float>, Infinity<float>, Infinity<float>};
    Vector3 zero3{0, 0, 0};
    Vector3 one3{1, 1, 1};
    Vector3 two3{2, 2, 2};

    bool compareBox(const Box3& a, const Box3& b, float threshold = 0.00001f) {

        return (a.min().distanceTo(b.min()) < threshold &&
                a.max().distanceTo(b.max()) < threshold);
    }

}// namespace

TEST_CASE("Instancing") {

    Box3 a;
    CHECK(a.min().equals(posInf3));
    CHECK(a.max().equals(negInf3));

    Box3 b(zero3, zero3);
    CHECK(b.min().equals(zero3));
    CHECK(b.max().equals(zero3));

    Box3 c(zero3, one3);
    CHECK(c.min().equals(zero3));
    CHECK(c.max().equals(one3));
}

TEST_CASE("set") {

    Box3 a;

    a.set(zero3, one3);
    CHECK(a.min().equals(zero3));
    CHECK(a.max().equals(one3));
}

TEST_CASE("setFromArray") {

    Box3 a;

    a.setFromArray(std::vector<float>{0, 0, 0, 1, 1, 1, 2, 2, 2});
    CHECK(a.min().equals(zero3));
    CHECK(a.max().equals(two3));
}

TEST_CASE("setFromBufferAttribute") {

    Box3 a(zero3, one3);
    auto bigger = FloatBufferAttribute::create({-2, -2, -2, 2, 2, 2, 1.5f, 1.5f, 1.5f, 0, 0, 0}, 3);
    auto smaller = FloatBufferAttribute::create({-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 0, 0, 0}, 3);

    Vector3 newMin(-2, -2, -2);
    Vector3 newMax(2, 2, 2);

    bigger->setFromBufferAttribute(a);
    CHECK(a.min().equals(newMin));
    CHECK(a.max().equals(newMax));

    newMin.set(-0.5f, -0.5f, -0.5f);
    newMax.set(0.5f, 0.5f, 0.5f);

    smaller->setFromBufferAttribute(a);
    CHECK(a.min().equals(newMin));
    CHECK(a.max().equals(newMax));
}

TEST_CASE("setFromPoints") {

    Box3 a;

    a.setFromPoints(std::vector<Vector3>{zero3, one3, two3});
    CHECK(a.min().equals(zero3));
    CHECK(a.max().equals(two3));

    a.setFromPoints(std::vector<Vector3>{one3});
    CHECK(a.min().equals(one3));
    CHECK(a.max().equals(one3));

    a.setFromPoints(std::vector<Vector3>{});
    CHECK(a.isEmpty());
}

TEST_CASE("setFromCenterAndSize") {

    Box3 a(zero3, one3);
    Box3 b = a.clone();
    Vector3 centerA;
    Vector3 sizeA;
    Vector3 sizeB;
    Vector3 newCenter = one3;
    Vector3 newSize = two3;

    a.getCenter(centerA);
    a.getSize(sizeA);
    a.setFromCenterAndSize(centerA, sizeA);
    CHECK(a.equals(b));

    a.setFromCenterAndSize(newCenter, sizeA);
    a.getCenter(centerA);
    a.getSize(sizeA);
    b.getSize(sizeB);

    CHECK(centerA.equals(newCenter));
    CHECK(sizeA.equals(sizeB));
    CHECK(!a.equals(b));

    a.setFromCenterAndSize(centerA, newSize);
    a.getCenter(centerA);
    a.getSize(sizeA);
    CHECK(centerA.equals(newCenter));
    CHECK(sizeA.equals(newSize));
    CHECK(!a.equals(b));
}

TEST_CASE("getCenter") {

    Box3 a(zero3, zero3);
    Vector3 center;

    a.getCenter(center);
    CHECK(center.equals(zero3));

    Box3 b(zero3, one3);
    Vector3 midpoint = one3.clone().multiplyScalar(0.5f);
    b.getCenter(center);
    CHECK(center.equals(midpoint));
}

TEST_CASE("getSize") {

    Box3 a(zero3, zero3);
    Vector3 size;

    a.getSize(size);
    CHECK(size.equals(zero3));

    Box3 b(zero3, one3);
    b.getSize(size);
    CHECK(size.equals(one3));
}

TEST_CASE("intersectsBox") {

    Box3 a(zero3, zero3);
    Box3 b(zero3, one3);
    Box3 c(one3.clone().negate(), one3);

    CHECK(a.intersectsBox(a));
    CHECK(a.intersectsBox(b));
    CHECK(a.intersectsBox(c));

    CHECK(b.intersectsBox(a));
    CHECK(c.intersectsBox(a));
    CHECK(b.intersectsBox(c));

    b.translate(Vector3(2, 2, 2));
    CHECK(!a.intersectsBox(b));
    CHECK(!b.intersectsBox(a));
    CHECK(!b.intersectsBox(c));
}

TEST_CASE("intersectSphere") {

    Box3 a(zero3, one3);
    Sphere b(zero3, 1);

    CHECK(a.intersectsSphere(b));

    b.translate(Vector3(2, 2, 2));
    CHECK(!a.intersectsSphere(b));
}

TEST_CASE("intersectPlane") {

    Box3 a(zero3, one3);
    Plane b(Vector3(0, 1, 0), 1);
    Plane c(Vector3(0, 1, 0), 1.25f);
    Plane d(Vector3(0, -1, 0), 1.25f);
    Plane e(Vector3(0, 1, 0), 0.25f);
    Plane f(Vector3(0, 1, 0), -0.25f);
    Plane g(Vector3(0, 1, 0), -0.75f);
    Plane h(Vector3(0, 1, 0), -1);
    Plane i(Vector3(1, 1, 1).normalize(), -1.732f);
    Plane j(Vector3(1, 1, 1).normalize(), -1.733f);

    CHECK(!a.intersectsPlane(b));
    CHECK(!a.intersectsPlane(c));
    CHECK(!a.intersectsPlane(d));
    CHECK(!a.intersectsPlane(e));
    CHECK(a.intersectsPlane(f));
    CHECK(a.intersectsPlane(g));
    CHECK(a.intersectsPlane(h));
    CHECK(a.intersectsPlane(i));
    CHECK(!a.intersectsPlane(j));
}

TEST_CASE("intersectTriangle") {

    Box3 a(one3, two3);
    Triangle b(Vector3(1.5f, 1.5f, 2.5f), Vector3(2.5f, 1.5f, 1.5f), Vector3(1.5f, 2.5f, 1.5f));
    Triangle c(Vector3(1.5f, 1.5, 3.5f), Vector3(3.5f, 1.5f, 1.5f), Vector3(1.5f, 1.5f, 1.5f));
    Triangle d(Vector3(1.5f, 1.75, 3), Vector3(3, 1.75f, 1.5f), Vector3(1.5f, 2.5f, 1.5f));
    Triangle e(Vector3(1.5f, 1.8f, 3), Vector3(3, 1.8f, 1.5f), Vector3(1.5f, 2.5f, 1.5f));
    Triangle f(Vector3(1.5f, 2.5f, 3), Vector3(3, 2.5f, 1.5f), Vector3(1.5f, 2.5f, 1.5f));

    CHECK(a.intersectsTriangle(b));
    CHECK(a.intersectsTriangle(c));
    CHECK(a.intersectsTriangle(d));
    CHECK(!a.intersectsTriangle(e));
    CHECK(!a.intersectsTriangle(f));
}

TEST_CASE("distanceToPoint") {

    Box3 a(zero3, zero3);
    Box3 b(one3.clone().negate(), one3);

    CHECK_THAT(a.distanceToPoint(Vector3(0, 0, 0)), Catch::Matchers::WithinRel(0.f));
    CHECK_THAT(a.distanceToPoint(Vector3(1, 1, 1)), Catch::Matchers::WithinRel(std::sqrt(3.f)));
    CHECK_THAT(a.distanceToPoint(Vector3(-1, -1, -1)), Catch::Matchers::WithinRel(std::sqrt(3.f)));

    CHECK_THAT(b.distanceToPoint(Vector3(2, 2, 2)), Catch::Matchers::WithinRel(std::sqrt(3.f)));
    CHECK_THAT(b.distanceToPoint(Vector3(1, 1, 1)), Catch::Matchers::WithinRel(0.f));
    CHECK_THAT(b.distanceToPoint(Vector3(0, 0, 0)), Catch::Matchers::WithinRel(0.f));
    CHECK_THAT(b.distanceToPoint(Vector3(-1, -1, -1)), Catch::Matchers::WithinRel(0.f));
    CHECK_THAT(b.distanceToPoint(Vector3(-2, -2, -2)), Catch::Matchers::WithinRel(std::sqrt(3.f)));
}

TEST_CASE("getBoundingSphere") {

    Box3 a(zero3, zero3);
    Box3 b(zero3, one3);
    Box3 c(one3.clone().negate(), one3);
    Sphere sphere;

    a.getBoundingSphere(sphere);
    CHECK(sphere.equals(Sphere(zero3, 0)));
    b.getBoundingSphere(sphere);
    CHECK(sphere.equals(Sphere(one3.clone().multiplyScalar(0.5f), std::sqrt(3) * 0.5f)));
    c.getBoundingSphere(sphere);
    CHECK(sphere.equals(Sphere(zero3, std::sqrt(12) * 0.5f)));
}

TEST_CASE("intersect") {

    Box3 a(zero3, zero3);
    Box3 b(zero3, one3);
    Box3 c(one3.clone().negate(), one3);

    CHECK(a.clone().intersect(a) == (a));
    CHECK(a.clone().intersect(b) == (a));
    CHECK(b.clone().intersect(b) == (b));
    CHECK(a.clone().intersect(c) == (a));
    CHECK(b.clone().intersect(c) == (b));
    CHECK(c.clone().intersect(c) == (c));
}

TEST_CASE("union") {

    Box3 a(zero3, zero3);
    Box3 b(zero3, one3);
    Box3 c(one3.clone().negate(), one3);

    CHECK(a.clone().union_(a).equals(a));
    CHECK(a.clone().union_(b).equals(b));
    CHECK(a.clone().union_(c).equals(c));
    CHECK(b.clone().union_(c).equals(c));
}

TEST_CASE("applyMatrix4") {

    Box3 a(zero3, zero3);
    Box3 b(zero3, one3);
    Box3 c(one3.clone().negate(), one3);
    Box3 d(one3.clone().negate(), zero3);

    auto m = Matrix4().makeTranslation(1, -2, 1);
    Vector3 t1 = Vector3(1, -2, 1);

    CHECK(compareBox(a.clone().applyMatrix4(m), a.clone().translate(t1)));
    CHECK(compareBox(b.clone().applyMatrix4(m), b.clone().translate(t1)));
    CHECK(compareBox(c.clone().applyMatrix4(m), c.clone().translate(t1)));
    CHECK(compareBox(d.clone().applyMatrix4(m), d.clone().translate(t1)));
}

TEST_CASE("translate") {

    Box3 a(zero3, zero3);
    Box3 b(zero3, one3);
    Box3 c(one3.clone().negate(), zero3);

    CHECK(a.clone().translate(one3) == (Box3(one3, one3)));
    CHECK(a.clone().translate(one3).translate(one3.clone().negate()) == (a));
    CHECK(c.clone().translate(one3) == (b));
    CHECK(b.clone().translate(one3.clone().negate()) == (c));
}
