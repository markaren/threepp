
#include <catch2/catch_test_macros.hpp>

#include "threepp/math/Box3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Sphere.hpp"

using namespace threepp;

namespace {

    float eps = 0.0001f;

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

    Sphere a;
    CHECK(a.center.equals(zero3));
    CHECK(a.radius == -1);

    Sphere b(one3, 1);
    CHECK(b.center.equals(one3));
    CHECK(b.radius == 1);
}

TEST_CASE("set") {

    Sphere a;
    CHECK(a.center.equals(zero3));
    CHECK(a.radius == -1);

    a.set(one3, 1);
    CHECK(a.center.equals(one3));
    CHECK(a.radius == 1);
}

TEST_CASE("setFromPoints") {

    Sphere a;
    Vector3 expectedCenter(0.9330126941204071f, 0, 0);
    float expectedRadius = 1.3676668773461689f;
    Vector3 optionalCenter(1, 1, 1);
    std::vector<Vector3> points{
            Vector3(1, 1, 0),
            Vector3(1, 1, 0),
            Vector3(1, 1, 0),
            Vector3(1, 1, 0),
            Vector3(1, 1, 0),
            Vector3(0.8660253882408142f, 0.5f, 0),
            Vector3(-0, 0.5f, 0.8660253882408142f),
            Vector3(1.8660253882408142f, 0.5f, 0),
            Vector3(0, 0.5f, -0.8660253882408142f),
            Vector3(0.8660253882408142f, 0.5f, -0),
            Vector3(0.8660253882408142f, -0.5f, 0),
            Vector3(-0, -0.5f, 0.8660253882408142f),
            Vector3(1.8660253882408142f, -0.5f, 0),
            Vector3(0, -0.5f, -0.8660253882408142f),
            Vector3(0.8660253882408142f, -0.5f, -0),
            Vector3(-0, -1, 0),
            Vector3(-0, -1, 0),
            Vector3(0, -1, 0),
            Vector3(0, -1, -0),
            Vector3(-0, -1, -0),
    };

    a.setFromPoints(points);
    CHECK(std::abs(a.center.x - expectedCenter.x) <= eps);
    CHECK(std::abs(a.center.y - expectedCenter.y) <= eps);
    CHECK(std::abs(a.center.z - expectedCenter.z) <= eps);
    CHECK(std::abs(a.radius - expectedRadius) <= eps);

    expectedRadius = 2.5946195770400102f;
    a.setFromPoints(points, &optionalCenter);
    CHECK(std::abs(a.center.x - optionalCenter.x) <= eps);
    CHECK(std::abs(a.center.y - optionalCenter.y) <= eps);
    CHECK(std::abs(a.center.z - optionalCenter.z) <= eps);
    CHECK(std::abs(a.radius - expectedRadius) <= eps);
}

TEST_CASE("isEmpty") {

    Sphere a;
    CHECK(a.isEmpty());

    a.set(one3, 1);
    CHECK(!a.isEmpty());

    // Negative radius contains no points
    a.set(one3, -1);
    CHECK(a.isEmpty());

    // Zero radius contains only the center point
    a.set(one3, 0);
    CHECK(!a.isEmpty());
}

TEST_CASE("makeEmpty") {

    Sphere a(one3, 1);

    CHECK(!a.isEmpty());

    a.makeEmpty();
    CHECK(a.isEmpty());
    CHECK(a.center.equals(zero3));
}

TEST_CASE("containsPoint") {

    Sphere a(one3, 1);

    CHECK(!a.containsPoint(zero3));
    CHECK(a.containsPoint(one3));

    a.set(zero3, 0);
    CHECK(a.containsPoint(a.center));
}

TEST_CASE("distanceToPoint") {

    Sphere a(one3, 1);

    CHECK((a.distanceToPoint(zero3) - 0.7320f) < 0.001f);
    CHECK(a.distanceToPoint(one3) == -1);
}

TEST_CASE("intersectSphere") {

    Sphere a(one3, 1);
    Sphere b(zero3, 1);
    Sphere c(zero3, 0.25f);

    CHECK(a.intersectsSphere(b));
    CHECK(!a.intersectsSphere(c));
}

TEST_CASE("intersectsBox") {

    Sphere a(zero3, 1);
    Sphere b(Vector3(-5, -5, -5), 1);
    Box3 box(zero3, one3);

    CHECK(a.intersectsBox(box));
    CHECK(!b.intersectsBox(box));
}

TEST_CASE("intersectsPlane") {

    Sphere a(zero3, 1);
    Plane b(Vector3(0, 1, 0), 1);
    Plane c(Vector3(0, 1, 0), 1.25);
    Plane d(Vector3(0, -1, 0), 1.25);

    CHECK(a.intersectsPlane(b));
    CHECK(!a.intersectsPlane(c));
    CHECK(!a.intersectsPlane(d));
}

TEST_CASE("getBoundingBox") {

    Sphere a(one3.clone(), 1);
    Box3 aabb;

    a.getBoundingBox(aabb);
    CHECK(aabb == (Box3(zero3, two3)));

    a.set(zero3, 0);
    a.getBoundingBox(aabb);
    CHECK(aabb == (Box3(zero3, zero3)));

    // Empty sphere produces empty bounding box
    a.makeEmpty();
    a.getBoundingBox(aabb);
    CHECK(aabb.isEmpty());
}

TEST_CASE("applyMatrix4") {

    Sphere a(one3, 1);
    auto m = Matrix4().makeTranslation(1, -2, 1);
    Box3 aabb1;
    Box3 aabb2;

    a.clone().applyMatrix4(m).getBoundingBox(aabb1);
    a.getBoundingBox(aabb2);

    CHECK(aabb1 == aabb2.applyMatrix4(m));
}

TEST_CASE("translate") {

    Sphere a(one3, 1);

    a.translate(one3.clone().negate());
    CHECK(a.center.equals(zero3));
}
