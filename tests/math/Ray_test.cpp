
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/Ray.hpp"

#include <cmath>

using namespace threepp;

namespace {

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
