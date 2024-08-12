
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/Triangle.hpp"

using namespace threepp;

TEST_CASE("getArea") {
    Triangle a;

    CHECK_THAT( a.getArea(), Catch::Matchers::WithinRel(0.));

    a = Triangle( Vector3( 0, 0, 0 ), Vector3( 1, 0, 0 ), Vector3( 0, 1, 0 ) );
    CHECK_THAT( a.getArea(), Catch::Matchers::WithinRel(0.5));

    a = Triangle( Vector3( 2, 0, 0 ), Vector3( 0, 0, 0 ), Vector3( 0, 0, 2 ) );
    CHECK_THAT( a.getArea(), Catch::Matchers::WithinRel(2.));

    // colinear triangle.
    a = Triangle( Vector3( 2, 0, 0 ), Vector3( 0, 0, 0 ), Vector3( 3, 0, 0 ) );
    CHECK_THAT( a.getArea(), Catch::Matchers::WithinRel(0.));
}

TEST_CASE("getMidpoint") {
    Triangle a;
    Vector3 midpoint;

    a.getMidpoint( midpoint);
    CHECK( midpoint == Vector3( 0, 0, 0 ) );

    a = Triangle( Vector3( 0, 0, 0 ), Vector3( 1, 0, 0 ), Vector3( 0, 1, 0 ) );
    a.getMidpoint( midpoint);
    CHECK( midpoint ==  Vector3( 1.f / 3, 1.f / 3, 0 ) );

    a = Triangle( Vector3( 2, 0, 0 ), Vector3( 0, 0, 0 ), Vector3( 0, 0, 2 ) );
    a.getMidpoint( midpoint);
    CHECK( midpoint == Vector3( 2.f / 3, 0, 2.f / 3 ) );
}
