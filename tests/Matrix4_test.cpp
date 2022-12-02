
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;

TEST_CASE("determinant") {

    Matrix4 m1;
    REQUIRE(m1.determinant() == 1);
}

TEST_CASE("set") {
    Matrix4 m;

    m.set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    REQUIRE(m.elements[0] == 0);REQUIRE(m.elements[1] == 4);REQUIRE(m.elements[2] == 8);
    REQUIRE(m.elements[3] == 12);
    REQUIRE(m.elements[4] == 1);
    REQUIRE(m.elements[5] == 5);
    REQUIRE(m.elements[6] == 9);
    REQUIRE(m.elements[7] == 13);
    REQUIRE(m.elements[8] == 2);
    REQUIRE(m.elements[9] == 6);
    REQUIRE(m.elements[10] == 10);
    REQUIRE(m.elements[11] == 14);
    REQUIRE(m.elements[12] == 3);
    REQUIRE(m.elements[13] == 7);
    REQUIRE(m.elements[14] == 11);
    REQUIRE(m.elements[15] == 15);
}

TEST_CASE("identity") {

    Matrix4 m1;
    Matrix4 m2;

    m1.set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    REQUIRE(m1.elements[0] == 0);REQUIRE(m1.elements[1] == 4);REQUIRE(m1.elements[2] == 8);
    REQUIRE(m1.elements[3] == 12);
    REQUIRE(m1.elements[4] == 1);
    REQUIRE(m1.elements[5] == 5);
    REQUIRE(m1.elements[6] == 9);
    REQUIRE(m1.elements[7] == 13);
    REQUIRE(m1.elements[8] == 2);
    REQUIRE(m1.elements[9] == 6);
    REQUIRE(m1.elements[10] == 10);
    REQUIRE(m1.elements[11] == 14);
    REQUIRE(m1.elements[12] == 3);
    REQUIRE(m1.elements[13] == 7);
    REQUIRE(m1.elements[14] == 11);
    REQUIRE(m1.elements[15] == 15);

    REQUIRE(m1 != m2);

    m1.identity();

    REQUIRE(m1 == m2);


}

TEST_CASE("equals") {

    Matrix4 m1;
    Matrix4 m2;

    REQUIRE(m1 == m2);

    m1.makeTranslation(1, 2, 3);

    REQUIRE(m1 != m2);

    m1.makeRotationAxis(Vector3::X, math::DEG2RAD * 90);
    m2.copy(m1);

    REQUIRE(m1 == m2);
}
