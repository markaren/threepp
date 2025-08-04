
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include "threepp/math/MathUtils.hpp"

#include "../equals_util.hpp"

#include <catch2/catch_approx.hpp>
#include <cmath>

using namespace threepp;

TEST_CASE("determinant") {

    Matrix4 a;
    REQUIRE_THAT(a.determinant(), Catch::Matchers::WithinRel(1.f));

    a.elements[0] = 2;
    REQUIRE_THAT(a.determinant(), Catch::Matchers::WithinRel(2.f));

    a.elements[0] = 0;
    REQUIRE_THAT(a.determinant(), Catch::Matchers::WithinRel(0.f));

    // calculated via http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
    a.set(2, 3, 4, 5, -1, -21, -3, -4, 6, 7, 8, 10, -8, -9, -10, -12);
    REQUIRE_THAT(a.determinant(), Catch::Matchers::WithinRel(76.f));
}

TEST_CASE("set") {
    Matrix4 m;

    m.set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    REQUIRE(m.elements[0] == 0);
    REQUIRE(m.elements[1] == 4);
    REQUIRE(m.elements[2] == 8);
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

    Matrix4 a;
    Matrix4 b;

    a.set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    REQUIRE(a.elements[0] == 0);
    REQUIRE(a.elements[1] == 4);
    REQUIRE(a.elements[2] == 8);
    REQUIRE(a.elements[3] == 12);
    REQUIRE(a.elements[4] == 1);
    REQUIRE(a.elements[5] == 5);
    REQUIRE(a.elements[6] == 9);
    REQUIRE(a.elements[7] == 13);
    REQUIRE(a.elements[8] == 2);
    REQUIRE(a.elements[9] == 6);
    REQUIRE(a.elements[10] == 10);
    REQUIRE(a.elements[11] == 14);
    REQUIRE(a.elements[12] == 3);
    REQUIRE(a.elements[13] == 7);
    REQUIRE(a.elements[14] == 11);
    REQUIRE(a.elements[15] == 15);

    REQUIRE(!matrixEquals4(a, b));

    a.identity();

    REQUIRE(matrixEquals4(a, b));
}

TEST_CASE("copy") {

    auto a = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    auto b = Matrix4().copy(a);

    REQUIRE(matrixEquals4(a, b));

    // ensure that it is a true copy
    a.elements[0] = 2;
    REQUIRE(!matrixEquals4(a, b));
}

TEST_CASE("setFromMatrix3") {

    auto a = Matrix3().set(
            0, 1, 2,
            3, 4, 5,
            6, 7, 8);
    auto b = Matrix4();
    auto c = Matrix4().set(
            0, 1, 2, 0,
            3, 4, 5, 0,
            6, 7, 8, 0,
            0, 0, 0, 1);
    b.setFromMatrix3(a);
    REQUIRE(b.equals(c));
}

TEST_CASE("copyPosition") {

    auto a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    auto b = Matrix4().set(1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 16);

    REQUIRE(!matrixEquals4(a, b));

    b.copyPosition(a);
    REQUIRE(matrixEquals4(a, b));
}

TEST_CASE("makeRotationFromEuler/extractRotation") {

    std::vector<Euler> testValues{
            Euler(0, 0, 0, Euler::RotationOrders::XYZ),
            Euler(1, 0, 0, Euler::RotationOrders::XYZ),
            Euler(0, 1, 0, Euler::RotationOrders::ZYX),
            Euler(0, 0, 0.5, Euler::RotationOrders::YZX),
            Euler(0, 0, -0.5, Euler::RotationOrders::YZX)};

    for (const auto& v : testValues) {

        auto m = Matrix4().makeRotationFromEuler(v);

        auto v2 = Euler().setFromRotationMatrix(m, v.getOrder());
        auto m2 = Matrix4().makeRotationFromEuler(v2);

        REQUIRE(matrixEquals4(m, m2, eps));
        REQUIRE(eulerEquals(v, v2, eps));

        auto m3 = Matrix4().extractRotation(m2);
        auto v3 = Euler().setFromRotationMatrix(m3, v.getOrder());

        REQUIRE(matrixEquals4(m, m3, eps));
        REQUIRE(eulerEquals(v, v3, eps));
    }
}

TEST_CASE("lookat") {

    Matrix4 a;
    Matrix4 expected = Matrix4().identity();
    Vector3 eye(0, 0, 0);
    Vector3 target(0, 1, -1);
    Vector3 up(0, 1, 0);

    a.lookAt(eye, target, up);
    auto rotation = Euler().setFromRotationMatrix(a);
    REQUIRE_THAT(rotation.x * (180 / math::PI), Catch::Matchers::WithinRel(45.f));

    // eye and target are in the same position
    eye.copy(target);
    a.lookAt(eye, target, up);
    REQUIRE(a == expected);

    // up and z are parallel
    eye.set(0, 1, 0);
    target.set(0, 0, 0);
    a.lookAt(eye, target, up);
    expected.set(
            1, 0, 0, 0,
            0, 0.0001, 1, 0,
            0, -1, 0.0001, 0,
            0, 0, 0, 1);
    REQUIRE(a == expected);
}

TEST_CASE("premultiply") {

    auto lhs = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53);
    auto rhs = Matrix4().set(59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131);

    rhs.premultiply(lhs);

    CHECK_THAT(rhs.elements[0], Catch::Matchers::WithinRel(1585.f));
    CHECK_THAT(rhs.elements[1], Catch::Matchers::WithinRel(5318.f));
    CHECK_THAT(rhs.elements[2], Catch::Matchers::WithinRel(10514.f));
    CHECK_THAT(rhs.elements[3], Catch::Matchers::WithinRel(15894.f));
    CHECK_THAT(rhs.elements[4], Catch::Matchers::WithinRel(1655.f));
    CHECK_THAT(rhs.elements[5], Catch::Matchers::WithinRel(5562.f));
    CHECK_THAT(rhs.elements[6], Catch::Matchers::WithinRel(11006.f));
    CHECK_THAT(rhs.elements[7], Catch::Matchers::WithinRel(16634.f));
    CHECK_THAT(rhs.elements[8], Catch::Matchers::WithinRel(1787.f));
    CHECK_THAT(rhs.elements[9], Catch::Matchers::WithinRel(5980.f));
    CHECK_THAT(rhs.elements[10], Catch::Matchers::WithinRel(11840.f));
    CHECK_THAT(rhs.elements[11], Catch::Matchers::WithinRel(17888.f));
    CHECK_THAT(rhs.elements[12], Catch::Matchers::WithinRel(1861.f));
    CHECK_THAT(rhs.elements[13], Catch::Matchers::WithinRel(6246.f));
    CHECK_THAT(rhs.elements[14], Catch::Matchers::WithinRel(12378.f));
    CHECK_THAT(rhs.elements[15], Catch::Matchers::WithinRel(18710.f));
}

TEST_CASE("transpose") {

    Matrix4 a;
    Matrix4 b = Matrix4(a).transpose();
    REQUIRE(matrixEquals4(a, b));

    b = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    Matrix4 c = Matrix4(b).transpose();
    REQUIRE(!matrixEquals4(b, c));
    c.transpose();
    REQUIRE(matrixEquals4(b, c));
}

TEST_CASE("multipyMatrices") {

    auto lhs = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53);
    auto rhs = Matrix4().set(59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131);
    auto ans = Matrix4();

    ans.multiplyMatrices(lhs, rhs);

    CHECK_THAT(ans.elements[0], Catch::Matchers::WithinRel(1585.f));
    CHECK_THAT(ans.elements[1], Catch::Matchers::WithinRel(5318.f));
    CHECK_THAT(ans.elements[2], Catch::Matchers::WithinRel(10514.f));
    CHECK_THAT(ans.elements[3], Catch::Matchers::WithinRel(15894.f));
    CHECK_THAT(ans.elements[4], Catch::Matchers::WithinRel(1655.f));
    CHECK_THAT(ans.elements[5], Catch::Matchers::WithinRel(5562.f));
    CHECK_THAT(ans.elements[6], Catch::Matchers::WithinRel(11006.f));
    CHECK_THAT(ans.elements[7], Catch::Matchers::WithinRel(16634.f));
    CHECK_THAT(ans.elements[8], Catch::Matchers::WithinRel(1787.f));
    CHECK_THAT(ans.elements[9], Catch::Matchers::WithinRel(5980.f));
    CHECK_THAT(ans.elements[10], Catch::Matchers::WithinRel(11840.f));
    CHECK_THAT(ans.elements[11], Catch::Matchers::WithinRel(17888.f));
    CHECK_THAT(ans.elements[12], Catch::Matchers::WithinRel(1861.f));
    CHECK_THAT(ans.elements[13], Catch::Matchers::WithinRel(6246.f));
    CHECK_THAT(ans.elements[14], Catch::Matchers::WithinRel(12378.f));
    CHECK_THAT(ans.elements[15], Catch::Matchers::WithinRel(18710.f));
}

TEST_CASE("invert") {

    auto zero = Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    auto identity = Matrix4();

    auto a = Matrix4();
    auto b = Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    a.copy(b).invert();
    REQUIRE(matrixEquals4(a, zero));

    std::vector<Matrix4> testMatrices = {
            Matrix4().makeRotationX(0.3),
            Matrix4().makeRotationX(-0.3),
            Matrix4().makeRotationY(0.3),
            Matrix4().makeRotationY(-0.3),
            Matrix4().makeRotationZ(0.3),
            Matrix4().makeRotationZ(-0.3),
            Matrix4().makeScale(1, 2, 3),
            Matrix4().makeScale(1 / 8., 1 / 2., 1 / 3.),
            Matrix4().makePerspective(-1, 1, 1, -1, 1, 1000),
            Matrix4().makePerspective(-16, 16, 9, -9, 0.1, 10000),
            Matrix4().makeTranslation(1, 2, 3)};

    for (const auto& m : testMatrices) {

        auto mInverse = Matrix4().copy(m).invert();
        auto mSelfInverse = Matrix4(m);
        mSelfInverse.copy(mSelfInverse).invert();

        // self-inverse should the same as inverse
        REQUIRE(matrixEquals4(mSelfInverse, mInverse));

        // the determinant of the inverse should be the reciprocal
        REQUIRE(std::abs((m.determinant() * mInverse.determinant()) - 1) < 0.0001);

        auto mProduct = Matrix4().multiplyMatrices(m, mInverse);

        // the determinant of the identity matrix is 1
        REQUIRE(std::abs(mProduct.determinant() - 1) < 0.0001);
        REQUIRE(matrixEquals4(mProduct, identity));
    }
}

TEST_CASE("scale") {

    auto a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    auto b = Vector3(2, 3, 4);
    auto c = Matrix4().set(2, 6, 12, 4, 10, 18, 28, 8, 18, 30, 44, 12, 26, 42, 60, 16);

    a.scale(b);
    REQUIRE(matrixEquals4(a, c));
}

TEST_CASE("getMaxScaleOnAxis") {

    auto a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    auto expected = std::sqrt(3 * 3 + 7 * 7 + 11 * 11);

    REQUIRE(std::abs(a.getMaxScaleOnAxis() - expected) <= eps);
}

TEST_CASE("makeScale") {

    auto a = Matrix4();
    auto c = Matrix4().set(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1);

    a.makeScale(2, 3, 4);
    REQUIRE(matrixEquals4(a, c));
}

TEST_CASE("makeShear") {

    auto a = Matrix4();
    auto c = Matrix4().set(1, 3, 5, 0, 1, 1, 6, 0, 2, 4, 1, 0, 0, 0, 0, 1);

    a.makeShear(1, 2, 3, 4, 5, 6);
    REQUIRE(matrixEquals4(a, c));
}

TEST_CASE("makePerspective") {

    auto a = Matrix4().makePerspective(-1, 1, -1, 1, 1, 100);
    auto expected = Matrix4().set(
            1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -101 / 99., -200 / 99.,
            0, 0, -1, 0);
    REQUIRE(matrixEquals4(a, expected));
}

TEST_CASE("equals") {

    auto a = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    auto b = Matrix4().set(0, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    REQUIRE(!a.equals(b));
    REQUIRE(!b.equals(a));

    a.copy(b);
    REQUIRE(a.equals(b));
    REQUIRE(b.equals(a));
}

TEST_CASE("conversions") {

    Matrix4 m;
    m.setPosition(1, 2, 3);

    std::array<float, 16> array = m;
    CHECK(array[12] == Catch::Approx(1.0));
    CHECK(array[13] == Catch::Approx(2.0));
    CHECK(array[14] == Catch::Approx(3.0));

    Vector3 v;
    v.setFromMatrixPosition(array);
    CHECK(v.x == Catch::Approx(1.0));
    CHECK(v.y == Catch::Approx(2.0));
    CHECK(v.z == Catch::Approx(3.0));
}
