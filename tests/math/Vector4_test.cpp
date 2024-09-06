
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector4.hpp"

#include <cmath>

using namespace threepp;

namespace {

    constexpr float eps = 0.0001f;

    const float x = 2;
    const float y = 3;
    const float z = 4;
    const float w = 5;

}// namespace

TEST_CASE("applyMatrix4") {
    auto a = Vector4(x, y, z, w);
    auto m = Matrix4().makeRotationX(math::PI);
    auto expected = Vector4(2, -3, -4, 5);

    a.applyMatrix4(m);
    CHECK(std::abs(a.x - expected.x) <= eps);
    CHECK(std::abs(a.y - expected.y) <= eps);
    CHECK(std::abs(a.z - expected.z) <= eps);
    CHECK(std::abs(a.w - expected.w) <= eps);

    a.set(x, y, z, w);
    m.makeTranslation(5, 7, 11);
    expected.set(27, 38, 59, 5);

    a.applyMatrix4(m);
    CHECK(std::abs(a.x - expected.x) <= eps);
    CHECK(std::abs(a.y - expected.y) <= eps);
    CHECK(std::abs(a.z - expected.z) <= eps);
    CHECK(std::abs(a.w - expected.w) <= eps);

    a.set(x, y, z, w);
    m.set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0);
    expected.set(2, 3, 4, 4);

    a.applyMatrix4(m);
    CHECK(std::abs(a.x - expected.x) <= eps);
    CHECK(std::abs(a.y - expected.y) <= eps);
    CHECK(std::abs(a.z - expected.z) <= eps);
    CHECK(std::abs(a.w - expected.w) <= eps);

    a.set(x, y, z, w);
    m.set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53);
    expected.set(68, 224, 442, 664);

    a.applyMatrix4(m);
    CHECK(std::abs(a.x - expected.x) <= eps);
    CHECK(std::abs(a.y - expected.y) <= eps);
    CHECK(std::abs(a.z - expected.z) <= eps);
    CHECK(std::abs(a.w - expected.w) <= eps);
}

TEST_CASE("setFromMatrixPosition") {
    auto a = Vector4();
    auto m = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53);

    a.setFromMatrixPosition(m);
    CHECK(a.x == 7);
    CHECK(a.y == 19);
    CHECK(a.z == 37);
    CHECK(a.w == 53);
}
