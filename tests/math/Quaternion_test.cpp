
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/math/Vector4.hpp"

#include "threepp/math/MathUtils.hpp"

#include <cmath>

using namespace threepp;

namespace {

    constexpr float eps = 0.0001f;

    const float x = 2;
    const float y = 3;
    const float z = 4;
    const float w = 5;

    const std::vector<Euler::RotationOrders> orders{
            Euler::RotationOrders::XYZ,
            Euler::RotationOrders::YXZ,
            Euler::RotationOrders::ZXY,
            Euler::RotationOrders::ZYX,
            Euler::RotationOrders::YZX,
            Euler::RotationOrders::XZY};

    const Euler eulerAngles = Euler(0.1, -0.3, 0.25);

}// namespace

TEST_CASE("instancing") {

    Quaternion a;
    REQUIRE(a.x == 0.f);
    REQUIRE(a.y == 0.f);
    REQUIRE(a.z == 0.f);
    REQUIRE(a.w == 1.f);

    a = Quaternion(x, y, z, w);
    REQUIRE(a.x == x);
    REQUIRE(a.y == y);
    REQUIRE(a.z == z);
    REQUIRE(a.w == w);
}

TEST_CASE("x") {

    Quaternion a;
    REQUIRE(a.x == 0.f);

    a = Quaternion(1, 2, 3);
    REQUIRE(a.x == 1.f);

    a = Quaternion(4, 5, 6, 1);
    REQUIRE(a.x == 4.f);

    a = Quaternion(7, 8, 9);
    a.x = 10;
    REQUIRE(a.x == 10.f);

    a = Quaternion(11, 12, 13);
    bool b = false;
    a._onChange([&b]() {
        b = true;
    });
    REQUIRE(!b);
    a.x = 14;
    REQUIRE(b);
    REQUIRE(a.x == 14.f);
}

TEST_CASE("y") {

    Quaternion a;
    REQUIRE(a.y == 0.f);

    a = Quaternion(1, 2, 3);
    REQUIRE(a.y == 2.f);

    a = Quaternion(4, 5, 6, 1);
    REQUIRE(a.y == 5.f);

    a = Quaternion(7, 8, 9);
    a.y = 10;
    REQUIRE(a.y == 10.f);

    a = Quaternion(11, 12, 13);
    bool b = false;
    a._onChange([&b]() {
        b = true;
    });
    REQUIRE(!b);
    a.y = 14;
    REQUIRE(b);
    REQUIRE(a.y == 14.f);
}

TEST_CASE("z") {

    Quaternion a;
    REQUIRE(a.z == 0.f);

    a = Quaternion(1, 2, 3);
    REQUIRE(a.z == 3.f);

    a = Quaternion(4, 5, 6, 1);
    REQUIRE(a.z == 6.f);

    a = Quaternion(7, 8, 9);
    a.z = 10;
    REQUIRE(a.z == 10.f);

    a = Quaternion(11, 12, 13);
    bool b = false;
    a._onChange([&b]() {
        b = true;
    });
    REQUIRE(!b);
    a.z = 14;
    REQUIRE(b);
    REQUIRE(a.z == 14.f);
}

TEST_CASE("w") {

    Quaternion a;
    REQUIRE(a.w == 1.f);

    a = Quaternion(1, 2, 3);
    REQUIRE(a.w == 1.f);

    a = Quaternion(4, 5, 6, 1);
    REQUIRE(a.w == 1.f);

    a = Quaternion(7, 8, 9);
    a.w = 10;
    REQUIRE(a.w == 10.f);

    a = Quaternion(11, 12, 13);
    bool b = false;
    a._onChange([&b]() {
        b = true;
    });
    REQUIRE(!b);
    a.w = 14;
    REQUIRE(b);
    REQUIRE(a.w == 14.f);
}

TEST_CASE("set") {

    Quaternion a;
    CHECK(a.x == 0.f);
    CHECK(a.y == 0.f);
    CHECK(a.z == 0.f);
    CHECK(a.w == 1.f);

    a.set(x, y, z, w);
    CHECK(a.x == x);
    CHECK(a.y == y);
    CHECK(a.z == z);
    CHECK(a.w == w);
}

TEST_CASE("clone") {

    auto a = Quaternion().clone();
    CHECK(a.x == 0.f);
    CHECK(a.y == 0.f);
    CHECK(a.z == 0.f);
    CHECK(a.w == 1.f);

    auto b = a.set(x, y, z, w).clone();
    CHECK(b.x == x);
    CHECK(b.y == y);
    CHECK(b.z == z);
    CHECK(b.w == w);
}

TEST_CASE("copy") {

    auto a = Quaternion(1, 2, 3, 4);
    auto b = Quaternion().copy(a);
    REQUIRE(b.x == 1.f);
    REQUIRE(b.y == 2.f);
    REQUIRE(b.z == 3.f);
    REQUIRE(b.w == 4.f);

    // ensure that it is a true copy
    a.x = 0;
    a.y = -1;
    a.z = 0;
    a.w = -1;
    REQUIRE(b.x == 1.f);
    REQUIRE(b.y == 2.f);
}

TEST_CASE("setFromEuler/setFromQuaternion") {

    std::vector<Vector3> angles{Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)};

    // ensure euler conversion to/from Quaternion matches.
    for (auto order : orders) {

        for (const auto& angle : angles) {

            auto eulers2 = Euler().setFromQuaternion(Quaternion().setFromEuler(Euler(angle.x, angle.y, angle.z, order)), order);
            auto newAngle = Vector3(eulers2.x, eulers2.y, eulers2.z);
            REQUIRE(newAngle.distanceTo(angle) < 0.001);
        }
    }
}

TEST_CASE("setFromRotationMatrix") {

    // contrived examples purely to please the god of code coverage...
    // match conditions in various 'else [if]' blocks

    auto a = Quaternion();
    auto q = Quaternion(-9, -2, 3, -4).normalize();
    auto m = Matrix4().makeRotationFromQuaternion(q);
    auto expected = Vector4(0.8581163303210332f, 0.19069251784911848f, -0.2860387767736777f, 0.38138503569823695f);

    a.setFromRotationMatrix(m);
    REQUIRE(std::abs(a.x - expected.x) <= eps);
    REQUIRE(std::abs(a.y - expected.y) <= eps);
    REQUIRE(std::abs(a.z - expected.z) <= eps);
    REQUIRE(std::abs(a.w - expected.w) <= eps);

    q = Quaternion(-1, -2, 1, -1).normalize();
    m.makeRotationFromQuaternion(q);
    expected = Vector4(0.37796447300922714f, 0.7559289460184544f, -0.37796447300922714f, 0.37796447300922714f);

    a.setFromRotationMatrix(m);
    REQUIRE(std::abs(a.x - expected.x) <= eps);
    REQUIRE(std::abs(a.y - expected.y) <= eps);
    REQUIRE(std::abs(a.z - expected.z) <= eps);
    REQUIRE(std::abs(a.w - expected.w) <= eps);
}

TEST_CASE("setFromUnitVectors") {

    Quaternion a;
    Vector3 b(1, 0, 0);
    Vector3 c(0, 1, 0);
    auto expected = Quaternion(0, 0, std::sqrt(2) / 2, std::sqrt(2) / 2);

    a.setFromUnitVectors(b, c);
    CHECK(std::abs(a.x - expected.x) <= eps);
    CHECK(std::abs(a.y - expected.y) <= eps);
    CHECK(std::abs(a.z - expected.z) <= eps);
    CHECK(std::abs(a.w - expected.w) <= eps);
}

TEST_CASE("angleTo") {

    Quaternion a;
    Quaternion b = Quaternion().setFromEuler(Euler(0, math::PI, 0));
    Quaternion c = Quaternion().setFromEuler(Euler(0, math::PI * 2, 0));

    CHECK_THAT(a.angleTo(a), Catch::Matchers::WithinRel(0.));
    CHECK_THAT(a.angleTo(b), Catch::Matchers::WithinRel(math::PI));
    CHECK_THAT(a.angleTo(c), Catch::Matchers::WithinRel(0.));
}

TEST_CASE("rotateTowards") {

    auto a = Quaternion();
    auto b = Quaternion().setFromEuler(Euler(0, math::PI, 0));
    auto c = Quaternion();

    float halfPI = math::PI * 0.5f;

    a.rotateTowards(b, 0);
    REQUIRE(a.equals(a) == true);

    a.rotateTowards(b, math::PI * 2.f);// test overshoot
    REQUIRE(a.equals(b) == true);

    a.set(0, 0, 0, 1);
    a.rotateTowards(b, halfPI);
    REQUIRE(a.angleTo(c) - halfPI <= eps);
}

TEST_CASE("identity") {

    auto a = Quaternion();

    a.set(1, 2, 3, -1);
    a.identity();

    REQUIRE(a.x == 0.f);
    REQUIRE(a.y == 0.f);
    REQUIRE(a.z == 0.f);
    REQUIRE(a.w == 1.f);
}

TEST_CASE("invert/conjugate") {

    Quaternion a(x, y, z, w);

    // TODO: add better validation here.

    auto b = a.clone().conjugate();

    CHECK(a.x == -b.x);
    CHECK(a.y == -b.y);
    CHECK(a.z == -b.z);
    CHECK(a.w == b.w);
}

TEST_CASE("dot") {

    Quaternion a;
    Quaternion b;

    CHECK(a.dot(b) == 1);
    a = Quaternion(1, 2, 3, 1);
    b = Quaternion(3, 2, 1, 1);

    CHECK_THAT(a.dot(b), Catch::Matchers::WithinRel(11.));
}

TEST_CASE("normalize/length/lengthSq") {

    Quaternion a(x, y, z, w);

    CHECK(a.length() != 1);
    CHECK(a.lengthSq() != 1);
    a.normalize();
    CHECK(a.length() == 1);
    CHECK(a.lengthSq() == 1);

    a.set(0, 0, 0, 0);
    CHECK(a.lengthSq() == 0);
    CHECK(a.length() == 0);
    a.normalize();
    CHECK(a.lengthSq() == 1);
    CHECK(a.length() == 1);
}

TEST_CASE("premultiply") {

    Quaternion a(x, y, z, w);
    Quaternion b(2 * x, -y, -2 * z, w);
    Quaternion expected(42, -32, -2, 58);

    a.premultiply(b);
    CHECK(std::abs(a.x - expected.x) <= eps);
    CHECK(std::abs(a.y - expected.y) <= eps);
    CHECK(std::abs(a.z - expected.z) <= eps);
    CHECK(std::abs(a.w - expected.w) <= eps);
}

TEST_CASE("slerp") {

    auto a = Quaternion(x, y, z, w);
    auto b = Quaternion(-x, -y, -z, -w);

    auto c = a.clone().slerp(b, 0);
    auto d = a.clone().slerp(b, 1);

    REQUIRE(a.equals(c));
    REQUIRE(b.equals(d));

    float D = std::sqrt(0.5f);

    auto e = Quaternion(1, 0, 0, 0);
    auto f = Quaternion(0, 0, 1, 0);
    auto expected = Quaternion(D, 0, D, 0);
    auto result = e.clone().slerp(f, 0.5);
    REQUIRE(std::abs(result.x - expected.x) <= eps);
    REQUIRE(std::abs(result.y - expected.y) <= eps);
    REQUIRE(std::abs(result.z - expected.z) <= eps);
    REQUIRE(std::abs(result.w - expected.w) <= eps);


    auto g = Quaternion(0, D, 0, D);
    auto h = Quaternion(0, -D, 0, D);
    expected = Quaternion(0, 0, 0, 1);
    result = g.clone().slerp(h, 0.5);

    REQUIRE(std::abs(result.x - expected.x) <= eps);
    REQUIRE(std::abs(result.y - expected.y) <= eps);
    REQUIRE(std::abs(result.z - expected.z) <= eps);
    REQUIRE(std::abs(result.w - expected.w) <= eps);
}

TEST_CASE("slerpQuaternions") {

    float SQRT1_2 = std::sqrt(0.5);

    Quaternion e(1, 0, 0, 0);
    Quaternion f(0, 0, 1, 0);
    Quaternion expected(SQRT1_2, 0, SQRT1_2, 0);

    Quaternion a;
    a.slerpQuaternions(e, f, 0.5);

    CHECK(std::abs(a.x - expected.x) <= eps);
    CHECK(std::abs(a.y - expected.y) <= eps);
    CHECK(std::abs(a.z - expected.z) <= eps);
    CHECK(std::abs(a.w - expected.w) <= eps);
}
