
#include <catch2/catch_test_macros.hpp>

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Spherical.hpp"

#include <cmath>

using namespace threepp;


TEST_CASE("Instancing") {

    auto a = Spherical();
    CHECK(a.radius == 1.f);
    CHECK(a.phi == 0);
    CHECK(a.theta == 0);

    float radius = 10.0f;
    float phi = std::acos(-0.5f);
    float theta = std::sqrt(math::PI) * phi;

    auto b = Spherical(radius, phi, theta);
    CHECK(b.radius == radius);
    CHECK(b.phi == phi);
    CHECK(b.theta == theta);
}

TEST_CASE("set") {

    auto a = Spherical();
    float radius = 10.0f;
    float phi = std::acos(-0.5f);
    float theta = std::sqrt(math::PI) * phi;

    a.set(radius, phi, theta);
    CHECK(a.radius == radius);
    CHECK(a.phi == phi);
    CHECK(a.theta == theta);
}

TEST_CASE("makeSafe") {

    float EPS = 0.000001f;// from source
    float tooLow = 0.0f;
    float tooHigh = math::PI;
    float justRight = 1.5f;
    auto a = Spherical(1, tooLow, 0);

    a.makeSafe();
    CHECK(a.phi == EPS);

    a.set(1, tooHigh, 0);
    a.makeSafe();
    CHECK(a.phi == math::PI - EPS);

    a.set(1, justRight, 0);
    a.makeSafe();
    CHECK(a.phi == justRight);
}

TEST_CASE("setFromVector3") {

    float eps = 0.0001f;

    auto a = Spherical(1, 1, 1);
    auto b = Vector3(0, 0, 0);
    auto c = Vector3(math::PI, 1, -math::PI);
    auto expected = Spherical(4.554032147688322f, 1.3494066171539107f, 2.356194490192345f);

    a.setFromVector3(b);
    CHECK(a.radius == 0);
    CHECK(a.phi == 0);
    CHECK(a.theta == 0);

    a.setFromVector3(c);
    CHECK(std::abs(a.radius - expected.radius) <= eps);
    CHECK(std::abs(a.phi - expected.phi) <= eps);
    CHECK(std::abs(a.theta - expected.theta) <= eps);
}
