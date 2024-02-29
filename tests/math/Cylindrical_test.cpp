
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/Cylindrical.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector3.hpp"

#include <cmath>

using namespace threepp;

namespace {

    constexpr float eps = 0.0001f;

}

TEST_CASE("Instancing") {

    Cylindrical a;
    float radius = 10.0;
    float theta = math::PI;
    float y = 5;

    CHECK(a.radius() == 1.0);
    CHECK(a.theta() == 0);
    CHECK(a.y() == 0);

    a = Cylindrical(radius, theta, y);
    CHECK_THAT(a.radius(), Catch::Matchers::WithinRel(radius));
    CHECK_THAT(a.theta(), Catch::Matchers::WithinRel(theta));
    CHECK_THAT(a.y(), Catch::Matchers::WithinRel(y));
}

TEST_CASE("set") {

    Cylindrical a;
    float radius = 10.0;
    float theta = math::PI;
    float y = 5;

    a.set(radius, theta, y);
    CHECK_THAT(a.radius(), Catch::Matchers::WithinRel(radius));
    CHECK_THAT(a.theta(), Catch::Matchers::WithinRel(theta));
    CHECK_THAT(a.y(), Catch::Matchers::WithinRel(y));
}

TEST_CASE("setFromVector3") {

    Cylindrical a(1, 1, 1);
    Vector3 b(0, 0, 0);
    Vector3 c(3, -1, -3);
    Cylindrical expected(std::sqrt(9.f + 9.f), std::atan2(3.f, -3.f), -1);

    a.setFromVector3(b);
    CHECK(a.radius() == 0);
    CHECK(a.theta() == 0);
    CHECK(a.y() == 0);

    a.setFromVector3(c);
    CHECK(std::abs(a.radius() - expected.radius()) <= eps);
    CHECK(std::abs(a.theta() - expected.theta()) <= eps);
    CHECK(std::abs(a.y() - expected.y()) <= eps);
}
