
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "threepp/math/MathUtils.hpp"

#include <cmath>
#include <regex>

using namespace threepp;

TEST_CASE("genereteUUID") {

    const auto uuid = math::generateUUID();
    const std::regex uuidRegex("[A-Z0-9]{8}-[A-Z0-9]{4}-4[A-Z0-9]{3}-[A-Z0-9]{4}-[A-Z0-9]{12}", std::regex_constants::icase);

    REQUIRE(std::regex_match(uuid, uuidRegex));
}

TEST_CASE("euclideanModulo") {
    CHECK(std::isnan(math::euclideanModulo(6, 0)));
    CHECK(math::euclideanModulo(6, 1) == Catch::Approx(0.));
    CHECK(math::euclideanModulo(6, 2) == Catch::Approx(0.));
    CHECK(math::euclideanModulo(6, 5) == Catch::Approx(1.));
    CHECK(math::euclideanModulo(6, 6) == Catch::Approx(0.));
    CHECK(math::euclideanModulo(6, 7) == Catch::Approx(6.));
}

TEST_CASE("mapLinear") {
    // Value within range
    CHECK(math::mapLinear(0.5, 0.0, 1.0, 0.0, 10.0) == Catch::Approx(5.0));

    // Value equal to lower boundary
    CHECK(math::mapLinear(0.0, 0.0, 1.0, 0.0, 10.0) == Catch::Approx(0.0));

    // Value equal to upper boundary
    CHECK(math::mapLinear(1.0, 0.0, 1.0, 0.0, 10.0) == Catch::Approx(10.0));
}

TEST_CASE("inverseLerp") {

    // 50% Percentage
    CHECK(math::inverseLerp(1.0, 2.0, 1.5) == Catch::Approx(0.5));
    // 100% Percentage
    CHECK(math::inverseLerp(1.0, 2.0, 2.0) == Catch::Approx(1.));
    // 0% Percentage
    CHECK(math::inverseLerp(1.0, 2.0, 1.0) == Catch::Approx(0.));
    // 0% Percentage, no NaN
    CHECK(math::inverseLerp(1.0, 1.0, 1.0) == Catch::Approx(0.));
}

TEST_CASE("lerp") {
    // Value equal to lower boundary
    CHECK(math::lerp(1.0, 2.0, 0.0) == Catch::Approx(1.0));

    // Value equal to upper boundary
    CHECK(math::lerp(1.0, 2.0, 1.0) == Catch::Approx(2.0));

    // Value within range
    CHECK(math::lerp(1.0, 2.0, 0.4) == Catch::Approx(1.4));
}
