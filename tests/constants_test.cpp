
#include <catch2/catch_test_macros.hpp>

#include "threepp/constants.hpp"

TEST_CASE("constants") {

    REQUIRE(threepp::NeverDepth == 0);
    REQUIRE(threepp::AlwaysDepth == 1);
    REQUIRE(threepp::LessDepth == 2);
    REQUIRE(threepp::LessEqualDepth == 3);
    REQUIRE(threepp::EqualDepth == 4);
    REQUIRE(threepp::GreaterEqualDepth == 5);
    REQUIRE(threepp::GreaterDepth == 6);
    REQUIRE(threepp::NotEqualDepth == 7);

    REQUIRE(threepp::UVMapping == 300);
    REQUIRE(threepp::CubeReflectionMapping == 301);
    REQUIRE(threepp::CubeRefractionMapping == 302);
    REQUIRE(threepp::EquirectangularReflectionMapping == 303);
    REQUIRE(threepp::EquirectangularRefractionMapping == 304);
    REQUIRE(threepp::CubeUVReflectionMapping == 306);

    REQUIRE(threepp::InterpolateDiscrete == 2300);
    REQUIRE(threepp::InterpolateLinear == 2301);
    REQUIRE(threepp::InterpolateSmooth == 2302);
    REQUIRE(threepp::ZeroCurvatureEnding == 2400);
    REQUIRE(threepp::ZeroSlopeEnding == 2401);
    REQUIRE(threepp::WrapAroundEnding == 2402);
    REQUIRE(threepp::TrianglesDrawMode == 0);
    REQUIRE(threepp::TriangleStripDrawMode == 1);
    REQUIRE(threepp::TriangleFanDrawMode == 2);

}
