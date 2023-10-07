
#include <catch2/catch_test_macros.hpp>

#include "threepp/constants.hpp"

TEST_CASE("constants") {

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
