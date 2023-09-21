#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/svg/SVGFunctions.hpp"

using namespace threepp::svg;

TEST_CASE("parseFloats test") {

    std::string input{"200,200"};
    CHECK(parseFloats(input) == std::vector<float>{200, 200});

    input = "-122.304 84.285";
    CHECK(parseFloats(input) == std::vector<float>{-122.304f, 84.285f});

    input = "-122.304 84.285 -122.203 86.179 -123.027 86.16";
    CHECK(parseFloats(input) == std::vector<float>{
                                        -122.304f,
                                        84.285f,
                                        -122.203f,
                                        86.179f,
                                        -123.027f,
                                        86.16f});

    input = "-109.009,110.072 -107.701 111.446 -108.34 111.967";
    CHECK(parseFloats(input) == std::vector<float>{
                                        -109.009f,
                                        110.072f,
                                        -107.701f,
                                        111.446f,
                                        -108.34f,
                                        111.967f});
}
