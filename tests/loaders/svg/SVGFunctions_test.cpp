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

TEST_CASE("classifyPoint") {

    using threepp::Vector2;
    using Loc = IntersectionLocationType;

    // The edge (0,0) -> (4,0).
    const Vector2 s{0, 0}, e{4, 0};

    CHECK(classifyPoint({0, 0}, s, e).loc == Loc::ORIGIN);
    CHECK(classifyPoint({4, 0}, s, e).loc == Loc::DESTINATION);
    CHECK(classifyPoint({4, 0}, s, e).t == 1);
    CHECK(classifyPoint({2, -1}, s, e).loc == Loc::LEFT);
    CHECK(classifyPoint({2, 1}, s, e).loc == Loc::RIGHT);
    CHECK(classifyPoint({-1, 0}, s, e).loc == Loc::BEHIND);
    CHECK(classifyPoint({5, 0}, s, e).loc == Loc::BEYOND);

    const auto between = classifyPoint({1, 0}, s, e);
    CHECK(between.loc == Loc::BETWEEN);
    CHECK(between.t == 0.25f);
}

TEST_CASE("findEdgeIntersection") {

    using threepp::Vector2;

    const Vector2 a0{0, 0}, a1{4, 0};

    SECTION("crossing edges meet at the first edge's t") {
        const auto hit = findEdgeIntersection(a0, a1, {2, -1}, {2, 1});
        REQUIRE(hit);
        CHECK(hit->x == 2);
        CHECK(hit->y == 0);
        CHECK(hit->t == 0.5f);
    }

    SECTION("parallel edges do not meet") {
        CHECK_FALSE(findEdgeIntersection(a0, a1, {0, 1}, {4, 1}));
    }

    SECTION("a collinear edge that starts at the first edge's origin meets it there") {
        const auto hit = findEdgeIntersection(a0, a1, {0, 0}, {-3, 0});
        REQUIRE(hit);
        CHECK(hit->x == 0);
        CHECK(hit->y == 0);
        CHECK(hit->t == 0);
    }

    SECTION("a collinear edge that starts behind meets the first edge at its other end") {
        const auto hit = findEdgeIntersection(a0, a1, {-2, 0}, {3, 0});
        REQUIRE(hit);
        CHECK(hit->x == 3);
        CHECK(hit->y == 0);
        CHECK(hit->t == 0.75f);
    }

    SECTION("a collinear edge beyond the first edge does not meet it") {
        CHECK_FALSE(findEdgeIntersection(a0, a1, {5, 0}, {7, 0}));
    }
}
