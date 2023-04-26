
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "threepp/math/Color.hpp"

using namespace threepp;


TEST_CASE("instancing") {

    // rgb ctor
    Color c(1, 1, 1);
    CHECK(c.r == 1);
    CHECK(c.g == 1);
    CHECK(c.b == 1);
}

TEST_CASE("set") {

    Color a;
    Color b(0.5, 0, 0);
    Color c(0xFF0000);
    Color d(0, 1.0, 0);

    a.copy(b);
    CHECK(a.equals(b));

    a.setHex(0xFF0000);
    CHECK(a.equals(c));
}

TEST_CASE("setRGB") {

    Color c;
    c.setRGB(0.3f, 0.5f, 0.7f);
    CHECK(c.r == Approx(0.3f));
    CHECK(c.g == Approx(0.5));
    CHECK(c.b == Approx(0.7));
}

TEST_CASE("setHSL") {

    Color c;
    HSL hsl = {0, 0, 0};
    c.setHSL(0.75f, 1.0f, 0.25f);
    c.getHSL(hsl);

    CHECK(hsl.h == Approx(0.75f));
    CHECK(hsl.s == Approx(1.00f));
    CHECK(hsl.l == Approx(0.25f));
}

TEST_CASE("getHex") {

    Color c = Color::red;
    auto res = c.getHex();
    CHECK(res == 0xFF0000);
}

TEST_CASE("getHexString") {

    Color c =  Color::tomato;
    auto res = c.getHexString();
    CHECK( res == "ff6347");
}
