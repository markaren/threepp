
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

    Color c = Color::tomato;
    auto res = c.getHexString();
    CHECK(res == "ff6347");
}

TEST_CASE("getHSL") {

    Color c = 0x80ffff;
    HSL hsl = {0, 0, 0};
    c.getHSL(hsl);

    CHECK(hsl.h == Approx(0.5f));
    CHECK(hsl.s == Approx(1.0f));
    CHECK((std::round(hsl.l * 100) / 100) == Approx(0.75));
}

TEST_CASE("getStyle") {

    Color c = Color::plum;
    auto res = c.getStyle();
    CHECK(res == "rgb(221,160,221)");
}

TEST_CASE("add") {

    Color a = 0x0000FF;
    Color b = 0xFF0000;
    Color c = 0xFF00FF;

    a.add(b);

    CHECK(a.equals(c));
}

TEST_CASE("setStyleRGBed") {

    Color c;
    c.setStyle("rgb(255,0,0)");
    CHECK(c.r == 1);
    CHECK(c.g == 0);
    CHECK(c.b == 0);
}

TEST_CASE("setStyleRGBAed") {

    Color c;
    c.setStyle("rgb(255,0,0,0.5)");
    CHECK(c.r == 1);
    CHECK(c.g == 0);
    CHECK(c.b == 0);
}

TEST_CASE("setStyleRGBedWithSpaces") {

    Color c;
    c.setStyle("rgb( 255, 0, 0)");
    CHECK(c.r == 1);
    CHECK(c.g == 0);
    CHECK(c.b == 0);
}

TEST_CASE("setStyleRGBAedWithSpaces") {

    Color c;
    c.setStyle("rgb( 255, 0, 0, 0.5)");
    CHECK(c.r == 1);
    CHECK(c.g == 0);
    CHECK(c.b == 0);
}

TEST_CASE("setStyleRGBedPercent") {

    Color c;
    c.setStyle("rgb(100%,50%,10%)");
    CHECK(c.r == 1);
    CHECK(c.g == Approx(0.5f));
    CHECK(c.b == Approx(0.1f));
}

TEST_CASE("setStyleRGBedPercentWithSpaces") {

    Color c;
    c.setStyle("rgb( 100%, 50%, 10%)");
    CHECK(c.r == 1);
    CHECK(c.g == Approx(0.5f));
    CHECK(c.b == Approx(0.1f));
}

TEST_CASE("setStyleRGBAedPercentWithSpaces") {

    Color c;
    c.setStyle("rgb( 100%, 50%, 10%, 0.5)");
    CHECK(c.r == 1);
    CHECK(c.g == Approx(0.5f));
    CHECK(c.b == Approx(0.1f));
}

TEST_CASE("setStyleHSLRed") {

    Color c;
    c.setStyle("hsl(360,100%,50%)");
    CHECK(c.r == 1);
    CHECK(c.g == 0);
    CHECK(c.b == Approx(0).margin(1e-4));
}
