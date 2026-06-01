
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/Color.hpp"

#include <cmath>

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
    CHECK_THAT(c.r, Catch::Matchers::WithinRel(0.3f));
    CHECK_THAT(c.g, Catch::Matchers::WithinRel(0.5f));
    CHECK_THAT(c.b, Catch::Matchers::WithinRel(0.7f));
}

TEST_CASE("setHSL") {

    Color c;
    HSL hsl = {0, 0, 0};
    c.setHSL(0.75f, 1.0f, 0.25f);
    c.getHSL(hsl);

    CHECK_THAT(hsl.h, Catch::Matchers::WithinRel(0.75f));
    CHECK_THAT(hsl.s, Catch::Matchers::WithinRel(1.00f));
    CHECK_THAT(hsl.l, Catch::Matchers::WithinRel(0.25f));
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
    // the hex was authored in sRGB; ask for HSL back in sRGB (ColorManagement
    // stores it linearized internally)
    c.getHSL(hsl, SRGBColorSpace);

    CHECK_THAT(hsl.h, Catch::Matchers::WithinRel(0.5f));
    CHECK_THAT(hsl.s, Catch::Matchers::WithinRel(1.0f));
    CHECK_THAT((std::round(hsl.l * 100) / 100), Catch::Matchers::WithinRel(0.75));
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

TEST_CASE("addColors") {

    Color a = 0x0000FF;
    Color b = 0xFF0000;
    Color c = 0xFF00FF;
    Color d;

    d.addColors(a, b);

    CHECK(d.equals(c));
}

TEST_CASE("multiply") {

    Color a(1, 0, 0.5f);
    Color b(0.5f, 1, 0.5f);
    Color c(0.5f, 0, 0.25f);

    a.multiply(b);
    CHECK(a.equals(c));
}

TEST_CASE("lerp") {

    Color c;
    Color c2;
    c.setRGB(0, 0, 0);
    c.lerp(c2, 0.2f);
    CHECK_THAT(c.r, Catch::Matchers::WithinRel(0.2f));
    CHECK_THAT(c.g, Catch::Matchers::WithinRel(0.2f));
    CHECK_THAT(c.b, Catch::Matchers::WithinRel(0.2f));
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

    // parse straight into the working space so percentages are checked verbatim,
    // without the default sRGB->linear conversion
    Color c;
    c.setStyle("rgb(100%,50%,10%)", LinearSRGBColorSpace);
    CHECK_THAT(c.r, Catch::Matchers::WithinRel(1.f));
    CHECK_THAT(c.g, Catch::Matchers::WithinRel(0.5f));
    CHECK_THAT(c.b, Catch::Matchers::WithinRel(0.1f));
}

TEST_CASE("setStyleRGBedPercentWithSpaces") {

    Color c;
    c.setStyle("rgb( 100%, 50%, 10%)", LinearSRGBColorSpace);
    CHECK_THAT(c.r, Catch::Matchers::WithinRel(1.f));
    CHECK_THAT(c.g, Catch::Matchers::WithinRel(0.5f));
    CHECK_THAT(c.b, Catch::Matchers::WithinRel(0.1f));
}

TEST_CASE("setStyleRGBAedPercentWithSpaces") {

    Color c;
    c.setStyle("rgb( 100%, 50%, 10%, 0.5)", LinearSRGBColorSpace);
    CHECK_THAT(c.r, Catch::Matchers::WithinRel(1.f));
    CHECK_THAT(c.g, Catch::Matchers::WithinRel(0.5f));
    CHECK_THAT(c.b, Catch::Matchers::WithinRel(0.1f));
}

TEST_CASE("setStyleHSLRed") {

    Color c;
    c.setStyle("hsl(360,100%,50%)");
    CHECK_THAT(c.r, Catch::Matchers::WithinRel(1.f));
    CHECK_THAT(c.g, Catch::Matchers::WithinRel(0.f));
    CHECK_THAT(c.b, Catch::Matchers::WithinAbs(0.f, 0.001f));
}

TEST_CASE("setStyleHexSkyBlue") {

    Color c;
    c.setStyle("#87CEEB");
    CHECK(c.getHex() == 0x87CEEB);
}

TEST_CASE("setStyleHexSkyBlueMixed") {

    Color c;
    c.setStyle("#87cEeB");
    CHECK(c.getHex() == 0x87CEEB);
}

TEST_CASE("setStyleHex2Olive") {

    Color c;
    c.setStyle("#F00");
    CHECK(c.getHex() == 0xFF0000);
}

TEST_CASE("setStyleHex2OliveMixed") {

    Color c;
    c.setStyle("#f00");
    CHECK(c.getHex() == 0xFF0000);
}

TEST_CASE("setStyleColorName") {

    Color c;
    c.setStyle("powderblue");
    CHECK(c.getHex() == 0xB0E0E6);
}

TEST_CASE("colorManagement linearizes sRGB input and round-trips") {

    // with management enabled (the default), hex/style input is stored linearized
    Color c(0x808080);
    CHECK(c.r < 0.5f);// linear value of sRGB 0x80 is ~0.216
    CHECK_THAT(c.r, Catch::Matchers::WithinAbs(0.2158f, 1e-3f));

    // getHex converts back to sRGB by default -> original value recovered
    CHECK(c.getHex() == 0x808080);
    CHECK(c.getHexString() == "808080");

    // copySRGBToLinear matches a managed setHex
    Color viaCopy;
    viaCopy.copySRGBToLinear(Color().setRGB(0x80 / 255.f, 0x80 / 255.f, 0x80 / 255.f));
    CHECK_THAT(viaCopy.r, Catch::Matchers::WithinAbs(c.r, 1e-5f));
}

TEST_CASE("colorManagement can be disabled for raw values") {

    ColorManagement::enabled = false;

    Color c(0x808080);
    CHECK_THAT(c.r, Catch::Matchers::WithinAbs(0.5019f, 1e-3f));// raw sRGB byte / 255
    CHECK(c.getHex() == 0x808080);

    ColorManagement::enabled = true;// restore default for subsequent tests
}
