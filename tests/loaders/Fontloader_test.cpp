#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/core/Shape.hpp"
#include "threepp/loaders/FontLoader.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

using namespace threepp;

TEST_CASE("Test FontLoader") {

    FontLoader loader;
    auto font = loader.load(std::string(DATA_FOLDER) + "/fonts/typeface/optimer_regular.typeface.json");;

    REQUIRE(font);

    const auto& o = font->glyphs.at('o');

    CHECK(font->familyName == "Optimer");
    CHECK_THAT(font->boundingBox.xMin, Catch::Matchers::WithinRel(-71.));
    CHECK_THAT(font->boundingBox.xMax, Catch::Matchers::WithinRel(1511.));
    CHECK_THAT(font->boundingBox.yMin, Catch::Matchers::WithinRel(-373.75));
    CHECK_THAT(font->boundingBox.yMax, Catch::Matchers::WithinRel(1267.));

    CHECK_THAT(o.x_min, Catch::Matchers::WithinRel(41.));
    CHECK_THAT(o.x_max, Catch::Matchers::WithinRel(710.));
    CHECK(o.ha == 753);
}

namespace {

    std::optional<Font> roboto() {
        return FontLoader().load(std::string(DATA_FOLDER) + "/fonts/truetype/Roboto-Regular.ttf");
    }

}// namespace

TEST_CASE("TTF curves list the end point before the control point") {

    // TrueType implies an on-curve point halfway between two consecutive
    // off-curve points. So wherever one quadratic follows another, the first
    // one's END must sit at the midpoint of the two CONTROL points. The loader
    // used to write the control point where the end belongs, which draws every
    // glyph through its control polygon (too fat, flattened bowls).
    auto font = roboto();
    REQUIRE(font);

    int pairs = 0, implied = 0;
    for (const char32_t c : {U'o', U'e', U'O', U'S', U'@'}) {
        const auto& ops = font->glyphs.at(c).o;
        // collect consecutive q commands as (end, control)
        std::vector<std::array<float, 4>> qs;
        for (size_t i = 0; i < ops.size();) {
            const auto& op = ops[i++];
            const int n = op == "q" ? 4 : op == "b" ? 6 : 2;
            if (op == "q") {
                qs.push_back({std::stof(ops[i]), std::stof(ops[i + 1]), std::stof(ops[i + 2]), std::stof(ops[i + 3])});
            } else {
                qs.push_back({NAN, NAN, NAN, NAN});// breaks a run
            }
            i += n;
        }
        for (size_t k = 0; k + 1 < qs.size(); ++k) {
            const auto& a = qs[k];
            const auto& b = qs[k + 1];
            if (std::isnan(a[0]) || std::isnan(b[0])) continue;
            ++pairs;
            const float mx = (a[2] + b[2]) / 2, my = (a[3] + b[3]) / 2;
            if (std::abs(a[0] - mx) <= 1.f && std::abs(a[1] - my) <= 1.f) ++implied;
        }
    }
    REQUIRE(pairs > 10);
    CHECK(implied * 2 > pairs);
}

TEST_CASE("Text is UTF-8: characters beyond ASCII get their own glyphs") {

    auto font = roboto();
    REQUIRE(font);

    for (const char32_t c : {char32_t{0x00B5}, char32_t{0x00B0}, char32_t{0x00F8}, char32_t{0x03BB}, char32_t{0x2212}}) {
        INFO("U+" << std::hex << static_cast<unsigned>(c));
        CHECK(font->glyphs.contains(c));
    }

    // "µm" is three bytes; it used to become three '?' glyphs.
    const auto micro = font->generateShapes("\xC2\xB5m", 100);
    const auto question = font->generateShapes("??m", 100);
    CHECK(!micro.empty());
    CHECK(micro.size() != question.size());

    // A truncated sequence falls back to '?' instead of throwing.
    CHECK_NOTHROW(font->generateShapes("a\xC3", 100));
    CHECK(font->generateShapes("a\xC3", 100).size() == font->generateShapes("a?", 100).size());
}

TEST_CASE("Typeface JSON keys are decoded as UTF-8") {

    FontLoader loader;
    auto font = loader.load(std::string(DATA_FOLDER) + "/fonts/typeface/optimer_regular.typeface.json");
    REQUIRE(font);
    // The file has 219 glyphs, 124 of them non-ASCII. Keyed by their first BYTE,
    // those collapsed onto a handful of UTF-8 lead bytes and overwrote each other.
    CHECK(font->glyphs.size() == 219);
    CHECK(font->glyphs.contains(char32_t{0x03BB}));// lambda
    CHECK(font->glyphs.contains(char32_t{0x00E9}));// e-acute
}
