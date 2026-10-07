#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/core/Shape.hpp"
#include "threepp/loaders/FontLoader.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
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

TEST_CASE("Font metrics for laying text out") {

    FontLoader loader;
    auto json = loader.load(std::string(DATA_FOLDER) + "/fonts/typeface/optimer_regular.typeface.json");
    REQUIRE(json);
    CHECK(json->ascender == 1267);
    CHECK(json->descender == -374);

    auto ttf = roboto();
    REQUIRE(ttf);
    CHECK(ttf->ascender > 0);
    CHECK(ttf->descender < 0);

    for (const auto* font : {&*json, &*ttf}) {
        const float scale = 50.f / static_cast<float>(font->resolution);
        const float ab = static_cast<float>(font->glyphs.at('a').ha + font->glyphs.at('b').ha) * scale;
        CHECK_THAT(font->advance("ab", 50), Catch::Matchers::WithinRel(ab, 1e-5f));
        // the widest line, and a multi-byte character counts once
        CHECK_THAT(font->advance("ab\na", 50), Catch::Matchers::WithinRel(ab, 1e-5f));
        CHECK(font->advance("", 50) == 0.f);
    }
    CHECK(ttf->advance("\xC2\xB5", 50) < ttf->advance("??", 50));
}

TEST_CASE("A line-box raster has one height and one baseline for every string") {

    FontLoader loader;
    auto json = loader.load(std::string(DATA_FOLDER) + "/fonts/typeface/optimer_regular.typeface.json");
    auto ttf = roboto();
    REQUIRE(json);
    REQUIRE(ttf);

    // The lowest and the highest row holding ink. Row 0 is the image's bottom.
    const auto inkRows = [](const Image& image) {
        const auto& rgba = image.data();
        int lo = -1, hi = -1;
        for (unsigned row = 0; row < image.height(); ++row) {
            for (unsigned col = 0; col < image.width(); ++col) {
                if (rgba[(row * image.width() + col) * 4 + 3] < 128) continue;
                if (lo < 0) lo = static_cast<int>(row);
                hi = static_cast<int>(row);
                break;
            }
        }
        return std::pair{lo, hi};
    };

    const float px = 96;
    for (const auto* font : {&*json, &*ttf}) {
        INFO(font->familyName);
        const auto line = [&](const std::string& text) { return font->rasterize(text, px, Color::white, 4, TextBox::Line); };

        // capitals only, a descender, descenders only, digits: all one height
        for (const char* text : {"OTTER X", "Kayak 2", "gypq", "0.1 m"}) {
            INFO(text);
            CHECK(line(text).height() == 96u);
        }

        // H stands on the baseline, which is where the font's metrics put it,
        // and neither a descender beside it nor the digits move it
        const float baseline = px * static_cast<float>(-font->descender) / static_cast<float>(font->ascender - font->descender);
        const auto [hLo, hHi] = inkRows(line("H"));
        const auto [gLo, gHi] = inkRows(line("Hg"));
        CHECK(std::abs(static_cast<float>(hLo) - baseline) <= 1.f);
        CHECK(hHi == gHi);
        CHECK(gLo < hLo - 5);
        CHECK(std::abs(inkRows(line("1")).first - hLo) <= 1);

        // the default crops to the ink, so a descender makes the image taller
        const auto ink = [&](const std::string& text) { return font->rasterize(text, px, Color::white, 4); };
        CHECK(ink("Hg").height() > ink("H").height() + 5);
    }

    // a second line adds the line advance, and the first line keeps its rows
    // from the top
    const auto one = json->rasterize("H", px, Color::white, 4, TextBox::Line);
    const auto two = json->rasterize("H\nH", px, Color::white, 4, TextBox::Line);
    CHECK(two.height() > one.height() + 48);
    CHECK(two.height() - inkRows(two).second == one.height() - inkRows(one).second);
}
