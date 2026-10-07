// // https://github.com/mrdoob/three.js/blob/r129/src/extras/core/Font.js

#ifndef THREEPP_FONT_HPP
#define THREEPP_FONT_HPP

#include "threepp/math/Color.hpp"
#include "threepp/textures/Image.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    class Shape;

    // Which rows a rasterized string's image covers.
    enum class TextBox {
        // The string's own ink, plus a 2 px margin. The image is as tall as the
        // string's glyphs happen to be: "OTTER" (capitals only) gets a shorter
        // image than "Kayak" (whose y hangs below the baseline), and the
        // baseline sits on a different row in each.
        Ink,
        // The font's line, ascender to descender, whatever the string says (and
        // one line further down per '\n'). Every string gets the same height,
        // with its baseline on the same row. Ink outside the font's own
        // ascender or descender is cut off.
        Line
    };

    struct Font {

        struct Glyph {
            float x_min{};
            float x_max{};
            int ha{};
            std::vector<std::string> o;
        };

        struct BoundingBox {
            float xMin{};
            float xMax{};
            float yMin{};
            float yMax{};
        };

        std::string familyName;
        BoundingBox boundingBox;

        int resolution{};
        int lineHeight{};
        int underlineThickness{};
        int ascender{};  // font units above the baseline (positive)
        int descender{}; // font units below the baseline (negative)

        // Keyed by Unicode code point. A character with no glyph falls back to '?'.
        std::unordered_map<char32_t, Glyph> glyphs;

        // `text` is UTF-8.
        [[nodiscard]] std::vector<Shape> generateShapes(const std::string& text, float size = 100) const;

        // Width of `text` (UTF-8) at `size`, measured exactly as generateShapes
        // lays it out: the sum of glyph advances, widest line if multi-line.
        // For aligning text: this is where the next character would start.
        [[nodiscard]] float advance(const std::string& text, float size = 100) const;

        // Rasterize text into an RGBA image. `pixelHeight` is the height of one
        // line of text, and `box` picks the rows of that line the image keeps:
        //   TextBox::Ink   the line is the font's bounding box, and the image is
        //                  cropped to the string's ink: shorter than pixelHeight
        //                  by an amount that depends on the string.
        //   TextBox::Line  the line is ascender to descender and the image is
        //                  that line: pixelHeight rows for any single-line string,
        //                  the baseline -descender / (ascender - descender) of
        //                  the way up from the bottom.
        // Either way the image is cropped to the ink horizontally (plus 2 px).
        // supersampling (1, 2, 4, 8) renders at N× resolution and box-filters down,
        // giving smoother edges. 4 is a good default for readable text.
        // The returned Image can be used directly as a Texture map.
        [[nodiscard]] Image rasterize(const std::string& text, float pixelHeight,
                                      const Color& color = Color::white,
                                      int supersampling = 4,
                                      TextBox box = TextBox::Ink) const;
    };

}// namespace threepp

#endif//THREEPP_FONT_HPP
