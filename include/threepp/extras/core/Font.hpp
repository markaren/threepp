// // https://github.com/mrdoob/three.js/blob/r129/src/extras/core/Font.js

#ifndef THREEPP_FONT_HPP
#define THREEPP_FONT_HPP

#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    class Shape;

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

        std::unordered_map<char, Glyph> glyphs;

        [[nodiscard]] std::vector<Shape> generateShapes(const std::string& text, float size = 100) const;
    };

}// namespace threepp

#endif//THREEPP_FONT_HPP
