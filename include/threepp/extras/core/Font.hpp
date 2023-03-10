// // https://github.com/mrdoob/three.js/blob/r129/src/extras/core/Font.js

#ifndef THREEPP_FONT_HPP
#define THREEPP_FONT_HPP

#include "threepp/extras/core/Shape.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    struct FontData {

        struct Glyph {
            float x_min;
            float x_max;
            int ha;
            std::vector<std::string> o;
        };

        struct BoundingBox {
            float xMin;
            float xMax;
            float yMin;
            float yMax;
        };

        std::string familyName;
        BoundingBox boundingBox;

        unsigned int resolution;
        int lineHeight;
        int underlineThickness;

        std::unordered_map<char, Glyph> glyphs;
    };


    class Font {

    public:
        explicit Font(FontData data);

        std::vector<std::shared_ptr<Shape>> generateShapes(const std::string& text, unsigned int size = 100);

    private:
        FontData data;
    };

}// namespace threepp

#endif//THREEPP_FONT_HPP
