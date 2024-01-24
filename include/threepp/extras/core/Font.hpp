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

        unsigned int resolution{};
        int lineHeight{};
        int underlineThickness{};

        std::unordered_map<char, Glyph> glyphs;

        [[nodiscard]] float fontHeight(unsigned int size) const {
            const auto scale = static_cast<float>(size) / static_cast<float>(resolution);
            return (boundingBox.yMax - boundingBox.yMin + static_cast<float>(underlineThickness)) * scale;
        }

    };


    class Font {

    public:
        explicit Font(FontData data);

        [[nodiscard]] const FontData& data() const;

        [[nodiscard]] std::vector<Shape> generateShapes(const std::string& text, unsigned int size = 100) const;

    private:
        FontData data_;
    };

}// namespace threepp

#endif//THREEPP_FONT_HPP
