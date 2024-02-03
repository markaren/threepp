
#include "threepp/extras/core/Font.hpp"
#include "threepp/extras/core/ShapePath.hpp"
#include "threepp/utils/StringUtils.hpp"

using namespace threepp;

namespace {

    struct FontPath {

        float offsetX{};
        ShapePath path;
    };

    FontPath createPath(char c, float scale, float offsetX, float offsetY, const Font& data) {

        const auto glyph = data.glyphs.count(c) ? data.glyphs.at(c) : data.glyphs.at('?');

        ShapePath path;

        float x, y, cpx, cpy, cpx1, cpy1, cpx2, cpy2;

        if (!glyph.o.empty()) {

            const auto outline = glyph.o;

            for (unsigned i = 0, l = outline.size(); i < l;) {

                const auto& action = outline[i++];

                if (action == "m") {// moveTo
                    x = utils::parseFloat(outline[i++]) * scale + offsetX;
                    y = utils::parseFloat(outline[i++]) * scale + offsetY;

                    path.moveTo(x, y);
                } else if (action == "l") {// lineTo

                    x = utils::parseFloat(outline[i++]) * scale + offsetX;
                    y = utils::parseFloat(outline[i++]) * scale + offsetY;

                    path.lineTo(x, y);

                } else if (action == "q") {// quadraticCurveTo

                    cpx = utils::parseFloat(outline[i++]) * scale + offsetX;
                    cpy = utils::parseFloat(outline[i++]) * scale + offsetY;
                    cpx1 = utils::parseFloat(outline[i++]) * scale + offsetX;
                    cpy1 = utils::parseFloat(outline[i++]) * scale + offsetY;

                    path.quadraticCurveTo(cpx1, cpy1, cpx, cpy);

                } else if (action == "b") {// bezierCurveTo

                    cpx = utils::parseFloat(outline[i++]) * scale + offsetX;
                    cpy = utils::parseFloat(outline[i++]) * scale + offsetY;
                    cpx1 = utils::parseFloat(outline[i++]) * scale + offsetX;
                    cpy1 = utils::parseFloat(outline[i++]) * scale + offsetY;
                    cpx2 = utils::parseFloat(outline[i++]) * scale + offsetX;
                    cpy2 = utils::parseFloat(outline[i++]) * scale + offsetY;

                    path.bezierCurveTo(cpx1, cpy1, cpx2, cpy2, cpx, cpy);
                }
            }
        }

        return FontPath{static_cast<float>(glyph.ha) * scale, path};
    }

    std::vector<ShapePath> createPaths(const std::string& text, float size, const Font& data) {

        const auto scale = size / static_cast<float>(data.resolution);
        const auto line_height = (data.boundingBox.yMax - data.boundingBox.yMin + static_cast<float>(data.underlineThickness)) * scale;

        std::vector<ShapePath> paths;

        float offsetX = 0, offsetY = 0;

        for (auto c : text) {

            if (c == '\n') {

                offsetX = 0;
                offsetY -= line_height;

            } else {

                const auto ret = createPath(c, scale, offsetX, offsetY, data);
                offsetX += ret.offsetX;
                paths.emplace_back(ret.path);
            }
        }

        return paths;
    }

}// namespace

std::vector<Shape> Font::generateShapes(const std::string& text, float size) const {

    std::vector<Shape> shapes;
    auto paths = createPaths(text, size, *this);

    for (const auto& path : paths) {

        auto pathShapes = path.toShapes();
        for (auto& s : pathShapes) {
            shapes.emplace_back(*s);
        }
    }

    return shapes;
}
