
#include <utility>

#include "threepp/extras/core/Font.hpp"
#include "threepp/extras/core/ShapePath.hpp"

using namespace threepp;

namespace {

    struct FontPath {

        float offsetX;
        ShapePath path;
    };

    FontPath createPath(char c, float scale, float offsetX, float offsetY, const FontData& data) {

        const auto glyph = data.glyphs.count(c) ? data.glyphs.at(c) : data.glyphs.at('?');

        ShapePath path;

        float x, y, cpx, cpy, cpx1, cpy1, cpx2, cpy2;

        if (!glyph.o.empty()) {

            const auto outline = glyph.o;

            for (unsigned i = 0, l = outline.size(); i < l;) {

                const auto& action = outline[i++];

                if (action == "m") {// moveTo
                    x = std::stof(outline[i++]) * scale + offsetX;
                    y = std::stof(outline[i++]) * scale + offsetY;

                    path.moveTo(x, y);
                } else if (action == "l") {// lineTo

                    x = std::stof(outline[i++]) * scale + offsetX;
                    y = std::stof(outline[i++]) * scale + offsetY;

                    path.lineTo(x, y);

                } else if (action == "q") {// quadraticCurveTo

                    cpx = std::stof(outline[i++]) * scale + offsetX;
                    cpy = std::stof(outline[i++]) * scale + offsetY;
                    cpx1 = std::stof(outline[i++]) * scale + offsetX;
                    cpy1 = std::stof(outline[i++]) * scale + offsetY;

                    path.quadraticCurveTo(cpx1, cpy1, cpx, cpy);

                } else if (action == "b") {// bezierCurveTo

                    cpx = std::stof(outline[i++]) * scale + offsetX;
                    cpy = std::stof(outline[i++]) * scale + offsetY;
                    cpx1 = std::stof(outline[i++]) * scale + offsetX;
                    cpy1 = std::stof(outline[i++]) * scale + offsetY;
                    cpx2 = std::stof(outline[i++]) * scale + offsetX;
                    cpy2 = std::stof(outline[i++]) * scale + offsetY;

                    path.bezierCurveTo(cpx1, cpy1, cpx2, cpy2, cpx, cpy);
                }
            }
        }

        return FontPath{static_cast<float>(glyph.ha) * scale, path};
    }

    std::vector<ShapePath> createPaths(const std::string& text, unsigned int size, const FontData& data) {

        const auto scale = static_cast<float>(size) / static_cast<float>(data.resolution);
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

Font::Font(FontData data): data(std::move(data)) {}


std::vector<std::shared_ptr<Shape>> Font::generateShapes(const std::string& text, unsigned int size) {

    std::vector<std::shared_ptr<Shape>> shapes;
    auto paths = createPaths(text, size, data);

    for (const auto& path : paths) {

        auto pathShapes = path.toShapes();
        shapes.insert(shapes.end(), pathShapes.begin(), pathShapes.end());
    }

    return shapes;
}
