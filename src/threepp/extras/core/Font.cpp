
#include "threepp/extras/core/Font.hpp"
#include "threepp/extras/core/Shape.hpp"
#include "threepp/extras/core/ShapePath.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <algorithm>
#include <limits>

using namespace threepp;

namespace {

    struct FontPath {

        float offsetX{};
        ShapePath path;
    };

    FontPath createPath(char c, float scale, float offsetX, float offsetY, const Font& data) {

        const auto glyph = data.glyphs.contains(c) ? data.glyphs.at(c) : data.glyphs.at('?');

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
    const auto paths = createPaths(text, size, *this);

    for (const auto& path : paths) {

        const auto pathShapes = path.toShapes();
        shapes.insert(shapes.end(), pathShapes.begin(), pathShapes.end());
    }

    return shapes;
}

Image Font::rasterize(const std::string& text, float pixelHeight, const Color& color,
                      int supersampling) const {

    const std::vector<unsigned char> empty(4, 0);
    if (text.empty() || pixelHeight < 1.f) return {empty, 1, 1};

    const int ss = std::max(1, supersampling);

    // Scale 'size' up by ss so we rasterize at ss× the output resolution.
    const float lineSpan = boundingBox.yMax - boundingBox.yMin + static_cast<float>(underlineThickness);
    const float size = (lineSpan > 0.f)
                               ? pixelHeight * static_cast<float>(ss) * static_cast<float>(resolution) / lineSpan
                               : pixelHeight * static_cast<float>(ss);

    const auto shapes = generateShapes(text, size);
    if (shapes.empty()) return {empty, 1, 1};

    struct Poly {
        std::vector<Vector2> outline;
        std::vector<std::vector<Vector2>> holes;
    };
    std::vector<Poly> polys;

    float mnX = std::numeric_limits<float>::max(), mxX = std::numeric_limits<float>::lowest();
    float mnY = std::numeric_limits<float>::max(), mxY = std::numeric_limits<float>::lowest();

    for (const auto& shape : shapes) {
        auto sp = shape.extractPoints(16);
        if (sp.shape.empty()) continue;
        for (const auto& p : sp.shape) {
            mnX = std::min(mnX, p.x);
            mxX = std::max(mxX, p.x);
            mnY = std::min(mnY, p.y);
            mxY = std::max(mxY, p.y);
        }
        for (const auto& hole : sp.holes) {
            for (const auto& p : hole) {
                mnX = std::min(mnX, p.x);
                mxX = std::max(mxX, p.x);
                mnY = std::min(mnY, p.y);
                mxY = std::max(mxY, p.y);
            }
        }
        polys.push_back({std::move(sp.shape), std::move(sp.holes)});
    }

    if (polys.empty() || mxX <= mnX || mxY <= mnY) return {empty, 1, 1};

    // High-res (super-sampled) buffer dimensions.
    const int pad = 2 * ss;
    const int hiW = static_cast<int>(std::ceil(mxX - mnX)) + 2 * pad;
    const int hiH = static_cast<int>(std::ceil(mxY - mnY)) + 2 * pad;

    std::vector<unsigned char> mask(static_cast<size_t>(hiW * hiH), 0);

    // Scanline-fill one polygon ring into `mask` at high resolution.
    // Font y-axis goes up; image rows go down, so y is flipped.
    auto rasterRing = [&](const std::vector<Vector2>& ring, unsigned char val) {
        const int n = static_cast<int>(ring.size());
        if (n < 3) return;
        std::vector<float> xs;
        xs.reserve(16);
        for (int row = 0; row < hiH; row++) {
            xs.clear();
            const float fy = mnY + static_cast<float>(hiH - 1 - row - pad) + 0.5f;
            for (int i = 0; i < n; i++) {
                const Vector2& a = ring[i];
                const Vector2& b = ring[(i + 1) % n];
                if (a.y == b.y) continue;
                const float lo = std::min(a.y, b.y), hi = std::max(a.y, b.y);
                if (fy < lo || fy >= hi) continue;
                const float t = (fy - a.y) / (b.y - a.y);
                xs.push_back(a.x + t * (b.x - a.x));
            }
            std::ranges::sort(xs);
            for (size_t k = 0; k + 1 < xs.size(); k += 2) {
                const int x0 = std::max(0, static_cast<int>(xs[k] - mnX) + pad);
                const int x1 = std::min(hiW - 1, static_cast<int>(xs[k + 1] - mnX) + pad);
                for (int col = x0; col <= x1; col++) {
                    mask[row * hiW + col] = val;
                }
            }
        }
    };

    for (const auto& poly : polys) {
        rasterRing(poly.outline, 255);
        for (const auto& hole : poly.holes) rasterRing(hole, 0);
    }

    // Box-filter ss×ss blocks down to output resolution → anti-aliased alpha.
    const int imgW = hiW / ss;
    const int imgH = hiH / ss;
    const auto ssArea = static_cast<float>(ss * ss);

    const auto cr = static_cast<unsigned char>(color.r * 255.f);
    const auto cg = static_cast<unsigned char>(color.g * 255.f);
    const auto cb = static_cast<unsigned char>(color.b * 255.f);

    std::vector<unsigned char> rgba(static_cast<size_t>(imgW * imgH * 4), 0);
    for (int row = 0; row < imgH; row++) {
        for (int col = 0; col < imgW; col++) {
            int sum = 0;
            for (int dy = 0; dy < ss; dy++)
                for (int dx = 0; dx < ss; dx++)
                    sum += mask[(row * ss + dy) * hiW + (col * ss + dx)];
            const auto alpha = static_cast<unsigned char>(static_cast<float>(sum) / ssArea);
            const int i = (row * imgW + col) * 4;
            rgba[i + 0] = cr;
            rgba[i + 1] = cg;
            rgba[i + 2] = cb;
            rgba[i + 3] = alpha;
        }
    }

    return {rgba, static_cast<unsigned int>(imgW), static_cast<unsigned int>(imgH)};
}
