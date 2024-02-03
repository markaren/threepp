
#include <utility>

#include "threepp/geometries/TextGeometry.hpp"

using namespace threepp;


TextGeometry::TextGeometry(const std::string& text, const Options& options)
    : ShapeGeometry(options.font.generateShapes(text, options.size), options.curveSegments) {}

std::shared_ptr<TextGeometry> TextGeometry::create(const std::string& text, const TextGeometry::Options& opts) {

    return std::shared_ptr<TextGeometry>(new TextGeometry(text, opts));
}

TextGeometry::Options::Options(Font font, float size, unsigned int curveSegments)
    : font(std::move(font)), size(size), curveSegments(curveSegments) {}
