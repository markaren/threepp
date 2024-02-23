
#ifndef THREEPP_TEXTGEOMETRY_HPP
#define THREEPP_TEXTGEOMETRY_HPP

#include "threepp/extras/core/Font.hpp"
#include "threepp/geometries/ShapeGeometry.hpp"

namespace threepp {

    // Specialisation of ShapeGeometry for 2D text
    class TextGeometry: public ShapeGeometry {
    public:
        struct Options {
            Font font;
            float size;
            unsigned int curveSegments;

            explicit Options(Font font, float size = 1, unsigned int curveSegments = 3);
        };

        static std::shared_ptr<TextGeometry> create(const std::string& text, const TextGeometry::Options& opts);

    private:
        TextGeometry(const std::string& text, const TextGeometry::Options& opts);
    };

}// namespace threepp

#endif//THREEPP_TEXTGEOMETRY_HPP
