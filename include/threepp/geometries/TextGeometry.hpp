
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
            unsigned int size = 1;
            unsigned int curveSegments = 2;

            explicit Options(Font font);

            Options& setSize(unsigned int size) {
                this->size = size;

                return *this;
            }
        };

        static std::shared_ptr<TextGeometry> create(const std::string& text, const TextGeometry::Options& opts);

    private:
        TextGeometry(const std::string& text, const TextGeometry::Options& opts);
    };

}// namespace threepp

#endif//THREEPP_TEXTGEOMETRY_HPP
