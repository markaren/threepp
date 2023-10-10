// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TextGeometry.js

#ifndef THREEPP_TEXTGEOMETRY_HPP
#define THREEPP_TEXTGEOMETRY_HPP

#include "threepp/extras/core/Font.hpp"
#include "threepp/geometries/ExtrudeGeometry.hpp"

namespace threepp {

    class TextGeometry: public ExtrudeGeometry {

    public:
        struct Options {
            const Font& font;
            unsigned int size;
            float height = 50;
            bool bevelEnabled = false;
            float bevelThickness = 10;
            float bevelSize = 8;

            Options(const Font& font, unsigned int size)
                : font(font), size(size) {}

        private:
            friend TextGeometry;

            [[nodiscard]] ExtrudeGeometry::Options toExtrudeOptions() const {
                ExtrudeGeometry::Options opts{};

                opts.depth = height;

                opts.bevelEnabled = bevelEnabled;
                opts.bevelSize = bevelSize;
                opts.bevelThickness = bevelThickness;

                return opts;
            }
        };

        static std::shared_ptr<TextGeometry> create(const std::string& text, const Options& options) {

            auto shapes = options.font.generateShapes(text, options.size);

            return std::shared_ptr<TextGeometry>(new TextGeometry(shapes, options.toExtrudeOptions()));
        }

    private:
        TextGeometry(const std::vector<Shape>& shapes, const ExtrudeGeometry::Options& options)
            : ExtrudeGeometry(shapes, options) {}
    };

}// namespace threepp

#endif//THREEPP_TEXTGEOMETRY_HPP
