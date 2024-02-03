// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TextGeometry.js

#ifndef THREEPP_EXTRUDETEXTGEOMETRY_HPP
#define THREEPP_EXTRUDETEXTGEOMETRY_HPP

#include "threepp/extras/core/Font.hpp"
#include "threepp/extras/core/Shape.hpp"
#include "threepp/geometries/ExtrudeGeometry.hpp"

namespace threepp {

    // Specialisation of ExtrudeGeometry for 3D text
    class ExtrudeTextGeometry: public ExtrudeGeometry {

    public:
        struct Options {
            const Font& font;
            float size;
            float height;
            bool bevelEnabled = false;
            float bevelThickness = 10;
            float bevelSize = 8;

            Options(const Font& font, float size, float height = 50)
                : font(font), size(size), height(height) {}

        private:
            friend ExtrudeTextGeometry;

            [[nodiscard]] ExtrudeGeometry::Options toExtrudeOptions() const {
                ExtrudeGeometry::Options opts{};

                opts.depth = height;

                opts.bevelEnabled = bevelEnabled;
                opts.bevelSize = bevelSize;
                opts.bevelThickness = bevelThickness;

                return opts;
            }
        };

        static std::shared_ptr<ExtrudeTextGeometry> create(const std::string& text, const Options& options) {

            auto shapes = options.font.generateShapes(text, options.size);

            return std::shared_ptr<ExtrudeTextGeometry>(new ExtrudeTextGeometry(shapes, options.toExtrudeOptions()));
        }

    private:
        ExtrudeTextGeometry(const std::vector<Shape>& shapes, const ExtrudeGeometry::Options& options)
            : ExtrudeGeometry(shapes, options) {}
    };

}// namespace threepp

#endif//THREEPP_EXTRUDETEXTGEOMETRY_HPP
