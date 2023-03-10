// https://github.com/mrdoob/three.js/blob/r150/src/geometries/EdgesGeometry.js

#ifndef THREEPP_EXTRUDEGEOMETRY_HPP
#define THREEPP_EXTRUDEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Shape.hpp"

namespace threepp {

    class ExtrudeGeometry: public BufferGeometry {

    public:
        struct Options {
            unsigned int curveSegments;
            unsigned int steps;
            float depth;
            bool bevelEnabled;
            float bevelThickness;
            float bevelSize;
            float bevelOffset;
            unsigned int bevelSegments;
            Curve3* extrudePath;

            Options(): curveSegments(12), steps(1), depth(1), bevelEnabled(true), bevelThickness(0.2f), bevelSize(bevelThickness - 0.1f), bevelOffset(0), bevelSegments(3), extrudePath(nullptr) {}
        };

        static std::shared_ptr<ExtrudeGeometry> create(Shape& shape, const Options options = {}) {

            return std::shared_ptr<ExtrudeGeometry>(new ExtrudeGeometry({&shape}, options));
        }

        static std::shared_ptr<ExtrudeGeometry> create(const std::vector<std::shared_ptr<Shape>>& shape, const Options options = {}) {

            std::vector<Shape*> ptrs(shape.size());
            std::transform(shape.begin(), shape.end(), ptrs.begin(), [&](auto& s) { return s.get(); });

            return std::shared_ptr<ExtrudeGeometry>(new ExtrudeGeometry(ptrs, options));
        }

    protected:
        ExtrudeGeometry(const std::vector<Shape*>& shapes, const Options& options);
    };

}// namespace threepp

#endif//THREEPP_EXTRUDEGEOMETRY_HPP
