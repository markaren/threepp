// https://github.com/mrdoob/three.js/blob/r150/src/geometries/EdgesGeometry.js

#ifndef THREEPP_EXTRUDEGEOMETRY_HPP
#define THREEPP_EXTRUDEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Curve.hpp"


namespace threepp {

    class Shape;

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

        static std::shared_ptr<ExtrudeGeometry> create(Shape& shape, const Options& options = {});

        static std::shared_ptr<ExtrudeGeometry> create(const std::vector<std::shared_ptr<Shape>>& shape, const Options& options = {});

    protected:
        ExtrudeGeometry(const std::vector<Shape*>& shapes, const Options& options);
    };

}// namespace threepp

#endif//THREEPP_EXTRUDEGEOMETRY_HPP
