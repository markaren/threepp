// https://github.com/mrdoob/three.js/blob/r150/src/geometries/ExtrudeGeometry.js

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

            Options();
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<ExtrudeGeometry> create(const std::vector<Shape>& shape, const Options& options = {});

    protected:
        ExtrudeGeometry(const std::vector<Shape>& shapes, const Options& options);
    };

}// namespace threepp

#endif//THREEPP_EXTRUDEGEOMETRY_HPP
