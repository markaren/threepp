// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TorusGeometry.js

#ifndef THREEPP_TORUSGEOMETRY_HPP
#define THREEPP_TORUSGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class TorusGeometry: public BufferGeometry {

    public:
        static std::shared_ptr<TorusGeometry> create(
                float radius = 1,
                float tube = 0.4f,
                unsigned int radialSegments = 10,
                unsigned int tubularSegments = 32,
                float arc = math::TWO_PI) {

            return std::shared_ptr<TorusGeometry>(new TorusGeometry(radius, tube, radialSegments, tubularSegments, arc));
        }

    protected:
        TorusGeometry(float radius, float tube, unsigned int radialSegments, unsigned int tubularSegments, float arc);
    };

}

#endif//THREEPP_TORUSGEOMETRY_HPP
