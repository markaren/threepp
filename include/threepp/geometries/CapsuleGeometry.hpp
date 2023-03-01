// https://github.com/mrdoob/three.js/blob/r150/src/geometries/CapsuleGeometry.js
#ifndef THREEPP_CAPSULEGEOMETRY_HPP
#define THREEPP_CAPSULEGEOMETRY_HPP

#include "threepp/geometries/LatheGeometry.hpp"

namespace threepp {

    class CapsuleGeometry: public LatheGeometry {

    public:

        const float radius;
        const float length;

        static std::shared_ptr<CapsuleGeometry> create(float radius = 0.5f, float length = 1, unsigned int capSegments = 8, unsigned int radialSegments = 16) {

            return std::shared_ptr<CapsuleGeometry>(new CapsuleGeometry(radius, length, capSegments, radialSegments));
        }

    protected:
        CapsuleGeometry(float radius, float length, unsigned int capSegments, unsigned int radialSegments);

    };

}

#endif//THREEPP_CAPSULEGEOMETRY_HPP
