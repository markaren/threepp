// https://github.com/mrdoob/three.js/blob/r129/src/geometries/CircleGeometry.js

#ifndef THREEPP_CIRCLEGEOMETRY_HPP
#define THREEPP_CIRCLEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class CircleGeometry: public BufferGeometry {

    public:
        static std::shared_ptr<CircleGeometry> create(
                float radius = 1,
                unsigned int segments = 8,
                float thetaStart = 0,
                float thetaLength = math::TWO_PI) {

            return std::shared_ptr<CircleGeometry>(new CircleGeometry(radius, segments, thetaStart, thetaLength));
        }

    protected:
        CircleGeometry(float radius, unsigned int segments, float thetaStart, float thetaLength);
    };

}// namespace threepp

#endif//THREEPP_CIRCLEGEOMETRY_HPP
