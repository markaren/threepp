// https://github.com/mrdoob/three.js/blob/r129/src/geometries/SphereGeometry.js

#ifndef THREEPP_SPHEREGEOMETRY_HPP
#define THREEPP_SPHEREGEOMETRY_HPP

#include "threepp/math/MathUtils.hpp"

#include "threepp/core/BufferGeometry.hpp"

#include <memory>

namespace threepp {

    class SphereGeometry : public BufferGeometry {

    public:
        const float radius;

        static std::shared_ptr<SphereGeometry> create(
                float radius = 1,
                int widthSegments = 8,
                int heightSegments = 6,
                float phiStart = 0,
                float phiLength = math::PI * 2,
                float thetaStart = 0,
                float thetaLength = math::PI) {

            return std::shared_ptr<SphereGeometry>(new SphereGeometry(radius, widthSegments, heightSegments, phiStart, phiLength, thetaStart, thetaLength));
        }

    private:
        SphereGeometry(float radius, int widthSegments, int heightSegments, float phiStart, float phiLength, float thetaStart, float thetaLength);
    };

}// namespace threepp

#endif//THREEPP_SPHEREGEOMETRY_HPP
