// https://github.com/mrdoob/three.js/blob/r129/src/geometries/ConeGeometry.js

#ifndef THREEPP_CONE_GEOMETRY_HPP
#define THREEPP_CONE_GEOMETRY_HPP

#include "threepp/geometries/CylinderGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class ConeGeometry : public CylinderGeometry {

    public:
        static std::shared_ptr<ConeGeometry> create(float radius = 1, float height = 1, int radialSegments = 8, int heightSegments = 1, bool openEnded = false, float thetaStart = 0, float thetaLength = math::PI * 2) {
            return std::shared_ptr<ConeGeometry>(new ConeGeometry(radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength));
        }

    protected:
        ConeGeometry(float radius, float height, int radialSegments, int heightSegments, bool openEnded, float thetaStart, float thetaLength)
            : CylinderGeometry(0, radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength) {}
    };

}// namespace threepp


#endif//THREEPP_CONE_GEOMETRY_HPP
