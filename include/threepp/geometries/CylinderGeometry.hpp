// https://github.com/mrdoob/three.js/blob/r129/src/geometries/CylinderGeometry.js

#ifndef THREEPP_CYLINDERGEOMETRY_HPP
#define THREEPP_CYLINDERGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class CylinderGeometry: public BufferGeometry {

    public:
        const float radiusTop;
        const float radiusBottom;
        const float height;

        [[nodiscard]] std::string type() const override {

            return "CylinderGeometry";
        }

        static std::shared_ptr<CylinderGeometry> create(
                float radiusTop = 1,
                float radiusBottom = 1,
                float height = 1,
                unsigned int radialSegments = 8,
                unsigned int heightSegments = 1,
                bool openEnded = false,
                float thetaStart = 0,
                float thetaLength = math::TWO_PI) {

            return std::shared_ptr<CylinderGeometry>(new CylinderGeometry(radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength));
        }

    protected:
        CylinderGeometry(float radiusTop, float radiusBottom, float height, unsigned int radialSegments, unsigned int heightSegments, bool openEnded, float thetaStart, float thetaLength);
    };

}// namespace threepp

#endif//THREEPP_CYLINDERGEOMETRY_HPP
