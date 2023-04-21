// https://github.com/mrdoob/three.js/blob/r129/src/geometries/CylinderGeometry.js

#ifndef THREEPP_CYLINDERGEOMETRY_HPP
#define THREEPP_CYLINDERGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class CylinderGeometry: public BufferGeometry {

    public:
        struct Params {
            float radiusTop;
            float radiusBottom;
            float height;
            unsigned int radialSegments;
            unsigned int heightSegments;
            bool openEnded;
            float thetaStart;
            float thetaLength;

            explicit Params(float radiusTop = 1,
                            float radiusBottom = 1,
                            float height = 1,
                            unsigned int radialSegments = 16,
                            unsigned int heightSegments = 1,
                            bool openEnded = false,
                            float thetaStart = 0,
                            float thetaLength = math::TWO_PI);
        };

        const float radiusTop;
        const float radiusBottom;
        const float height;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<CylinderGeometry> create(const Params& params);

        static std::shared_ptr<CylinderGeometry> create(
                float radiusTop = 1,
                float radiusBottom = 1,
                float height = 1,
                unsigned int radialSegments = 16,
                unsigned int heightSegments = 1,
                bool openEnded = false,
                float thetaStart = 0,
                float thetaLength = math::TWO_PI);

    protected:
        explicit CylinderGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_CYLINDERGEOMETRY_HPP
