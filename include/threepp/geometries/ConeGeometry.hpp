// https://github.com/mrdoob/three.js/blob/r129/src/geometries/ConeGeometry.js

#ifndef THREEPP_CONE_GEOMETRY_HPP
#define THREEPP_CONE_GEOMETRY_HPP

#include "threepp/geometries/CylinderGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class ConeGeometry: public CylinderGeometry {

    public:
        struct Params {
            float radius;
            float height;
            unsigned int radialSegments;
            unsigned int heightSegments;
            bool openEnded;
            float thetaStart;
            float thetaLength;

            explicit Params(float radius = 1,
                            float height = 1,
                            unsigned int radialSegments = 16,
                            unsigned int heightSegments = 1,
                            bool openEnded = false,
                            float thetaStart = 0,
                            float thetaLength = math::TWO_PI);
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<ConeGeometry> create(const Params& params);

        static std::shared_ptr<ConeGeometry> create(
                float radius = 1,
                float height = 1,
                unsigned int radialSegments = 16,
                unsigned int heightSegments = 1,
                bool openEnded = false,
                float thetaStart = 0,
                float thetaLength = math::PI * 2);

    protected:
        explicit ConeGeometry(const Params& params);
    };

}// namespace threepp


#endif//THREEPP_CONE_GEOMETRY_HPP
