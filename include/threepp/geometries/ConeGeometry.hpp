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

            explicit Params(float radius = 1, float height = 1, unsigned int radialSegments = 16, unsigned int heightSegments = 1, bool openEnded = false, float thetaStart = 0, float thetaLength = math::TWO_PI)
                : radius(radius), height(height), radialSegments(radialSegments), heightSegments(heightSegments), openEnded(openEnded), thetaStart(thetaStart), thetaLength(thetaLength) {}
        };

        [[nodiscard]] std::string type() const override {

            return "ConeGeometry";
        }

        static std::shared_ptr<ConeGeometry> create(const Params& params) {

            return std::shared_ptr<ConeGeometry>(new ConeGeometry(params));
        }

        static std::shared_ptr<ConeGeometry> create(
                float radius = 1,
                float height = 1,
                unsigned int radialSegments = 16,
                unsigned int heightSegments = 1,
                bool openEnded = false,
                float thetaStart = 0,
                float thetaLength = math::PI * 2) {

            return create(Params(radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength));
        }

    protected:
        explicit ConeGeometry(const Params& params)
            : CylinderGeometry(CylinderGeometry::Params(0, params.radius, params.height, params.radialSegments, params.heightSegments, params.openEnded, params.thetaStart, params.thetaLength)) {}
    };

}// namespace threepp


#endif//THREEPP_CONE_GEOMETRY_HPP
