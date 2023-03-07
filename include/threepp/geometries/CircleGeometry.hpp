// https://github.com/mrdoob/three.js/blob/r129/src/geometries/CircleGeometry.js

#ifndef THREEPP_CIRCLEGEOMETRY_HPP
#define THREEPP_CIRCLEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class CircleGeometry: public BufferGeometry {

    public:
        struct Params {
            float radius;
            unsigned int segments;
            float thetaStart;
            float thetaLength;

            explicit Params(float radius = 1, unsigned int segments = 16, float thetaStart = 0, float thetaLength = math::PI)
                : radius(radius), segments(segments), thetaStart(thetaStart), thetaLength(thetaLength) {}
        };

        [[nodiscard]] std::string type() const override {

            return "CircleGeometry";
        }

        static std::shared_ptr<CircleGeometry> create(const Params& params) {

            return std::shared_ptr<CircleGeometry>(new CircleGeometry(params));
        }

        static std::shared_ptr<CircleGeometry> create(
                float radius = 1,
                unsigned int segments = 16,
                float thetaStart = 0,
                float thetaLength = math::TWO_PI) {

            return create(Params(radius, segments, thetaStart, thetaLength));
        }

    protected:
        explicit CircleGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_CIRCLEGEOMETRY_HPP
