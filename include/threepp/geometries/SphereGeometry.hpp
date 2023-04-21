// https://github.com/mrdoob/three.js/blob/r129/src/geometries/SphereGeometry.js

#ifndef THREEPP_SPHEREGEOMETRY_HPP
#define THREEPP_SPHEREGEOMETRY_HPP

#include "threepp/math/MathUtils.hpp"

#include "threepp/core/BufferGeometry.hpp"

#include <memory>

namespace threepp {

    class SphereGeometry: public BufferGeometry {

    public:
        struct Params {
            float radius;
            unsigned int widthSegments;
            unsigned int heightSegments;
            float phiStart;
            float phiLength;
            float thetaStart;
            float thetaLength;

            explicit Params(float radius = 1,
                            unsigned int widthSegments = 16,
                            unsigned int heightSegments = 12,
                            float phiStart = 0,
                            float phiLength = math::TWO_PI,
                            float thetaStart = 0,
                            float thetaLength = math::PI);
        };

        const float radius;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<SphereGeometry> create(const Params& params);

        static std::shared_ptr<SphereGeometry> create(
                float radius = 1,
                unsigned int widthSegments = 16,
                unsigned int heightSegments = 12,
                float phiStart = 0,
                float phiLength = math::TWO_PI,
                float thetaStart = 0,
                float thetaLength = math::PI);

    protected:
        explicit SphereGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_SPHEREGEOMETRY_HPP
