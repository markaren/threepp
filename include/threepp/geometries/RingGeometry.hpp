
#ifndef THREEPP_RINGGEOMETRY_HPP
#define THREEPP_RINGGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class RingGeometry: public BufferGeometry {

    public:
        struct Params {
            float innerRadius;
            float outerRadius;
            unsigned int thetaSegments;
            unsigned int phiSegments;
            float thetaStart;
            float thetaLength;

            explicit Params(float innerRadius = 0.5f,
                            float outerRadius = 1,
                            unsigned int thetaSegments = 16,
                            unsigned int phiSegments = 1,
                            float thetaStart = 0,
                            float thetaLength = math::TWO_PI);
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<RingGeometry> create(const Params& params);

        static std::shared_ptr<RingGeometry> create(
                float innerRadius = 0.5f,
                float outerRadius = 1,
                unsigned int thetaSegments = 16,
                unsigned int phiSegments = 2,
                float thetaStart = 0,
                float thetaLength = math::TWO_PI);

    protected:
        explicit RingGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_RINGGEOMETRY_HPP
