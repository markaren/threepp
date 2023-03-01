
#ifndef THREEPP_RINGGEOMETRY_HPP
#define THREEPP_RINGGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class RingGeometry: public BufferGeometry {

    public:
        [[nodiscard]] std::string type() const override {

            return "RingGeometry";
        }

        static std::shared_ptr<RingGeometry> create(
                float innerRadius = 0.5f,
                float outerRadius = 1,
                unsigned int thetaSegments = 8,
                unsigned int phiSegments = 1,
                float thetaStart = 0,
                float thetaLength = math::TWO_PI) {

            return std::shared_ptr<RingGeometry>(new RingGeometry(innerRadius, outerRadius, thetaSegments, phiSegments, thetaStart, thetaLength));
        }

    protected:
        RingGeometry(float innerRadius, float outerRadius, unsigned int thetaSegments, unsigned int phiSegments, float thetaStart, float thetaLength);
    };

}// namespace threepp

#endif//THREEPP_RINGGEOMETRY_HPP
