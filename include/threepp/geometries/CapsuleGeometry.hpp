// https://github.com/mrdoob/three.js/blob/r150/src/geometries/CapsuleGeometry.js
#ifndef THREEPP_CAPSULEGEOMETRY_HPP
#define THREEPP_CAPSULEGEOMETRY_HPP

#include "threepp/geometries/LatheGeometry.hpp"

namespace threepp {

    class CapsuleGeometry: public LatheGeometry {

    public:
        struct Params {
            float radius;
            float length;
            unsigned int capSegments;
            unsigned int radialSegments;

            explicit Params(float radius = 0.5f, float length = 1, unsigned int capSegments = 8, unsigned int radialSegments = 16);
        };

        const float radius;
        const float length;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<CapsuleGeometry> create(const Params& params);

        static std::shared_ptr<CapsuleGeometry> create(
                float radius = 0.5f,
                float length = 1,
                unsigned int capSegments = 8,
                unsigned int radialSegments = 16);

    protected:
        explicit CapsuleGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_CAPSULEGEOMETRY_HPP
