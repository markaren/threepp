// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TorusGeometry.js

#ifndef THREEPP_TORUSGEOMETRY_HPP
#define THREEPP_TORUSGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class TorusGeometry: public BufferGeometry {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<TorusGeometry> create(
                float radius = 1,
                float tube = 0.4f,
                unsigned int radialSegments = 20,
                unsigned int tubularSegments = 64,
                float arc = math::TWO_PI);

    protected:
        TorusGeometry(float radius, float tube, unsigned int radialSegments, unsigned int tubularSegments, float arc);
    };

}// namespace threepp

#endif//THREEPP_TORUSGEOMETRY_HPP
