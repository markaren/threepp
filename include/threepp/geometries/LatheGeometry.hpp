// https://github.com/mrdoob/three.js/blob/r129/src/geometries/LatheGeometry.js

#ifndef THREEPP_LATHEGEOMETRY_HPP
#define THREEPP_LATHEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class LatheGeometry: public BufferGeometry {

    public:
        [[nodiscard]] std::string type() const override;

        template<class ArrayLike>
        static std::shared_ptr<LatheGeometry> create(const ArrayLike& points, unsigned int segments = 24, float phiStart = 0, float phiLength = math::TWO_PI) {

            return std::shared_ptr<LatheGeometry>(new LatheGeometry({points.begin(), points.end()}, segments, phiStart, phiLength));
        }

    protected:
        explicit LatheGeometry(const std::vector<Vector2>& points, unsigned int segments = 12, float phiStart = 0, float phiLength = math::TWO_PI);
    };

}// namespace threepp

#endif//THREEPP_LATHEGEOMETRY_HPP
