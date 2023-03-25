// // https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/SplineCurve.js

#ifndef THREEPP_SPLINECURVE_HPP
#define THREEPP_SPLINECURVE_HPP

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class SplineCurve: public Curve2 {

    public:
        std::vector<Vector2> points;

        explicit SplineCurve(std::vector<Vector2> points = {});

        void getPoint(float t, Vector2& point) const override;
    };

}// namespace threepp

#endif//THREEPP_SPLINECURVE_HPP
