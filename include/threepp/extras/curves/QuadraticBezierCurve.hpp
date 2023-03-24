// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/QuadraticBezierCurve.js

#ifndef THREEPP_QUADRATICBEZIERCURVE_HPP
#define THREEPP_QUADRATICBEZIERCURVE_HPP

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class QuadraticBezierCurve: public Curve2 {

    public:
        Vector2 v0;
        Vector2 v1;
        Vector2 v2;

        explicit QuadraticBezierCurve(const Vector2& v0 = {}, const Vector2& v1 = {}, const Vector2& v2 = {});

        void getPoint(float t, Vector2& point) const override;
    };

}// namespace threepp

#endif//THREEPP_QUADRATICBEZIERCURVE_HPP
