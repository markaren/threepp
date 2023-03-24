// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/CubicBezierCurve.js

#ifndef THREEPP_CUBICBEZIERCURVE_HPP
#define THREEPP_CUBICBEZIERCURVE_HPP

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class CubicBezierCurve: public Curve2 {

    public:
        Vector2 v0;
        Vector2 v1;
        Vector2 v2;
        Vector2 v3;

        explicit CubicBezierCurve(const Vector2& v0 = {}, const Vector2& v1 = {}, const Vector2& v2 = {}, const Vector2& v3 = {});

        void getPoint(float t, Vector2& point) const override;
    };

}// namespace threepp

#endif//THREEPP_CUBICBEZIERCURVE_HPP
