// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/CubicBezierCurve.js

#ifndef THREEPP_CUBICBEZIERCURVE3_HPP
#define THREEPP_CUBICBEZIERCURVE3_HPP

#include "threepp/extras/core/Curve.hpp"


namespace threepp {

    class CubicBezierCurve3: public Curve3 {

    public:
        Vector3 v0;
        Vector3 v1;
        Vector3 v2;
        Vector3 v3;

        explicit CubicBezierCurve3(const Vector3& v0 = {}, const Vector3& v1 = {}, const Vector3& v2 = {}, const Vector3& v3 = {});

        void getPoint(float t, Vector3& point) const override;
    };

}// namespace threepp

#endif//THREEPP_CUBICBEZIERCURVE3_HPP
