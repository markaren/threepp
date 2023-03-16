// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/QuadraticBezierCurve3.js

#ifndef THREEPP_QUADRATICBEZIERCURVE3_HPP
#define THREEPP_QUADRATICBEZIERCURVE3_HPP

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class QuadraticBezierCurve: public Curve3 {

    public:
        Vector3 v0;
        Vector3 v1;
        Vector3 v2;

        explicit QuadraticBezierCurve(const Vector3& v0 = {}, const Vector3& v1 = {}, const Vector3& v2 = {});

        void getPoint(float t, Vector3& point) override;
    };

}// namespace threepp

#endif//THREEPP_QUADRATICBEZIERCURVE3_HPP
