// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/LineCurve3.js

#ifndef THREEPP_LINECURVE3_HPP
#define THREEPP_LINECURVE3_HPP

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class LineCurve3: public Curve3 {

    public:
        Vector3 v1;
        Vector3 v2;

        LineCurve3(const Vector3& v1, const Vector3& v2);

        void getPoint(float t, Vector3& point) override;

        void getPointAt(float u, Vector3& target) override;

        void getTangent(float t, Vector3& tangent) override;
    };

}// namespace threepp

#endif//THREEPP_LINECURVE3_HPP
