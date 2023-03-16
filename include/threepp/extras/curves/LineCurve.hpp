// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/LineCurve.js

#ifndef THREEPP_LINECURVE_HPP
#define THREEPP_LINECURVE_HPP

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class LineCurve: public Curve2 {

    public:
        Vector2 v1;
        Vector2 v2;

        LineCurve(const Vector2& v1, const Vector2& v2);

        void getPoint(float t, Vector2& point) override;

        void getPointAt(float u, Vector2& target) override;

        void getTangent(float t, Vector2& tangent) override;
    };

}// namespace threepp

#endif//THREEPP_LINECURVE_HPP
