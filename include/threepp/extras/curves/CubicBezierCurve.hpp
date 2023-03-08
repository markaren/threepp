// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/CubicBezierCurve.js

#ifndef THREEPP_CUBICBEZIERCURVE_HPP
#define THREEPP_CUBICBEZIERCURVE_HPP

#include "threepp/extras/core/Curve.hpp"
#include "threepp/extras/core/Interpolations.hpp"

#include <memory>

namespace threepp {

    class CubicBezierCurve: public Curve2 {

    public:
        Vector2 v0;
        Vector2 v1;
        Vector2 v2;
        Vector2 v3;

        explicit CubicBezierCurve(const Vector2& v0 = {}, const Vector2& v1 = {}, const Vector2& v2 = {}, const Vector2& v3 = {})
            : v0(v0), v1(v1), v2(v2), v3(v3) {}

        void getPoint(float t, Vector2& point) override {

            point.set(
                    interpolants::CubicBezier(t, v0.x, v1.x, v2.x, v3.x),
                    interpolants::CubicBezier(t, v0.y, v1.y, v2.y, v3.y));
        }

        static std::shared_ptr<CubicBezierCurve> create(const Vector2& v0 = {}, const Vector2& v1 = {}, const Vector2& v2 = {}, const Vector2& v3 = {}) {

            return std::make_shared<CubicBezierCurve>(v0, v1, v2, v3);
        }
    };

}// namespace threepp

#endif//THREEPP_CUBICBEZIERCURVE_HPP
