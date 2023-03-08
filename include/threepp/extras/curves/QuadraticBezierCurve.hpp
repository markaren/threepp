// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/QuadraticBezierCurve.js

#ifndef THREEPP_QUADRATICBEZIERCURVE_HPP
#define THREEPP_QUADRATICBEZIERCURVE_HPP

#include "threepp/extras/core/Curve.hpp"
#include "threepp/extras/core/Interpolations.hpp"

#include <memory>

namespace threepp {

    class QuadraticBezierCurve: public Curve2 {

    public:
        Vector2 v0;
        Vector2 v1;
        Vector2 v2;

        explicit QuadraticBezierCurve(const Vector2& v0 = {}, const Vector2& v1 = {}, const Vector2& v2 = {})
            : v0(v0), v1(v1), v2(v2) {}

        void getPoint(float t, Vector2& point) override {

            point.set(
                    QuadraticBezier(t, v0.x, v1.x, v2.x),
                    QuadraticBezier(t, v0.y, v1.y, v2.y));
        }

        static std::shared_ptr<QuadraticBezierCurve> create(const Vector2& v0 = {}, const Vector2& v1 = {}, const Vector2& v2 = {}) {

            return std::make_shared<QuadraticBezierCurve>(v0, v1, v2);
        }
    };

}// namespace threepp

#endif//THREEPP_QUADRATICBEZIERCURVE_HPP
