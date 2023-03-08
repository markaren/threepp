// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/QuadraticBezierCurve3.js

#ifndef THREEPP_QUADRATICBEZIERCURVE3_HPP
#define THREEPP_QUADRATICBEZIERCURVE3_HPP

#include "threepp/extras/core/Curve.hpp"
#include "threepp/extras/core/Interpolations.hpp"

#include <memory>

namespace threepp {

    class QuadraticBezierCurve: public Curve3 {

    public:
        explicit QuadraticBezierCurve(const Vector3& v0 = {}, const Vector3& v1 = {}, const Vector3& v2 = {})
            : v0(v0), v1(v1), v2(v2) {}

        void getPoint(float t, Vector3& point) override {

            point.set(
                    QuadraticBezier(t, v0.x, v1.x, v2.x),
                    QuadraticBezier(t, v0.y, v1.y, v2.y),
                    QuadraticBezier(t, v0.z, v1.z, v2.z));
        }

        static std::shared_ptr<QuadraticBezierCurve> create(const Vector3& v0 = {}, const Vector3& v1 = {}, const Vector3& v2 = {}) {

            return std::make_shared<QuadraticBezierCurve>(v0, v1, v2);
        }

    private:
        Vector3 v0;
        Vector3 v1;
        Vector3 v2;
    };

}// namespace threepp

#endif//THREEPP_QUADRATICBEZIERCURVE3_HPP
