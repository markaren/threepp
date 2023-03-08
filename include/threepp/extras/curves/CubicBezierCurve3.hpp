// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/CubicBezierCurve.js

#ifndef THREEPP_CUBICBEZIERCURVE3_HPP
#define THREEPP_CUBICBEZIERCURVE3_HPP

#include "threepp/extras/core/Curve.hpp"
#include "threepp/extras/core/Interpolations.hpp"

#include <memory>

namespace threepp {

    class CubicBezierCurve3: public Curve3 {

    public:
        Vector3 v0;
        Vector3 v1;
        Vector3 v2;
        Vector3 v3;

        explicit CubicBezierCurve3(const Vector3& v0 = {}, const Vector3& v1 = {}, const Vector3& v2 = {}, const Vector3& v3 = {})
            : v0(v0), v1(v1), v2(v2), v3(v3) {}

        void getPoint(float t, Vector3& point) override {

            point.set(
                    interpolants::CubicBezier(t, v0.x, v1.x, v2.x, v3.x),
                    interpolants::CubicBezier(t, v0.y, v1.y, v2.y, v3.y),
                    interpolants::CubicBezier(t, v0.z, v1.z, v2.z, v3.z));
        }

        static std::shared_ptr<CubicBezierCurve3> create(const Vector3& v0 = {}, const Vector3& v1 = {}, const Vector3& v2 = {}, const Vector3& v3 = {}) {

            return std::make_shared<CubicBezierCurve3>(v0, v1, v2, v3);
        }
    };

}// namespace threepp

#endif//THREEPP_CUBICBEZIERCURVE3_HPP
