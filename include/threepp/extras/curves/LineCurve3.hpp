// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/LineCurve3.js

#ifndef THREEPP_LINECURVE3_HPP
#define THREEPP_LINECURVE3_HPP

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class LineCurve: public Curve3 {

    public:
        LineCurve(const Vector3& v1, const Vector3& v2): v1(v1), v2(v2) {}

        void getPoint(float t, Vector3& point) override {

            if (t == 1) {

                point.copy(this->v2);

            } else {

                point.copy(this->v2).sub(this->v1);
                point.multiplyScalar(t).add(this->v1);
            }
        }

        void getPointAt(float u, Vector3& target) override {

            getPoint(u, target);
        }

        void getTangent(float t, Vector3& tangent) override {

            tangent.copy(this->v2).sub(this->v1).normalize();
        }

    private:
        Vector3 v1;
        Vector3 v2;
    };

}// namespace threepp

#endif//THREEPP_LINECURVE3_HPP
