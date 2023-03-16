
#include "threepp/extras/curves/LineCurve3.hpp"

using namespace threepp;


LineCurve3::LineCurve3(const Vector3& v1, const Vector3& v2)
    : v1(v1), v2(v2) {}

void LineCurve3::getPoint(float t, Vector3& point) {

    if (t == 1) {

        point.copy(this->v2);

    } else {

        point.copy(this->v2).sub(this->v1);
        point.multiplyScalar(t).add(this->v1);
    }
}

void LineCurve3::getPointAt(float u, Vector3& target) {

    getPoint(u, target);
}

void LineCurve3::getTangent(float t, Vector3& tangent) {

    tangent.copy(this->v2).sub(this->v1).normalize();
}
