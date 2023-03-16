
#include "threepp/extras/curves/LineCurve.hpp"

using namespace threepp;


LineCurve::LineCurve(const Vector2& v1, const Vector2& v2)
    : v1(v1), v2(v2) {}

void LineCurve::getPoint(float t, Vector2& point) {

    if (t == 1) {

        point.copy(this->v2);

    } else {

        point.copy(this->v2).sub(this->v1);
        point.multiplyScalar(t).add(this->v1);
    }
}

void LineCurve::getPointAt(float u, Vector2& target) {

    getPoint(u, target);
}

void LineCurve::getTangent(float t, Vector2& tangent) {

    tangent.copy(this->v2).sub(this->v1).normalize();
}
