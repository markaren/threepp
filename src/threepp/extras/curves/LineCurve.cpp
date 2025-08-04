
#include "threepp/extras/curves/LineCurve.hpp"

using namespace threepp;

template<class T>
LineCurveT<T>::LineCurveT(const T& v1, const T& v2)
    : v1(v1), v2(v2) {}

template<class T>
void LineCurveT<T>::getPoint(float t, T& point) const {

    if (t == 1) {

        point.copy(this->v2);

    } else {

        point.copy(this->v2).sub(this->v1);
        point.multiplyScalar(t).add(this->v1);
    }
}

template<class T>
void LineCurveT<T>::getPointAt(float u, T& target) const {

    getPoint(u, target);
}

template<class T>
void LineCurveT<T>::getTangent(float /*t*/, T& tangent) const {

    tangent.copy(this->v2).sub(this->v1).normalize();
}

template class threepp::LineCurveT<Vector2>;
template class threepp::LineCurveT<Vector3>;
