
#include "threepp/extras/curves/QuadraticBezierCurve3.hpp"

#include "threepp/extras/core/Interpolations.hpp"

using namespace threepp;

QuadraticBezierCurve3::QuadraticBezierCurve3(const Vector3& v0, const Vector3& v1, const Vector3& v2)
    : v0(v0), v1(v1), v2(v2) {}

void QuadraticBezierCurve3::getPoint(float t, Vector3& point) const {

    point.set(
            interpolants::QuadraticBezier(t, v0.x, v1.x, v2.x),
            interpolants::QuadraticBezier(t, v0.y, v1.y, v2.y),
            interpolants::QuadraticBezier(t, v0.z, v1.z, v2.z));
}
