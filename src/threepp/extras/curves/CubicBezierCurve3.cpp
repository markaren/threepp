
#include "threepp/extras/curves/CubicBezierCurve3.hpp"

#include "threepp/extras/core/Interpolations.hpp"

using namespace threepp;

CubicBezierCurve3::CubicBezierCurve3(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& v3)
    : v0(v0), v1(v1), v2(v2), v3(v3) {}

void CubicBezierCurve3::getPoint(float t, Vector3& point) const {

    point.set(
            interpolants::CubicBezier(t, v0.x, v1.x, v2.x, v3.x),
            interpolants::CubicBezier(t, v0.y, v1.y, v2.y, v3.y),
            interpolants::CubicBezier(t, v0.z, v1.z, v2.z, v3.z));
}
