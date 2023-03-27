
#include "threepp/extras/curves/CubicBezierCurve.hpp"

#include "threepp/extras/core/Interpolations.hpp"

using namespace threepp;

CubicBezierCurve::CubicBezierCurve(const Vector2& v0, const Vector2& v1, const Vector2& v2, const Vector2& v3)
    : v0(v0), v1(v1), v2(v2), v3(v3) {}

void CubicBezierCurve::getPoint(float t, Vector2& point) const {

    point.set(
            interpolants::CubicBezier(t, v0.x, v1.x, v2.x, v3.x),
            interpolants::CubicBezier(t, v0.y, v1.y, v2.y, v3.y));
}
