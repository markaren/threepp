
#include "threepp/extras/curves/QuadraticBezierCurve.hpp"

#include "threepp/extras/core/Interpolations.hpp"

using namespace threepp;

QuadraticBezierCurve::QuadraticBezierCurve(const Vector2& v0, const Vector2& v1, const Vector2& v2)
    : v0(v0), v1(v1), v2(v2) {}

void QuadraticBezierCurve::getPoint(float t, Vector2& point) const {

    point.set(
            interpolants::QuadraticBezier(t, v0.x, v1.x, v2.x),
            interpolants::QuadraticBezier(t, v0.y, v1.y, v2.y));
}
