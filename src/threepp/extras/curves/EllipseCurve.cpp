
#include "threepp/extras/curves/EllipseCurve.hpp"

#include <cmath>

using namespace threepp;

EllipseCurve::EllipseCurve(float aX, float aY, float xRadius, float yRadius, float aStartAngle, float aEndAngle, bool aClockwise, float aRotation)
        : aX(aX), aY(aY),
          xRadius(xRadius), yRadius(yRadius),
          aStartAngle(aStartAngle), aEndAngle(aEndAngle),
          aClockwise(aClockwise), aRotation(aRotation) {}

void EllipseCurve::getPoint(float t, Vector2& target) const {

    auto& point = target;

    auto twoPi = math::TWO_PI;
    auto deltaAngle = this->aEndAngle - this->aStartAngle;
    auto samePoints = std::abs(deltaAngle) < std::numeric_limits<float>::epsilon();

    // ensures that deltaAngle is 0 .. 2 PI
    while (deltaAngle < 0) deltaAngle += twoPi;
    while (deltaAngle > twoPi) deltaAngle -= twoPi;

    if (deltaAngle < std::numeric_limits<float>::epsilon()) {

        if (samePoints) {

            deltaAngle = 0;

        } else {

            deltaAngle = twoPi;
        }
    }

    if (this->aClockwise && !samePoints) {

        if (deltaAngle == twoPi) {

            deltaAngle = -twoPi;

        } else {

            deltaAngle = deltaAngle - twoPi;
        }
    }

    auto angle = this->aStartAngle + t * deltaAngle;
    auto x = this->aX + this->xRadius * std::cos(angle);
    auto y = this->aY + this->yRadius * std::sin(angle);

    if (this->aRotation != 0) {

        auto cos = std::cos(this->aRotation);
        auto sin = std::sin(this->aRotation);

        auto tx = x - this->aX;
        auto ty = y - this->aY;

        // Rotate the point about the center of the ellipse.
        x = tx * cos - ty * sin + this->aX;
        y = tx * sin + ty * cos + this->aY;
    }

    point.set(x, y);
}
