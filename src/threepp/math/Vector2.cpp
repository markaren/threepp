
#include "threepp/math/Vector2.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

float Vector2::angle() const {

    // computes the angle in radians with respect to the positive x-axis

    const auto angle = std::atan2(-this->y, -this->x) + math::PI;

    return angle;
}

float Vector2::angleTo(const Vector2& v) const {

    const auto denominator = std::sqrt(this->lengthSq() * v.lengthSq());

    if (denominator == 0) return math::PI / 2;

    const auto theta = this->dot(v) / denominator;

    // clamp, to handle numerical problems

    return std::acos(std::clamp(theta, -1.f, 1.f));
}

Vector2::operator std::pair<int, int>() const {

    return {static_cast<int>(x), static_cast<int>(y)};
}

Vector2::operator std::pair<float, float>() const {

    return {x, y};
}
