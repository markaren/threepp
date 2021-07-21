
#ifndef THREEPP_MATHUTILS_HPP
#define THREEPP_MATHUTILS_HPP

#include <cmath>

namespace threepp::math {

    const float LN2 = std::log(2.f);
    const float PI = 2.f * std::acos(0.f);

    const float DEG2RAD = PI / 180.f;
    const float RAD2DEG = 180.f / PI;

    float degToRad(float degrees);

    float radToDeg(float radians);

    bool isPowerOfTwo(int value);

    float ceilPowerOfTwo(float value);

    float floorPowerOfTwo(float value);


}// namespace threepp::math

#endif//THREEPP_MATHUTILS_HPP
