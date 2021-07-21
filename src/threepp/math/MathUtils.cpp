
#include "threepp/math/MathUtils.hpp"

float threepp::math::degToRad(const float degrees) {

    return degrees * DEG2RAD;
}

float threepp::math::radToDeg(const float radians) {

    return radians * RAD2DEG;
}

bool threepp::math::isPowerOfTwo(int value) {

    return (value & (value - 1)) == 0 && value != 0;
}

float threepp::math::floorPowerOfTwo(float value) {

    return std::powf(2.f, floor(std::log(value) / LN2));
}

float threepp::math::ceilPowerOfTwo(float value) {

    return std::powf(2.f, std::ceil(std::log(value) / LN2));
}
