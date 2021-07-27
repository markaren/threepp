
#include "threepp/math/MathUtils.hpp"

#include <random>

float threepp::math::degToRad(const float degrees) {

    return degrees * DEG2RAD;
}

float threepp::math::radToDeg(const float radians) {

    return radians * RAD2DEG;
}

float threepp::math::randomInRange(float min, float max) {

    static std::random_device rd;
    static std::mt19937 e2(rd());

    std::uniform_real_distribution<> dist(min, max);

    return static_cast<float>(dist(e2));
}

bool threepp::math::isPowerOfTwo(int value) {

    return (value & (value - 1)) == 0 && value != 0;
}

float threepp::math::floorPowerOfTwo(float value) {

    return std::pow(2.f, floor(std::log(value) / LN2));
}

float threepp::math::ceilPowerOfTwo(float value) {

    return std::pow(2.f, std::ceil(std::log(value) / LN2));
}
