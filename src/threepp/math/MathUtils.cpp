
#include "threepp/math/MathUtils.hpp"

#include <random>

// Linear mapping from range <a1, a2> to range <b1, b2>
float mapLinear(float x, float a1, float a2, float b1, float b2) {

    return b1 + (x - a1) * (b2 - b1) / (a2 - a1);
}

// https://www.gamedev.net/tutorials/programming/general-and-gameplay-programming/inverse-lerp-a-super-useful-yet-often-overlooked-function-r5230/
float inverseLerp(float x, float y, float value) {

    if (x != y) {

        return (value - x) / (y - x);

    } else {

        return 0;
    }
}

// https://en.wikipedia.org/wiki/Linear_interpolation
float lerp(float x, float y, float t) {

    return (1 - t) * x + t * y;
}

// http://www.rorydriscoll.com/2016/03/07/frame-rate-independent-damping-using-lerp/
float damp(float x, float y, float lambda, float dt) {

    return lerp(x, y, 1 - std::exp(-lambda * dt));
}

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
