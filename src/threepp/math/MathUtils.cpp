
#include "threepp/math/MathUtils.hpp"

#include <random>

using namespace threepp;

// Linear mapping from range <a1, a2> to range <b1, b2>
float math::mapLinear(float x, float a1, float a2, float b1, float b2) {

    return b1 + (x - a1) * (b2 - b1) / (a2 - a1);
}

// https://www.gamedev.net/tutorials/programming/general-and-gameplay-programming/inverse-lerp-a-super-useful-yet-often-overlooked-function-r5230/
float math::inverseLerp(float x, float y, float value) {

    if (x != y) {

        return (value - x) / (y - x);

    } else {

        return 0;
    }
}

// https://en.wikipedia.org/wiki/Linear_interpolation
float math::lerp(float x, float y, float t) {

    return (1 - t) * x + t * y;
}

// http://www.rorydriscoll.com/2016/03/07/frame-rate-independent-damping-using-lerp/
float math::damp(float x, float y, float lambda, float dt) {

    return lerp(x, y, 1 - std::exp(-lambda * dt));
}

float math::degToRad(float degrees) {

    return degrees * DEG2RAD;
}

float math::radToDeg(float radians) {

    return radians * RAD2DEG;
}

int math::randomInRange(int min, int max) {

    static std::random_device rd;
    static std::mt19937 e2(rd());

    std::uniform_int_distribution<> dist(min, max);

    return dist(e2);
}

float math::randomInRange(float min, float max) {

    static std::random_device rd;
    static std::mt19937 e2(rd());

    std::uniform_real_distribution<> dist(min, max);

    return static_cast<float>(dist(e2));
}

bool math::isPowerOfTwo(int value) {

    return (value & (value - 1)) == 0 && value != 0;
}

float math::floorPowerOfTwo(float value) {

    return std::pow(2.f, floor(std::log(value) / LN2));
}

float math::ceilPowerOfTwo(float value) {

    return std::pow(2.f, std::ceil(std::log(value) / LN2));
}

bool math::compareFloats(float f1, float f2) {

    return (std::fabs(f1 - f2) <= std::numeric_limits<float>::epsilon() * std::fmax(std::fabs(f1), std::fabs(f2)));
}
