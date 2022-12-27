
#ifndef THREEPP_MATHUTILS_HPP
#define THREEPP_MATHUTILS_HPP

#include <algorithm>
#include <cmath>

namespace threepp::math {

    const float LN2 = std::log(2.f);
    const float PI = 2.f * std::acos(0.f);

    const float DEG2RAD = PI / 180.f;
    const float RAD2DEG = 180.f / PI;

    template<class T>
    T clamp(T value, T min, T max) {

        return std::max(min, std::min(max, value));
    }

    template<class T>
    void clampInPlace(T& value, T min, T max) {

        value = std::max(min, std::min(max, value));
    }

    float mapLinear(float x, float a1, float a2, float b1, float b2);

    float inverseLerp(float x, float y, float value);

    float lerp(float x, float y, float t);

    float damp(float x, float y, float lambda, float dt);

    float degToRad(float degrees);

    float radToDeg(float radians);

    int randomInRange(int min, int max);

    float randomInRange(float min, float max);

    bool isPowerOfTwo(int value);

    float ceilPowerOfTwo(float value);

    float floorPowerOfTwo(float value);

    template<typename T>
    inline int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    bool compareFloats(float f1, float f2);

}// namespace threepp::math

#endif//THREEPP_MATHUTILS_HPP
