
#ifndef THREEPP_MATHUTILS_HPP
#define THREEPP_MATHUTILS_HPP

#include <limits>
#include <string>

namespace threepp::math {

    const float LN2 = 0.6931471805599453094f;
    const float PI = 3.14159265358979323846f;// 2.f * std::acos(0.f)
    const float TWO_PI = 6.28318530718f;

    const float DEG2RAD = PI / 180.f;
    const float RAD2DEG = 180.f / PI;

    std::string generateUUID();

    float mapLinear(float x, float a1, float a2, float b1, float b2);

    float inverseLerp(float x, float y, float value);

    float lerp(float x, float y, float t);

    float damp(float x, float y, float lambda, float dt);

    float degToRad(float degrees);

    float radToDeg(float radians);

    int randomInRange(int min, int max);

    float random();

    float randomInRange(float min, float max);

    bool isPowerOfTwo(int value);

    float ceilPowerOfTwo(float value);

    float floorPowerOfTwo(float value);

    //https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    template<typename T>
    inline int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    bool compareFloats(float f1, float f2, float eps = std::numeric_limits<float>::epsilon());

}// namespace threepp::math

#endif//THREEPP_MATHUTILS_HPP
