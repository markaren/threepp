
#ifndef THREEPP_MATHUTILS_HPP
#define THREEPP_MATHUTILS_HPP

#include <limits>
#include <numbers>
#include <string>

namespace threepp::math {

    constexpr float LN2 = std::numbers::ln2_v<float>;
    constexpr float PI = std::numbers::pi_v<float>;
    constexpr float TWO_PI = PI * 2.f;

    constexpr float DEG2RAD = PI / 180.f;
    constexpr float RAD2DEG = 180.f / PI;

    std::string generateUUID();

    // compute euclidian modulo of m % n
    // https://en.wikipedia.org/wiki/Modulo_operation
    float euclideanModulo( float n, float m );

    // Linear mapping from range <a1, a2> to range <b1, b2>
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
