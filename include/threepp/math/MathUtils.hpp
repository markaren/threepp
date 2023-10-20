
#ifndef THREEPP_MATHUTILS_HPP
#define THREEPP_MATHUTILS_HPP

#include <limits>
#include <string>

namespace threepp::math {

    const float LN2 = 0.6931471805599453094172321214581765680755001343602552541206800094;
    const float PI = 3.14159265358979323846f;// 2.f * std::acos(0.f)
    const float TWO_PI = 6.28318530718f;

    const float DEG2RAD = PI / 180.f;
    const float RAD2DEG = 180.f / PI;

    // Generate a UUID (universally unique identifier).
    std::string generateUUID();

    // compute euclidian modulo of m % n
    // https://en.wikipedia.org/wiki/Modulo_operation
    float euclideanModulo(float n, float m);

    // Linear mapping from range <a1, a2> to range <b1, b2>
    float mapLinear(float x, float a1, float a2, float b1, float b2);

    // Returns the percentage in the closed interval [0, 1] of the given value between the start and end point.
    float inverseLerp(float x, float y, float value);

    // Returns a value linearly interpolated from two known points based on the given interval - t = 0 will return x and t = 1 will return y.
    float lerp(float x, float y, float t);

    // Smoothly interpolate a number from x toward y in a spring-like manner using the dt to maintain frame rate independent movement.
    // For details, see Frame rate independent damping using lerp.
    float damp(float x, float y, float lambda, float dt);

    // Converts degrees to radians.
    float degToRad(float degrees);

    // Converts radians to degrees.
    float radToDeg(float radians);

    // Random integer in the interval [low, high].
    int randInt(int low, int high);

    // Random float in the interval [0, 1]
    float randFloat();

    // Random float in the interval [low, high]
    float randFloat(float min, float max);

    // Random float in the interval [- range / 2, range / 2].
    float randFloatSpread(float range);

    // Return true if n is a power of 2.
    bool isPowerOfTwo(int value);

    // Returns the smallest power of 2 that is greater than or equal to n.
    int ceilPowerOfTwo(float value);

    // Returns the largest power of 2 that is less than or equal to n.
    int floorPowerOfTwo(float value);

    //https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    template<typename T>
    inline int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    bool compareFloats(float f1, float f2, float eps = std::numeric_limits<float>::epsilon());

}// namespace threepp::math

#endif//THREEPP_MATHUTILS_HPP
