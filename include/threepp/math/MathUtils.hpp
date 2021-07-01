
#ifndef THREEPP_MATHUTILS_HPP
#define THREEPP_MATHUTILS_HPP

#include <cmath>
#include <random>
#include <sstream>

namespace threepp::math {

    const float LN2 = std::log(2.f);
    const float PI = 2.f * std::acos(0.f);

    const float DEG2RAD = PI / 180.f;
    const float RAD2DEG = 180.f / PI;

    //https://stackoverflow.com/questions/24365331/how-can-i-generate-uuid-in-c-without-using-boost-library/58467162
    std::string generateUUID();

    float degToRad(float degrees);

    float radToDeg(float radians);

    bool isPowerOfTwo(int value);

    float ceilPowerOfTwo( float value );

    float floorPowerOfTwo( float value );


}// namespace threepp

#endif//THREEPP_MATHUTILS_HPP
