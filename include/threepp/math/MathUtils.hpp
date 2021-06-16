
#ifndef THREEPP_MATHUTILS_HPP
#define THREEPP_MATHUTILS_HPP

#include <cmath>
#include <random>
#include <sstream>

namespace threepp {

    const float PI = 2.0f * std::acos(0.0f);

    const float DEG2RAD = PI / 180.0f;
    const float RAD2DEG = 180.0f / PI;

    //https://stackoverflow.com/questions/24365331/how-can-i-generate-uuid-in-c-without-using-boost-library/58467162
    std::string generateUUID();

    float degToRad( float degrees);

    float radToDeg( float radians);

}// namespace threepp

#endif//THREEPP_MATHUTILS_HPP
