
#ifndef THREEPP_MATH_UTILS_HPP
#define THREEPP_MATH_UTILS_HPP

#include <cmath>
#include <random>
#include <sstream>

namespace threepp {

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    const float PI = 2.0f * std::acos(0.0f);

    const float DEG2RAD = PI / 180.0f;
    const float RAD2DEG = 180.0f / PI;

    //https://stackoverflow.com/questions/24365331/how-can-i-generate-uuid-in-c-without-using-boost-library/58467162
    std::string generateUUID() {

        std::stringstream ss;
        int i;
        ss << std::hex;
        for (i = 0; i < 8; i++) {
            ss << dis(gen);
        }
        ss << "-";
        for (i = 0; i < 4; i++) {
            ss << dis(gen);
        }
        ss << "-4";
        for (i = 0; i < 3; i++) {
            ss << dis(gen);
        }
        ss << "-";
        ss << dis2(gen);
        for (i = 0; i < 3; i++) {
            ss << dis(gen);
        }
        ss << "-";
        for (i = 0; i < 12; i++) {
            ss << dis(gen);
        };
        return ss.str();
    }


    float degToRad(const float degrees) {

        return degrees * DEG2RAD;
    }

    float radToDeg(const float radians) {

        return radians * RAD2DEG;
    }

}// namespace threepp

#endif//THREEPP_MATH_UTILS_HPP
