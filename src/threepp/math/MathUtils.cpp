
#include "threepp/math/MathUtils.hpp"

using namespace threepp::math;

namespace {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);
}// namespace

std::string threepp::math::generateUUID() {

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

float threepp::math::degToRad(const float degrees) {

    return degrees * DEG2RAD;
}

float threepp::math::radToDeg(const float radians) {

    return radians * RAD2DEG;
}

bool threepp::math::isPowerOfTwo(int value) {

    return (value & (value - 1)) == 0 && value != 0;
}

float threepp::math::floorPowerOfTwo(float value) {

    return std::powf(2.f, floor(std::log(value) / LN2));
}

float threepp::math::ceilPowerOfTwo(float value) {

    return std::powf(2.f, std::ceil(std::log(value) / LN2));
}
