
#ifndef THREEPP_EQUALS_UTIL_HPP
#define THREEPP_EQUALS_UTIL_HPP

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"

#include <cmath>

constexpr float eps = 0.0001f;

inline bool matrixEquals4(const threepp::Matrix4& a, const threepp::Matrix4& b, float tolerance = eps) {

    for (unsigned i = 0, il = a.elements.size(); i < il; i++) {

        if (const auto delta = a.elements[i] - b.elements[i]; delta > tolerance) {

            return false;
        }
    }

    return true;
}

inline bool quatEquals(const threepp::Quaternion& a, const threepp::Quaternion& b, double tolerance = eps) {

    const auto diff = std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z) + std::abs(a.w - b.w);
    return diff < tolerance;
}

inline bool eulerEquals(const threepp::Euler& a, const threepp::Euler& b, float tolerance = eps) {

    const auto diff = std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z);
    return diff < tolerance;
}


#endif//THREEPP_EQUALS_UTIL_HPP
