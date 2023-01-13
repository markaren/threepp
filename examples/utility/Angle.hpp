

#ifndef THREEPP_ANGLE_HPP
#define THREEPP_ANGLE_HPP

#include "threepp/math/MathUtils.hpp"

#include <vector>

class Angle {

public:
    enum Repr {
        DEG,
        RAD
    };

    Angle(float value, Repr repr) {
        if (repr == DEG) {
            rad_ = value * threepp::math::DEG2RAD;
        } else {
            rad_ = value;
        }
    }

    [[nodiscard]] float inRadians() const {
        return rad_;
    }

    [[nodiscard]] float inDegrees() const {
        return rad_ * threepp::math::RAD2DEG;
    }

    static Angle degrees(float value) {
        return {value, Repr::DEG};
    }

    static Angle radians(float value) {
        return {value, Repr::RAD};
    }

private:
    float rad_;
};

inline std::vector<float> inRadians(const std::vector<Angle>& values) {
    std::vector<float> result(values.size());
    for (unsigned i = 0; i < values.size(); ++i) {
        result[i] = values[i].inRadians();
    }
    return result;
}

inline std::vector<float> inDegrees(const std::vector<Angle>& values) {
    std::vector<float> result(values.size());
    for (unsigned i = 0; i < values.size(); ++i) {
        result[i] = values[i].inDegrees();
    }
    return result;
}

inline std::vector<Angle> asAngles(const std::vector<float>& values, Angle::Repr repr) {
    std::vector<Angle> result;
    for (const auto& v : values) {
        result.emplace_back(v, repr);
    }
    return result;
}

#endif//THREEPP_ANGLE_HPP
