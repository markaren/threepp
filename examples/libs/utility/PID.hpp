
#ifndef THREEPP_PID_HPP
#define THREEPP_PID_HPP

#include <limits>
#include <optional>
#include <algorithm>

#include "threepp/math/MathUtils.hpp"

struct PIDParameters {
    float kp;
    float ti;
    float td;
};

class PID {

public:

    PID() : PID(1, 0.01, 0.001) {}

    PID(float kp, float ti, float td) : params_{kp, ti, td} {}

    float regulate(float setPoint, float measuredValue, float dt) {
        if (dt == 0) dt = std::numeric_limits<float>::min();

        float curr_error = (setPoint - measuredValue);

        // integration with windup guarding
        integral_ += (curr_error * dt);
        if (windup_guard_) {
            integral_ = std::clamp(integral_, -windup_guard_.value(), windup_guard_.value());
        }

        // differentiation
        float diff = ((curr_error - prev_error_) / dt);

        // save current error as previous error for next iteration
        prev_error_ = curr_error;

        // scaling
        float P = (params_.kp * curr_error);
        float I = (params_.ti * integral_);
        float D = (params_.td * diff);

        // summation of terms
        return std::clamp(P + I + D, -1.f, 1.f);
    }

    void setWindupGuard(const std::optional<float> &windupGuard) {
        windup_guard_ = windupGuard;
    }

    [[nodiscard]] float error() const {
        return prev_error_;
    }

    [[nodiscard]] PIDParameters& params() {
        return params_;
    }

    [[nodiscard]] const PIDParameters& params() const{
        return params_;
    }

private:

    float integral_{};
    float prev_error_{};
    PIDParameters params_;
    std::optional<float> windup_guard_;
};


#endif//THREEPP_PID_HPP
