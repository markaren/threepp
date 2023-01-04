
#ifndef THREEPP_PID_HPP
#define THREEPP_PID_HPP

#include <limits>
#include <optional>

#include "threepp/math/MathUtils.hpp"

class PID {

public:
    float kp;
    float ti;
    float td;

    PID(float kp, float ti, float td) : kp(kp), ti(ti), td(td) {}

    float regulate(float setPoint, float measuredValue, float dt) {
        if (dt == 0) dt = std::numeric_limits<float>::min();

        float curr_error = (setPoint - measuredValue);

        // integration with windup guarding
        integral += (curr_error * dt);
        if (windup_guard) {
            if (integral < -(*windup_guard)) {
                integral = -(*windup_guard);
            } else if (integral > *windup_guard) {
                integral = *windup_guard;
            }
        }

        // differentiation
        float diff = ((curr_error - prev_error) / dt);

        // save current error as previous error for next iteration
        prev_error = curr_error;

        // scaling
        float P = (kp * curr_error);
        float I = (ti * integral);
        float D = (td * diff);

        // summation of terms
        return threepp::math::clamp(P + I + D, -1.f, 1.f);
    }

    void setWindupGuard(const std::optional<float> &windupGuard) {
        windup_guard = windupGuard;
    }

private:


    float integral;
    float prev_error{};
    std::optional<float> windup_guard;
};


#endif//THREEPP_PID_HPP
