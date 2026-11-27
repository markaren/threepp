
#ifndef THREEPP_REGULATOR_HPP
#define THREEPP_REGULATOR_HPP

#include <algorithm>
#include <limits>
#include <optional>

class Regulator {
public:
    virtual float regulate(float error, float dt) = 0;

    virtual ~Regulator() = default;
};

class PDRegulator: public Regulator {

public:
    struct Parameters {
        float kp;
        float td;
    };

    PDRegulator(): PDRegulator(1, 0.001) {}

    PDRegulator(float kp, float td): params_{kp, td} {}

    float regulate(float error, float dt) override {
        if (dt == 0) dt = std::numeric_limits<float>::min();

        // differentiation
        float diff = ((error - prevError_) / dt);

        // save current error as previous error for next iteration
        prevError_ = error;

        // scaling
        float P = (params_.kp * error);
        float D = (params_.td * diff);

        // summation of terms
        return P + D;
    }

    [[nodiscard]] PDRegulator::Parameters& params() {
        return params_;
    }

    [[nodiscard]] const PDRegulator::Parameters& params() const {
        return params_;
    }

private:
    float prevError_{};
    PDRegulator::Parameters params_;
};

// Parallel-form PID:  u = kp * e  +  ki * integral(e dt)  +  kd * de/dt
//
// - The derivative is low-pass filtered (time constant derivativeFilter), so
//   measurement noise and the step-to-step jitter of a fast loop do not reach
//   the output amplified by 1/dt.
// - The first sample has no derivative: there is no previous error yet, and
//   taking it from zero would turn the initial error into a one-step spike.
// - With output limits set, the output is clamped to them and the integrator
//   holds while the output sits at a limit and the error pushes it further
//   (conditional integration, a common anti-windup), so it does not build up
//   a charge that overshoots once the error changes sign.
class PIDRegulator: public Regulator {

public:
    struct Parameters {
        float kp;// proportional gain
        float ki;// integral gain [1/s]
        float kd;// derivative gain [s]
    };

    // The terms of the last output, for display.
    struct Terms {
        float p{};
        float i{};
        float d{};
    };

    PIDRegulator(): PIDRegulator(1, 0, 0) {}

    PIDRegulator(float kp, float ki, float kd): params_{kp, ki, kd} {}

    float regulate(float error, float dt) override {
        if (dt <= 0) return output_;// no time has passed: hold

        const float rawDerivative = hasPrevious_ ? (error - prevError_) / dt : 0.f;
        prevError_ = error;
        hasPrevious_ = true;
        const float alpha = derivativeFilter_ > 0 ? dt / (derivativeFilter_ + dt) : 1.f;
        derivative_ += alpha * (rawDerivative - derivative_);

        terms_.p = params_.kp * error;
        terms_.d = params_.kd * derivative_;

        const float integral = integral_ + error * dt;
        const float unclamped = terms_.p + params_.ki * integral + terms_.d;
        const bool windingUp = limits_ && ((unclamped > limits_->max && error > 0) ||
                                           (unclamped < limits_->min && error < 0));
        if (!windingUp) integral_ = integral;
        terms_.i = params_.ki * integral_;

        output_ = terms_.p + terms_.i + terms_.d;
        if (limits_) output_ = std::clamp(output_, limits_->min, limits_->max);
        return output_;
    }

    void setOutputLimits(float min, float max) {
        limits_ = Limits{min, max};
    }

    void clearOutputLimits() {
        limits_.reset();
    }

    // Time constant of the derivative's low-pass filter [s]; 0 turns it off.
    void setDerivativeFilter(float timeConstant) {
        derivativeFilter_ = std::max(0.f, timeConstant);
    }

    // Forget the integral, the derivative history and the last output.
    void reset() {
        integral_ = 0;
        prevError_ = 0;
        derivative_ = 0;
        hasPrevious_ = false;
        output_ = 0;
        terms_ = {};
    }

    [[nodiscard]] const Terms& terms() const {
        return terms_;
    }

    [[nodiscard]] PIDRegulator::Parameters& params() {
        return params_;
    }

    [[nodiscard]] const PIDRegulator::Parameters& params() const {
        return params_;
    }

private:
    struct Limits {
        float min;
        float max;
    };

    PIDRegulator::Parameters params_;
    std::optional<Limits> limits_;
    float derivativeFilter_{};

    float integral_{};
    float prevError_{};
    float derivative_{};
    bool hasPrevious_{};
    float output_{};
    Terms terms_;
};


#endif//THREEPP_REGULATOR_HPP
