
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

class PIDRegulator: public Regulator {

public:
    struct Parameters {
        float kp;
        float ti;
        float td;
    };

    PIDRegulator(): PIDRegulator(1, 0.01, 0.001) {}

    PIDRegulator(float kp, float ti, float td): params_{kp, ti, td} {}

    float regulate(float error, float dt) override {
        if (dt == 0) dt = std::numeric_limits<float>::min();

        // integration with windup guarding
        integral_ += (error * dt);
        if (windup_guard_) {
            integral_ = std::clamp(integral_, -windup_guard_.value(), windup_guard_.value());
        }

        // differentiation
        float diff = ((error - prevError_) / dt);

        // save current error as previous error for next iteration
        prevError_ = error;

        // scaling
        float P = (params_.kp * error);
        float I = (params_.ti * integral_);
        float D = (params_.td * diff);

        // summation of terms
        return P + I + D;
    }

    void setWindupGuard(const std::optional<float>& windupGuard) {
        windup_guard_ = windupGuard;
    }

    [[nodiscard]] PIDRegulator::Parameters& params() {
        return params_;
    }

    [[nodiscard]] const PIDRegulator::Parameters& params() const {
        return params_;
    }

private:
    float integral_{};
    float prevError_{};
    PIDRegulator::Parameters params_;
    std::optional<float> windup_guard_;
};


#endif//THREEPP_REGULATOR_HPP
