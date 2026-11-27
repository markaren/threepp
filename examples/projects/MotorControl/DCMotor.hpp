#ifndef THREEPP_DCMOTOR_HPP
#define THREEPP_DCMOTOR_HPP

#include <algorithm>

// A permanent-magnet DC motor turning an inertia. The armature inductance is
// left out (its electrical time constant is far shorter than the mechanical
// one), so the current follows the voltage at once:
//
//   i = (V - ke * w) / R          armature current, back-EMF ke * w
//   J * dw/dt = kt * i - b * w - load
//
// In SI units the torque constant kt [N m/A] and the back-EMF constant ke
// [V s/rad] are the same number, so the model carries one of them.
class DCMotor {

public:
    struct Parameters {
        double resistance = 2.0;    // R  [ohm]
        double torqueConstant = 0.5;// kt = ke
        double inertia = 0.1;       // J  [kg m^2]
        double damping = 0.05;      // b  [N m s/rad]
        double maxVoltage = 12.0;   // supply limit [V]
    };

    explicit DCMotor(const Parameters& params = {}): params_(params) {}

    // Advance by `dt` seconds with `voltage` applied (clamped to the supply)
    // and a `loadTorque` acting against positive rotation. Semi-implicit Euler:
    // the new velocity moves the position, which is stable for the small fixed
    // steps the example takes.
    void update(double voltage, double loadTorque, double dt) {
        voltage_ = std::clamp(voltage, -params_.maxVoltage, params_.maxVoltage);
        current_ = (voltage_ - params_.torqueConstant * velocity_) / params_.resistance;
        const double torque = params_.torqueConstant * current_ - params_.damping * velocity_ - loadTorque;
        velocity_ += torque / params_.inertia * dt;
        position_ += velocity_ * dt;
    }

    void reset(double position = 0) {
        position_ = position;
        velocity_ = 0;
        current_ = 0;
        voltage_ = 0;
    }

    [[nodiscard]] double position() const { return position_; }// [rad]
    [[nodiscard]] double velocity() const { return velocity_; }// [rad/s]
    [[nodiscard]] double current() const { return current_; }  // [A]
    [[nodiscard]] double voltage() const { return voltage_; }  // [V], as applied

    [[nodiscard]] const Parameters& params() const { return params_; }

private:
    Parameters params_;

    double position_{};
    double velocity_{};
    double current_{};
    double voltage_{};
};

#endif//THREEPP_DCMOTOR_HPP
