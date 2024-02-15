
#ifndef THREEPP_INVERTEDPENDELUM_HPP
#define THREEPP_INVERTEDPENDELUM_HPP

#include <cmath>

#include "threepp/math/MathUtils.hpp"

class InvertedPendulum {

public:
    InvertedPendulum(double kp, double kd): kp(kp), kd(kd) {
        cartPosition = 0.0;
        pendulumAngle = threepp::math::degToRad(5);// Initial angle (upright position)
        cartVelocity = 0.0;
        pendulumAngularVelocity = 0.0;
    }

    // Function to simulate the system for a certain duration
    void simulate(double duration, double externalForce) {
        int steps = static_cast<int>(duration / internalTimeStep);
        for (int i = 0; i < steps; ++i) {
            double controlSignal = control_ ? control() : 0;
            update(controlSignal + externalForce);
        }
    }

    [[nodiscard]] double getCartPosition() const {
        return cartPosition;
    }

    [[nodiscard]] double getPendulumAngle() const {
        return pendulumAngle;
    }

    [[nodiscard]] double getPendulumLength() const {
        return pendulumLength;
    }

    [[nodiscard]] bool isControl() const {
        return control_;
    }

    void setControl(bool control) {
        control_ = control;
    }

private:
    bool control_ = true;

    double cartPosition;           // Position of the cart
    double pendulumAngle;          // Angle of the pendulum
    double cartVelocity;           // Velocity of the cart
    double pendulumAngularVelocity;// Angular velocity of the pendulum

    const double g = 9.81;            // Acceleration due to gravity
    const double cartMass = 2;      // Mass of the cart
    const double pendulumMass = 0.1;    // Mass of the pendulum
    const double pendulumLength = 1.5;// Length of the pendulum

    double internalTimeStep = 0.01;// Internal time step for the simulation

    double minPendulumAngle = threepp::math::degToRad(-60);
    double maxPendulumAngle = threepp::math::degToRad(60);

    // Controller gains
    double kp;// Proportional gain
    double kd;// Derivative gain

    // Boundary limits for the cart's position
    double minX = -4.0;
    double maxX = 4.0;

    double dampingCoefficient = 0.1;// Damping coefficient for the pendulum

    // Function to update the state of the system
    void update(double force) {
        double totalMass = cartMass + pendulumMass;

        // Compute accelerations
        double cartAcceleration = (force - pendulumMass * pendulumLength * pendulumAngularVelocity * pendulumAngularVelocity * sin(pendulumAngle)) / totalMass;
        double pendulumAngularAcceleration = (g * std::sin(pendulumAngle) - std::cos(pendulumAngle) * (force / totalMass)) / (pendulumLength * (4.0 / 3.0 - (pendulumMass * std::cos(pendulumAngle) * std::cos(pendulumAngle)) / totalMass));

        // Apply damping to velocities
        cartVelocity += (cartAcceleration - dampingCoefficient * cartVelocity) * internalTimeStep;
        pendulumAngularVelocity += (pendulumAngularAcceleration - dampingCoefficient * pendulumAngularVelocity) * internalTimeStep;

        // Update positions
        cartPosition += cartVelocity * internalTimeStep;

        // Handle boundary collisions
        if (cartPosition < minX || cartPosition > maxX) {
            cartPosition *= -1;
        }

        pendulumAngle += pendulumAngularVelocity * internalTimeStep;

        // Limit pendulum angle to prevent it from falling below the cart
        if (pendulumAngle < minPendulumAngle) {
            pendulumAngle = minPendulumAngle;
            pendulumAngularVelocity = 0; // Stop pendulum motion if it reaches the limit
        } else if (pendulumAngle > maxPendulumAngle) {
            pendulumAngle = maxPendulumAngle;
            pendulumAngularVelocity = 0; // Stop pendulum motion if it reaches the limit
        }
    }

    static double shortestAngularDistance(double angle1, double angle2) {
        // Normalize angles to be within the range of -pi to pi radians
        angle1 = fmod(angle1 + threepp::math::PI, 2 * threepp::math::PI) - threepp::math::PI;
        angle2 = fmod(angle2 + threepp::math::PI, 2 * threepp::math::PI) - threepp::math::PI;

        // Calculate the signed angular distance
        double delta = angle2 - angle1;

        // Ensure delta is within the range of -pi to pi radians
        if (delta <= -threepp::math::PI) {
            delta += threepp::math::TWO_PI;
        } else if (delta > threepp::math::PI) {
            delta -= threepp::math::TWO_PI;
        }

        return delta;
    }

    // Function to control the pendulum using PD control
    [[nodiscard]] double control() const {
        double desiredAngle = 0;
        double error = shortestAngularDistance(desiredAngle, pendulumAngle);
        double controlSignal = kp * error + kd * pendulumAngularVelocity;
        return controlSignal;
    }
};

#endif//THREEPP_INVERTEDPENDELUM_HPP
