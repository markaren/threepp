
#ifndef THREEPP_INVERTEDPENDELUM_HPP
#define THREEPP_INVERTEDPENDELUM_HPP

#include <cmath>

#include "threepp/math/MathUtils.hpp"

class InvertedPendulum {

public:
    InvertedPendulum(double kp, double kd): kp(kp), kd(kd) {
        cartPosition = 0.0;
        pendulumAngle = threepp::math::degToRad(25);// Initial angle (upright position)
        cartVelocity = 0.0;
        pendulumAngularVelocity = 0.0;
    }

    // Function to simulate the system for a certain duration
    void simulate(double duration, double externalForce) {
        int steps = static_cast<int>(duration / internalTimeStep);
        for (int i = 0; i < steps; ++i) {
            double controlSignal = control();
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

private:
    double cartPosition;           // Position of the cart
    double pendulumAngle;          // Angle of the pendulum
    double cartVelocity;           // Velocity of the cart
    double pendulumAngularVelocity;// Angular velocity of the pendulum

    const double g = 9.81;            // Acceleration due to gravity
    const double cartMass = 2.0;      // Mass of the cart
    const double pendulumMass = 0.2;   // Mass of the pendulum
    const double pendulumLength = 1.0;// Length of the pendulum

    double internalTimeStep = 0.01;// Internal time step for the simulation

    // Controller gains
    double kp;// Proportional gain
    double kd;// Derivative gain

    // Boundary limits for the cart's position
    double minX = -4.0;
    double maxX = 4.0;

    // Coefficient of restitution for the cart
    double restitution = 0.2;// Adjust as needed

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
        double prevCartPosition = cartPosition;
        cartPosition += cartVelocity * internalTimeStep;

        // Handle boundary collisions
        if (cartPosition < minX || cartPosition > maxX) {
            cartPosition = prevCartPosition;// Reset cart position to previous value
            cartVelocity *= -restitution;   // Reverse cart velocity with restitution factor
        }

        pendulumAngle += pendulumAngularVelocity * internalTimeStep;
    }

    // Function to control the pendulum using PD control
    [[nodiscard]] double control() const {
        double desiredAngle = 0;// Desired angle (upright position)
        double error = desiredAngle - pendulumAngle;
        double controlSignal = kp * error - kd * pendulumAngularVelocity;
        return -controlSignal;
    }
};

#endif//THREEPP_INVERTEDPENDELUM_HPP
