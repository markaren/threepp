
#ifndef THREEPP_INVERTEDPENDELUM_HPP
#define THREEPP_INVERTEDPENDELUM_HPP

#include <cmath>

#include "utility/Regulator.hpp"

#include "threepp/math/MathUtils.hpp"

class InvertedPendulum {

public:
    // Function to simulate the system for a certain duration
    void simulate(double duration, double externalForce, Regulator* regulator = nullptr) {
        int steps = static_cast<int>(duration / internalTimeStep);
        for (int i = 0; i < steps; ++i) {
            double controlSignal = regulator ? control(*regulator) : 0;
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
    double cartPosition{0};                          // Position of the cart
    double pendulumAngle{threepp::math::degToRad(5)};// Angle of the pendulum
    double cartVelocity{0};                          // Velocity of the cart
    double pendulumAngularVelocity{0};               // Angular velocity of the pendulum

    const double g = 9.81;            // Acceleration due to gravity
    const double cartMass = 2;        // Mass of the cart
    const double pendulumMass = 0.1;  // Mass of the pendulum
    const double pendulumLength = 1.5;// Length of the pendulum

    double internalTimeStep = 0.01;// Internal time step for the simulation

    double minPendulumAngle = threepp::math::degToRad(-60);
    double maxPendulumAngle = threepp::math::degToRad(60);

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
            pendulumAngularVelocity = 0;// Stop pendulum motion if it reaches the limit
        } else if (pendulumAngle > maxPendulumAngle) {
            pendulumAngle = maxPendulumAngle;
            pendulumAngularVelocity = 0;// Stop pendulum motion if it reaches the limit
        }
    }

    static double shortestAngularDistance(double angle1, double angle2) {
        // Normalize angles to be within the range of -pi to pi radians
        angle1 = std::fmod(angle1 + threepp::math::PI, 2 * threepp::math::PI) - threepp::math::PI;
        angle2 = std::fmod(angle2 + threepp::math::PI, 2 * threepp::math::PI) - threepp::math::PI;

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
    [[nodiscard]] double control(Regulator& regulator) const {
        double desiredAngle = 0;
        double error = shortestAngularDistance(desiredAngle, pendulumAngle);
        double controlSignal = regulator.regulate(error, internalTimeStep);
        return controlSignal;
    }
};

#endif//THREEPP_INVERTEDPENDELUM_HPP
