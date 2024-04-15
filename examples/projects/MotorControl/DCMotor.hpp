
#ifndef THREEPP_DCMOTOR_HPP
#define THREEPP_DCMOTOR_HPP

class DCMotor {

public:
    DCMotor(double resistance, double inertia, double damping)
        : resistance(resistance), inertia(inertia),
          damping(damping), position(0), velocity(0) {}

    // Update the motor's state based on a position control signal and a time step
    void update(double positionControlSignal, double timeStep) {
        // Calculate current through the motor using Ohm's law
        double current = positionControlSignal / resistance;

        // Calculate torque
        double torque = current;

        // Calculate angular acceleration (torque = inertia * angular acceleration)
        double angularAcceleration = torque / inertia;

        // Apply damping
        angularAcceleration -= velocity * damping / inertia;

        // Update velocity
        velocity += angularAcceleration * timeStep;

        // Update position
        position += velocity * timeStep;
    }

    // Get the current position of the motor
    [[nodiscard]] double getPosition() const {
        return position;
    }

private:
    double resistance;// Resistance of the motor
    double inertia;   // Moment of inertia of the motor's rotor
    double damping;   // Damping coefficient

    double position;// Position of the motor
    double velocity;// Velocity of the motor
};

#endif//THREEPP_DCMOTOR_HPP
