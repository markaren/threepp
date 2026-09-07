// RLPendulumTrainer — a plain-C++ facade over an RLtools (https://rl.tools) SAC
// training loop for the classic Pendulum swing-up task.
//
// ALL of RLtools' heavily templated machinery is confined to RLPendulumTrainer.cpp,
// so the rest of the demo — and threepp — only ever sees plain floats and longs.
//
// Threading contract (see RLPendulumTrainer.cpp for the why):
//   * trainStep() and reset() must be called from ONE thread (the trainer thread).
//   * policyAction(), stepCount(), done() are safe to call from another thread
//     (the render thread) concurrently with trainStep().
// Internally a periodically-refreshed policy "snapshot" decouples the heavy,
// lock-free training step from the cheap, lock-guarded inference call.

#ifndef THREEPP_RL_PENDULUM_TRAINER_HPP
#define THREEPP_RL_PENDULUM_TRAINER_HPP

#include <memory>

namespace rldemo {

    class RLPendulumTrainer {

    public:
        explicit RLPendulumTrainer(unsigned int seed = 1);
        ~RLPendulumTrainer();

        RLPendulumTrainer(const RLPendulumTrainer&) = delete;
        RLPendulumTrainer& operator=(const RLPendulumTrainer&) = delete;

        // Run a single SAC loop step (collect one transition + gradient updates).
        // Returns true once the training-step budget is exhausted. [trainer thread]
        bool trainStep();

        // Deterministic policy evaluation: observation -> torque action in [-1, 1].
        // [render thread — safe alongside trainStep()]
        float policyAction(float cosTheta, float sinTheta, float thetaDot);

        // Restart training from scratch with a new seed. [trainer thread]
        void reset(unsigned int seed);

        [[nodiscard]] long stepCount() const;  // steps completed so far
        [[nodiscard]] bool done() const;        // budget exhausted?

        static long stepLimit();    // total training-step budget
        static long warmupSteps();  // random-exploration steps before learning kicks in

        // Pendulum-v1 constants (mirror RLtools' pendulum::DefaultParameters), exposed
        // so the visualization integrates the *identical* dynamics the policy trained on.
        static constexpr float kG = 10.0f;        // gravity
        static constexpr float kL = 1.0f;         // pole length
        static constexpr float kM = 1.0f;         // pole mass
        static constexpr float kDt = 0.05f;       // simulation timestep (20 Hz)
        static constexpr float kMaxTorque = 2.0f; // |torque| ceiling
        static constexpr float kMaxSpeed = 8.0f;  // |angular velocity| ceiling
        static constexpr int kEpisodeSteps = 200; // env episode length

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}// namespace rldemo

#endif//THREEPP_RL_PENDULUM_TRAINER_HPP
