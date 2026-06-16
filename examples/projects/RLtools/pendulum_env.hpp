// Plain-C++ Pendulum-v1 dynamics, byte-for-byte matching RLtools'
// rl::environments::pendulum (DefaultParameters). Shared by the headless learner
// harness and the swarm demo so the simulated environments never diverge from
// what the policy trains on. No RLtools / no threepp dependency — this is the
// "environment" half of the hybrid (the part that later moves to a GPU shader).

#ifndef THREEPP_RL_PENDULUM_ENV_HPP
#define THREEPP_RL_PENDULUM_ENV_HPP

#include <cmath>
#include <random>

namespace rlenv {

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kG = 10.0f, kL = 1.0f, kM = 1.0f, kDt = 0.05f;
    constexpr float kMaxTorque = 2.0f, kMaxSpeed = 8.0f;
    constexpr int kEpisodeSteps = 200;// EPISODE_STEP_LIMIT

    struct State {
        float theta = kPi;   // 0 = upright, +/-PI = hanging down
        float thetaDot = 0.f;
    };

    inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

    // angle_normalize (python-style modulo), matching pendulum/operations_generic.h
    inline float angleNormalize(float x) {
        const float twoPi = 2 * kPi;
        float m = std::fmod(x + kPi, twoPi);
        if (m < 0) m += twoPi;
        return m - kPi;
    }

    // Fourier observation [cos(theta), sin(theta), thetaDot]
    inline void observe(const State& s, float* out3) {
        out3[0] = std::cos(s.theta);
        out3[1] = std::sin(s.theta);
        out3[2] = s.thetaDot;
    }

    // Reward from the PRE-step state + action (action in [-1,1]); matches reward()
    inline float reward(const State& s, float action) {
        const float u = kMaxTorque * action;
        const float an = angleNormalize(s.theta);
        return -(an * an + 0.1f * s.thetaDot * s.thetaDot + 0.001f * (u * u));
    }

    // One integration step; matches pendulum step()
    inline State step(const State& s, float action) {
        const float u = kMaxTorque * clampf(action, -1.f, 1.f);
        float newThetaDot = s.thetaDot + (3.f * kG / (2.f * kL) * std::sin(s.theta) + 3.f / (kM * kL * kL) * u) * kDt;
        newThetaDot = clampf(newThetaDot, -kMaxSpeed, kMaxSpeed);
        State n;
        n.thetaDot = newThetaDot;
        n.theta = s.theta + newThetaDot * kDt;
        return n;
    }

    // Initial state: theta ~ U[-PI,PI], thetaDot ~ U[-1,1] (sample_initial_state)
    template<class RNG>
    inline State sampleInitial(RNG& rng) {
        std::uniform_real_distribution<float> da(-kPi, kPi);
        std::uniform_real_distribution<float> dv(-1.f, 1.f);
        return State{da(rng), dv(rng)};
    }

}// namespace rlenv

#endif//THREEPP_RL_PENDULUM_ENV_HPP
