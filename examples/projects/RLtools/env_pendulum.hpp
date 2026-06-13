// PendulumEnv -- the classic 1-link swing-up, wrapped as an Env conforming to the
// swarm's Env concept (see env_acrobot.hpp for the concept summary). The dynamics live
// in the plain-C++ pendulum_env.hpp.

#ifndef THREEPP_ENV_PENDULUM_HPP
#define THREEPP_ENV_PENDULUM_HPP

#include "pendulum_env.hpp"

#include <cmath>

namespace rldemo {

    struct PendulumEnv {
        static constexpr int kObsDim = 3;
        static constexpr int kActDim = 1;
        static constexpr int kEpisodeSteps = rlenv::kEpisodeSteps;
        static constexpr int kRods = 1;       // renderable links
        static constexpr float kReach = rlenv::kL;// max extent from a pivot (camera framing)
        static constexpr const char* kName = "Pendulum";

        using State = rlenv::State;

        template<class RNG>
        static State sampleInitial(RNG& rng) { return rlenv::sampleInitial(rng); }
        static void observe(const State& s, float* obs) { rlenv::observe(s, obs); }
        static State step(const State& s, const float* act) { return rlenv::step(s, act[0]); }
        static float reward(const State& s, const float* act) { return rlenv::reward(s, act[0]); }

        // "solved" test for the deterministic learning curve.
        static bool upright(const State& s) {
            return std::fabs(rlenv::angleNormalize(s.theta)) < 0.3f && std::fabs(s.thetaDot) < 2.f;
        }

        // Rod r endpoints in the env's local XY frame (pivot at origin). theta=0 -> up.
        static void rod(const State& s, int, float& x0, float& y0, float& x1, float& y1) {
            x0 = 0.f;
            y0 = 0.f;
            x1 = std::sin(s.theta) * rlenv::kL;
            y1 = std::cos(s.theta) * rlenv::kL;
        }
    };

}// namespace rldemo

#endif//THREEPP_ENV_PENDULUM_HPP
