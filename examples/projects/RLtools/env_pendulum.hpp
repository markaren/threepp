// PendulumEnv -- the classic 1-link swing-up as an Env conforming to the swarm's Env
// concept (see env_acrobot.hpp for the concept summary). The DYNAMICS live once in
// envdyn/pendulum.envdyn -- the very same file the GPU rollout shader compiles -- so
// the CPU learner/viz and the GPU rollout never diverge. This header is just the thin
// adapter that exposes that single source through the Env concept.

#ifndef THREEPP_ENV_PENDULUM_HPP
#define THREEPP_ENV_PENDULUM_HPP

#include "envdyn/env_dyn_compat.hpp"

#include <cmath>
#include <random>

namespace rldemo {
    namespace pendulum_dyn {
        using namespace rldemo::envmath;
#include "envdyn/pendulum.envdyn"
    }// namespace pendulum_dyn

    struct PendulumEnv {
        using State = pendulum_dyn::EnvState;
        static constexpr int kObsDim = pendulum_dyn::ENV_OBS_DIM;
        static constexpr int kActDim = pendulum_dyn::ENV_ACT_DIM;
        static constexpr int kStateDim = pendulum_dyn::ENV_STATE_DIM;// dynamics vars in the GPU state buffer
        static constexpr int kEpisodeSteps = pendulum_dyn::ENV_EPISODE_STEPS;
        static constexpr int kRods = 1;       // renderable links
        static constexpr float kReach = pendulum_dyn::ENV_L;// max extent from a pivot (camera framing)
        static constexpr float kDt = pendulum_dyn::ENV_DT;
        static constexpr const char* kName = "Pendulum";

        // initial state: theta ~ U[-PI,PI], thetaDot ~ U[-1,1] (RNG -> CPU-only)
        template<class RNG>
        static State sampleInitial(RNG& rng) {
            std::uniform_real_distribution<float> da(-pendulum_dyn::ENV_PI, pendulum_dyn::ENV_PI);
            std::uniform_real_distribution<float> dv(-1.f, 1.f);
            return State{da(rng), dv(rng)};
        }
        static void observe(const State& s, float* obs) {
            pendulum_dyn::EnvObs o = pendulum_dyn::envObserve(s);
            for (int i = 0; i < kObsDim; ++i) obs[i] = o.v[i];
        }
        static State step(const State& s, const float* act) { return pendulum_dyn::envStep(s, act); }
        static float reward(const State& s, const float* act) { return pendulum_dyn::envReward(s, act); }
        static bool upright(const State& s) { return pendulum_dyn::envUpright(s); }

        // Rod r endpoints in the env's local XY frame (pivot at origin). theta=0 -> up.
        static void rod(const State& s, int, float& x0, float& y0, float& x1, float& y1) {
            x0 = 0.f;
            y0 = 0.f;
            x1 = std::sin(s.theta) * pendulum_dyn::ENV_L;
            y1 = std::cos(s.theta) * pendulum_dyn::ENV_L;
        }

        // GPU rollout hooks: State <-> the flat per-env GPU state buffer (the layout the
        // shader's envLoad/envStore use -- same source, so they always agree).
        static void packState(const State& s, float* out) { pendulum_dyn::envStore(s, out); }
        static State unpackState(const float* in) { return pendulum_dyn::envLoad(in); }
    };

}// namespace rldemo

#endif//THREEPP_ENV_PENDULUM_HPP
