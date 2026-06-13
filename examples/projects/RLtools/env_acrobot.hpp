// AcrobotEnv -- the classic 2-link underactuated swing-up (torque on the elbow only),
// a genuinely different task (obs 6 vs 3, two links, harder) that drops into the SAME
// swarm + learner by changing one typedef. The DYNAMICS live once in
// envdyn/acrobot.envdyn -- the same file the GPU rollout shader compiles -- so the CPU
// and GPU paths never diverge. This header is the thin adapter over that single source.
//
// Env concept (what swarm.cpp / vulkan_rltools_swarm / the trainer expect):
//   using State; static constexpr int kObsDim, kActDim, kStateDim, kEpisodeSteps, kRods;
//   static constexpr float kReach, kDt;  static constexpr const char* kName;
//   template<RNG> static State sampleInitial(RNG&);
//   static void  observe(const State&, float* obs);     // obs[kObsDim]
//   static State step(const State&, const float* act);  // act[kActDim], in [-1,1]
//   static float reward(const State&, const float* act);
//   static bool  upright(const State&);                 // "solved" test for the curve
//   static void  rod(const State&, int r, float& x0,&y0,&x1,&y1); // link endpoints, local XY
//   static void  packState(const State&, float* out);   // State -> GPU buffer
//   static State unpackState(const float* in);          // GPU buffer -> State
// Adding a task = write one envdyn/<task>.envdyn + an adapter like this.

#ifndef THREEPP_ENV_ACROBOT_HPP
#define THREEPP_ENV_ACROBOT_HPP

#include "envdyn/env_dyn_compat.hpp"

#include <cmath>
#include <random>

namespace rldemo {
    namespace acrobot_dyn {
        using namespace rldemo::envmath;
#include "envdyn/acrobot.envdyn"
    }// namespace acrobot_dyn

    struct AcrobotEnv {
        using State = acrobot_dyn::EnvState;
        static constexpr int kObsDim = acrobot_dyn::ENV_OBS_DIM;
        static constexpr int kActDim = acrobot_dyn::ENV_ACT_DIM;
        static constexpr int kStateDim = acrobot_dyn::ENV_STATE_DIM;
        static constexpr int kEpisodeSteps = acrobot_dyn::ENV_EPISODE_STEPS;
        static constexpr int kRods = 2;
        static constexpr float kReach = acrobot_dyn::ENV_L1 + acrobot_dyn::ENV_L2;
        static constexpr float kDt = acrobot_dyn::ENV_DT;
        static constexpr const char* kName = "Acrobot";

        template<class RNG>
        static State sampleInitial(RNG& rng) {
            std::uniform_real_distribution<float> u(-0.1f, 0.1f);
            return State{u(rng), u(rng), u(rng), u(rng)};
        }
        static void observe(const State& s, float* obs) {
            acrobot_dyn::EnvObs o = acrobot_dyn::envObserve(s);
            for (int i = 0; i < kObsDim; ++i) obs[i] = o.v[i];
        }
        static State step(const State& s, const float* act) { return acrobot_dyn::envStep(s, act); }
        static float reward(const State& s, const float* act) { return acrobot_dyn::envReward(s, act); }
        static bool upright(const State& s) { return acrobot_dyn::envUpright(s); }

        // Link endpoints in local XY (pivot at origin). t1=0 -> link 1 hangs down.
        static void rod(const State& s, int r, float& x0, float& y0, float& x1, float& y1) {
            const float jx = acrobot_dyn::ENV_L1 * std::sin(s.t1), jy = -acrobot_dyn::ENV_L1 * std::cos(s.t1);// elbow
            if (r == 0) {
                x0 = 0.f;
                y0 = 0.f;
                x1 = jx;
                y1 = jy;
            } else {
                x0 = jx;
                y0 = jy;
                x1 = jx + acrobot_dyn::ENV_L2 * std::sin(s.t1 + s.t2);
                y1 = jy - acrobot_dyn::ENV_L2 * std::cos(s.t1 + s.t2);
            }
        }

        static void packState(const State& s, float* out) { acrobot_dyn::envStore(s, out); }
        static State unpackState(const float* in) { return acrobot_dyn::envLoad(in); }
    };

}// namespace rldemo

#endif//THREEPP_ENV_ACROBOT_HPP
