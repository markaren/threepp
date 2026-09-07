// RLSwarmTrainer<OBS_DIM, ACT_DIM> — a plain-C++ facade over an RLtools SAC learner
// fed by externally-collected parallel rollouts, generalized over observation/action
// dimensions so the SAME learner trains ANY continuous-control task (pendulum,
// cartpole, acrobot, ...). The environment dynamics live outside (in the env headers
// / a GPU shader); this class only does policy inference + replay-buffer insertion +
// SAC gradient updates. ALL RLtools template machinery is confined to the .cpp;
// instantiations are provided explicitly there (one per task's dims).
//
// Threading: rolloutActions()/addTransitions()/update() from ONE thread; the rest are
// safe from another thread (they read a periodically-published policy snapshot).

#ifndef THREEPP_RL_SWARM_TRAINER_HPP
#define THREEPP_RL_SWARM_TRAINER_HPP

#include <memory>

namespace rldemo {

    template<int OBS_DIM, int ACT_DIM>
    class RLSwarmTrainer {

    public:
        explicit RLSwarmTrainer(unsigned int seed = 1);
        ~RLSwarmTrainer();

        RLSwarmTrainer(const RLSwarmTrainer&) = delete;
        RLSwarmTrainer& operator=(const RLSwarmTrainer&) = delete;

        static constexpr int kNumEnvs = 64;// M parallel environments (replay-buffer count)
        static constexpr int kObsDim = OBS_DIM;
        static constexpr int kActDim = ACT_DIM;
        static constexpr int kHidden = 64;// actor MLP hidden width
        // flattened actor weight count (per layer: W row-major [out][in], then b[out]):
        //   OBS->64, 64->64, 64->2*ACT
        static constexpr int kWeightCount =
                OBS_DIM * kHidden + kHidden + kHidden * kHidden + kHidden + 2 * ACT_DIM * kHidden + 2 * ACT_DIM;
        static constexpr float kLogStdLo = -20.f, kLogStdHi = 2.f;

        // [collector thread] Batched collection actions for m<=kNumEnvs envs. Warmup ->
        // uniform random in [-1,1]; afterwards -> stochastic policy. obs: m*OBS_DIM; actions: m*ACT_DIM.
        void rolloutActions(const float* obs, float* actions, int m);

        // [collector thread] Ingest m transitions (caller owns dynamics + reward).
        void addTransitions(const float* obs, const float* actions, const float* rewards,
                            const float* nextObs, const unsigned char* truncated, int m);

        // [collector thread] Up to gradSteps SAC gradient steps (no-op until past warmup / done).
        int update(int gradSteps);

        // [render thread] Deterministic batched eval on the published snapshot.
        void policyActionsDet(const float* obs, float* actions, int m);

        // [any thread] Flatten the actor MLP weights into `out` (kWeightCount floats) for a GPU shader.
        void extractWeights(float* out) const;

        [[nodiscard]] long transitionsCollected() const;
        [[nodiscard]] bool pastWarmup() const;
        [[nodiscard]] long gradientSteps() const;
        [[nodiscard]] bool done() const;

        static long updateBudget();
        static long warmupTransitions();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // Explicit instantiations live in the .cpp (one per task's dims). To add a task
    // whose dims aren't a native RLtools env, add a generic-env mapping in the .cpp.
    extern template class RLSwarmTrainer<3, 1>;// pendulum
    extern template class RLSwarmTrainer<6, 1>;// acrobot

}// namespace rldemo

#endif//THREEPP_RL_SWARM_TRAINER_HPP
