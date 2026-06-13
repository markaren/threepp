// RLSwarmTrainer — a plain-C++ facade over an RLtools (https://rl.tools) SAC
// *learner* fed by externally-collected, parallel rollouts.
//
// Unlike RLPendulumTrainer (which runs RLtools' canned loop on ONE internal env),
// this trainer is environment-agnostic: the caller steps M environments in
// parallel (on CPU now, on a GPU shader later) and feeds the resulting transitions
// here. The trainer only does: policy inference (batched), replay-buffer insertion,
// and SAC gradient updates. That boundary is the GPU seam — moving the rollout to
// Vulkan compute requires no change to this class.
//
// Threading: rolloutActions()/addTransitions()/update() are called from ONE thread
// (the collector/learner thread). policyActionsDet() is safe from another thread
// (the render thread) — it reads a periodically-published policy snapshot.

#ifndef THREEPP_RL_SWARM_TRAINER_HPP
#define THREEPP_RL_SWARM_TRAINER_HPP

#include <memory>

namespace rldemo {

    class RLSwarmTrainer {

    public:
        explicit RLSwarmTrainer(unsigned int seed = 1);
        ~RLSwarmTrainer();

        RLSwarmTrainer(const RLSwarmTrainer&) = delete;
        RLSwarmTrainer& operator=(const RLSwarmTrainer&) = delete;

        static constexpr int kNumEnvs = 64;// M parallel environments (compile-time)
        static constexpr int kObsDim = 3;
        static constexpr int kActDim = 1;
        static constexpr int kHidden = 64;// actor MLP hidden width
        // actor MLP weight count (flattened, per layer: W row-major [out][in], then b[out]):
        //   3->64, 64->64, 64->2  =  (3*64+64) + (64*64+64) + (2*64+2) = 4546
        static constexpr int kWeightCount = kObsDim * kHidden + kHidden + kHidden * kHidden + kHidden + 2 * kHidden + 2;
        static constexpr float kLogStdLo = -20.f, kLogStdHi = 2.f;// sample_and_squash bounds

        // [any thread] Snapshot the actor MLP weights into `out` (kWeightCount floats), in the
        // layout above — for uploading to a GPU rollout shader. Reads the published snapshot.
        void extractWeights(float* out) const;

        // [collector thread] Batched collection actions for m<=kNumEnvs envs.
        // While in warmup -> uniform random in [-1,1]; afterwards -> stochastic policy
        // (exploration). obs: m*3 row-major; actions: m out.
        void rolloutActions(const float* obs, float* actions, int m);

        // [collector thread] Ingest m transitions (caller owns env dynamics + reward).
        // obs/nextObs: m*3; actions/rewards: m; truncated: m (0/1). terminated is always
        // false for the pendulum.
        void addTransitions(const float* obs, const float* actions, const float* rewards,
                            const float* nextObs, const unsigned char* truncated, int m);

        // [collector thread] Run up to gradSteps SAC gradient steps. No-op until past
        // warmup or once the budget is exhausted. Returns the number actually performed.
        int update(int gradSteps);

        // [render thread] Deterministic batched policy eval on the published snapshot,
        // for the optional "showcase" visualization (m<=kNumEnvs).
        void policyActionsDet(const float* obs, float* actions, int m);

        [[nodiscard]] long transitionsCollected() const;
        [[nodiscard]] bool pastWarmup() const;// has training started?
        [[nodiscard]] long gradientSteps() const;
        [[nodiscard]] bool done() const;       // gradient-step budget exhausted

        static long updateBudget();  // total gradient-step budget
        static long warmupTransitions();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}// namespace rldemo

#endif//THREEPP_RL_SWARM_TRAINER_HPP
