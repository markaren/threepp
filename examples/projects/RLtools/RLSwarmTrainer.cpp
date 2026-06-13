// RLtools-isolated implementation of RLSwarmTrainer (the parallel-rollout learner).
//
// This is the ONLY translation unit that instantiates RLtools templates for the
// swarm demo. It drives SAC manually: external transitions are inserted into the
// off-policy runner's replay buffers via rlt::add(), and the gradient updates
// replicate the *training half* of rl/algorithms/sac/loop/core/operations_generic.h
// step() verbatim (SHARED_BATCH path). The data-collection half (which would step
// RLtools' own internal env) is intentionally NOT used — the caller supplies the
// transitions, which is exactly the seam where a GPU rollout later plugs in.

#include "RLSwarmTrainer.hpp"

#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
#include <rl_tools/rl/components/replay_buffer/operations_generic.h>

#include <atomic>
#include <cmath>
#include <mutex>
#include <random>

namespace rlt = rl_tools;

namespace {

    using DEVICE = rlt::devices::DefaultCPU;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using T = float;
    using TI = typename DEVICE::index_t;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;

    using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
    using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

    constexpr int kM = rldemo::RLSwarmTrainer::kNumEnvs;// 64

    struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
        struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM> {
            static constexpr TI ACTOR_BATCH_SIZE = 100;
            static constexpr TI CRITIC_BATCH_SIZE = 100;
        };
        static constexpr TI N_ENVIRONMENTS = kM;       // M parallel replay buffers
        static constexpr TI N_WARMUP_STEPS = 1000;      // random-action transitions before training
        static constexpr TI N_WARMUP_STEPS_CRITIC = 1000;
        static constexpr TI N_WARMUP_STEPS_ACTOR = 1000;
        static constexpr TI STEP_LIMIT = 8000;          // gradient-step budget
        static constexpr TI REPLAY_BUFFER_CAP = 10000;  // per-env capacity (must be set explicitly)
        static constexpr TI ACTOR_NUM_LAYERS = 3;
        static constexpr TI ACTOR_HIDDEN_DIM = 64;
        static constexpr TI CRITIC_NUM_LAYERS = 3;
        static constexpr TI CRITIC_HIDDEN_DIM = 64;
    };

    using LOOP_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
    using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;
    using SAC_PARAMS = LOOP_CORE_PARAMETERS::SAC_PARAMETERS;

    static_assert(ENVIRONMENT::Observation::DIM == 3 && ENVIRONMENT::ACTION_DIM == 1);

    constexpr long kSnapshotInterval = 25;// refresh deterministic-viz snapshot every N grad steps

    using ObsMat = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 3>>;
    using ActMat = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1>>;
    using BatchInput = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, kM, 3>>>;
    using BatchOutput = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, kM, 1>>>;

}// namespace

namespace rldemo {

    struct RLSwarmTrainer::Impl {
        DEVICE trainDevice;// collector/learner thread
        DEVICE inferDevice;// render thread (policyActionsDet)
        LOOP_STATE ts;     // owned by the collector/learner thread

        // shared deterministic-viz snapshot (same type as the live actor)
        decltype(ts.actor_critic.actor) snapshot;
        std::mutex mutex;

        // collector-thread rollout scratch (reuses ts.actor_buffers_eval as the eval buffer)
        BatchInput rolloutIn;
        BatchOutput rolloutOut;
        ObsMat addObs, addNextObs;
        ActMat addAction;
        std::mt19937 warmupRng;

        // render-thread deterministic scratch
        typename LOOP_CONFIG::EVAL_ACTOR_TYPE::template Buffer<true> detBuffer;
        RNG detRng;
        BatchInput detIn;
        BatchOutput detOut;

        std::atomic<long> transitions{0};
        std::atomic<long> gradStep{0};
        std::atomic<bool> done{false};

        explicit Impl(unsigned int seed)
            : warmupRng(seed) {
            rlt::malloc(trainDevice, ts);
            rlt::init(trainDevice, ts, seed);
            rlt::malloc(trainDevice, snapshot);
            rlt::malloc(trainDevice, rolloutIn);
            rlt::malloc(trainDevice, rolloutOut);
            rlt::malloc(trainDevice, addObs);
            rlt::malloc(trainDevice, addNextObs);
            rlt::malloc(trainDevice, addAction);
            rlt::malloc(inferDevice, detBuffer);
            rlt::malloc(inferDevice, detRng);
            rlt::init(inferDevice, detRng, 0);
            rlt::malloc(inferDevice, detIn);
            rlt::malloc(inferDevice, detOut);
            rlt::copy(trainDevice, trainDevice, ts.actor_critic.actor, snapshot);
        }
        ~Impl() {
            rlt::free(trainDevice, ts);
            rlt::free(trainDevice, snapshot);
            rlt::free(trainDevice, rolloutIn);
            rlt::free(trainDevice, rolloutOut);
            rlt::free(trainDevice, addObs);
            rlt::free(trainDevice, addNextObs);
            rlt::free(trainDevice, addAction);
            rlt::free(inferDevice, detBuffer);
            rlt::free(inferDevice, detRng);
            rlt::free(inferDevice, detIn);
            rlt::free(inferDevice, detOut);
        }
    };

    RLSwarmTrainer::RLSwarmTrainer(unsigned int seed)
        : impl_(std::make_unique<Impl>(seed)) {}
    RLSwarmTrainer::~RLSwarmTrainer() = default;

    void RLSwarmTrainer::rolloutActions(const float* obs, float* actions, int m) {
        if (impl_->transitions.load() < LOOP_CORE_PARAMETERS::N_WARMUP_STEPS) {
            std::uniform_real_distribution<float> d(-1.f, 1.f);
            for (int i = 0; i < m; ++i) actions[i] = d(impl_->warmupRng);
            return;
        }
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < 3; ++c)
                rlt::set(impl_->trainDevice, impl_->rolloutIn, obs[i * 3 + c], 0, i, c);
        rlt::evaluate(impl_->trainDevice, impl_->ts.actor_critic.actor, impl_->rolloutIn, impl_->rolloutOut,
                      impl_->ts.actor_buffers_eval, impl_->ts.rng, rlt::Mode<rlt::mode::Rollout<>>{});
        for (int i = 0; i < m; ++i) actions[i] = rlt::get(impl_->trainDevice, impl_->rolloutOut, 0, i, 0);
    }

    void RLSwarmTrainer::addTransitions(const float* obs, const float* actions, const float* rewards,
                                        const float* nextObs, const unsigned char* truncated, int m) {
        auto& dev = impl_->trainDevice;
        for (int i = 0; i < m; ++i) {
            const float c = obs[i * 3], s = obs[i * 3 + 1], wd = obs[i * 3 + 2];
            const float nc = nextObs[i * 3], ns = nextObs[i * 3 + 1], nwd = nextObs[i * 3 + 2];
            typename ENVIRONMENT::State st;
            st.theta = std::atan2(s, c);
            st.theta_dot = wd;
            typename ENVIRONMENT::State nst;
            nst.theta = std::atan2(ns, nc);
            nst.theta_dot = nwd;
            rlt::set(impl_->addObs, 0, 0, c);
            rlt::set(impl_->addObs, 0, 1, s);
            rlt::set(impl_->addObs, 0, 2, wd);
            rlt::set(impl_->addNextObs, 0, 0, nc);
            rlt::set(impl_->addNextObs, 0, 1, ns);
            rlt::set(impl_->addNextObs, 0, 2, nwd);
            rlt::set(impl_->addAction, 0, 0, actions[i]);
            auto& rb = get(impl_->ts.off_policy_runner.replay_buffers, 0, i % kM);
            rlt::add(dev, rb, st, impl_->addObs, impl_->addObs, impl_->addAction, rewards[i],
                     nst, impl_->addNextObs, impl_->addNextObs, false, truncated[i] != 0);
        }
        impl_->transitions.fetch_add(m);
    }

    int RLSwarmTrainer::update(int gradSteps) {
        if (impl_->transitions.load() < LOOP_CORE_PARAMETERS::N_WARMUP_STEPS_CRITIC) return 0;
        auto& dev = impl_->trainDevice;
        auto& ts = impl_->ts;
        int performed = 0;
        for (int g = 0; g < gradSteps; ++g) {
            long step = impl_->gradStep.load();
            if (step >= LOOP_CORE_PARAMETERS::STEP_LIMIT) {
                impl_->done.store(true);
                break;
            }
            const bool trainCritic = step % SAC_PARAMS::CRITIC_TRAINING_INTERVAL == 0;
            const bool updateTargets = step % SAC_PARAMS::CRITIC_TARGET_UPDATE_INTERVAL == 0;
            const bool trainActor = step % SAC_PARAMS::ACTOR_TRAINING_INTERVAL == 0;

            if (trainCritic || trainActor) {
                rlt::gather_batch(dev, ts.off_policy_runner, ts.critic_batch, ts.rng);
                rlt::randn(dev, ts.action_noise_critic, ts.rng);
            }
            if (trainCritic) {
                for (int ci = 0; ci < 2; ++ci) {
                    rlt::train_critic(dev, ts.actor_critic, ts.actor_critic.critics[ci], ts.critic_batch,
                                      ts.actor_critic.critic_optimizers[ci], ts.actor_target_buffers[ci],
                                      ts.critic_buffers[ci], ts.critic_target_buffers[ci],
                                      ts.critic_training_buffers[ci], ts.action_noise_critic, ts.rng);
                }
            }
            if (updateTargets) rlt::update_critic_targets(dev, ts.actor_critic);
            if (trainActor) {
                rlt::randn(dev, ts.action_noise_actor, ts.rng);
                rlt::train_actor(dev, ts.actor_critic, ts.critic_batch, ts.actor_critic.actor_optimizer,
                                 ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers,
                                 ts.action_noise_actor, ts.rng);
            }

            const long ns = step + 1;
            impl_->gradStep.store(ns);
            ++performed;
            if (ns % kSnapshotInterval == 0) {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                rlt::copy(dev, dev, ts.actor_critic.actor, impl_->snapshot);
            }
        }
        return performed;
    }

    void RLSwarmTrainer::policyActionsDet(const float* obs, float* actions, int m) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < 3; ++c)
                rlt::set(impl_->inferDevice, impl_->detIn, obs[i * 3 + c], 0, i, c);
        rlt::evaluate(impl_->inferDevice, impl_->snapshot, impl_->detIn, impl_->detOut,
                      impl_->detBuffer, impl_->detRng, rlt::Mode<rlt::mode::Evaluation<>>{});
        for (int i = 0; i < m; ++i) actions[i] = rlt::get(impl_->inferDevice, impl_->detOut, 0, i, 0);
    }

    void RLSwarmTrainer::extractWeights(float* out) const {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto& dev = impl_->inferDevice;
        const auto& mlp = impl_->snapshot.content;// sequential: content = MLP, next_module = sample_and_squash
        int p = 0;
        auto emit = [&](const auto& layer, int outDim, int inDim) {
            for (int o = 0; o < outDim; ++o)
                for (int i = 0; i < inDim; ++i)
                    out[p++] = static_cast<float>(rlt::get(dev, layer.weights.parameters, o, i));
            for (int o = 0; o < outDim; ++o)
                out[p++] = static_cast<float>(rlt::get(dev, layer.biases.parameters, o));
        };
        emit(mlp.input_layer, kHidden, kObsDim);
        emit(mlp.hidden_layers[0], kHidden, kHidden);
        emit(mlp.output_layer, 2 * kActDim, kHidden);
    }

    long RLSwarmTrainer::transitionsCollected() const { return impl_->transitions.load(); }
    bool RLSwarmTrainer::pastWarmup() const { return impl_->transitions.load() >= LOOP_CORE_PARAMETERS::N_WARMUP_STEPS_CRITIC; }
    long RLSwarmTrainer::gradientSteps() const { return impl_->gradStep.load(); }
    bool RLSwarmTrainer::done() const { return impl_->done.load(); }
    long RLSwarmTrainer::updateBudget() { return static_cast<long>(LOOP_CORE_PARAMETERS::STEP_LIMIT); }
    long RLSwarmTrainer::warmupTransitions() { return static_cast<long>(LOOP_CORE_PARAMETERS::N_WARMUP_STEPS); }

}// namespace rldemo
