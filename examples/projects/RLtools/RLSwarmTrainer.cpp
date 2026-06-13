// RLtools-isolated implementation of RLSwarmTrainer<OBS_DIM, ACT_DIM>.
//
// The ONLY TU that instantiates RLtools templates for the swarm demos. SAC is driven
// manually: external transitions go into the off-policy runner's replay buffers via
// rlt::add(), and the gradient updates replicate the SHARED_BATCH train half of
// rl/algorithms/sac/loop/core/operations_generic.h step(). The data-collection half
// (which would step RLtools' own env) is NOT used — the caller supplies transitions,
// so the learner is environment-agnostic. The trainer needs an RLtools env type only
// for its DIMS + State (its dynamics are bypassed), so we map dims -> a native env.

#include "RLSwarmTrainer.hpp"

#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_generic.h>
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

    constexpr int kM = 64;                // RLSwarmTrainer::kNumEnvs (replay-buffer / eval batch)
    constexpr long kSnapshotInterval = 25;// refresh the inference snapshot every N grad steps

    // Map (obs_dim, act_dim) -> a native RLtools environment. The env's DYNAMICS are
    // never used (the caller owns them) — only its DIMS + State type matter here.
    template<int OBS, int ACT>
    struct RLEnv;
    template<>
    struct RLEnv<3, 1> {
        using type = rlt::rl::environments::Pendulum<rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>>;
    };
    template<>
    struct RLEnv<6, 1> {
        using type = rlt::rl::environments::Acrobot<rlt::rl::environments::acrobot::Specification<T, TI, rlt::rl::environments::acrobot::DefaultParameters<T>>>;
    };

    template<typename ENVIRONMENT>
    struct Cfg {
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
            struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM> {
                static constexpr TI ACTOR_BATCH_SIZE = 100;
                static constexpr TI CRITIC_BATCH_SIZE = 100;
            };
            static constexpr TI N_ENVIRONMENTS = kM;
            static constexpr TI N_WARMUP_STEPS = 1000;
            static constexpr TI N_WARMUP_STEPS_CRITIC = 1000;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 1000;
            static constexpr TI STEP_LIMIT = 8000;
            static constexpr TI REPLAY_BUFFER_CAP = 10000;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
        };
        using LOOP_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
        using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;
        using SAC_PARAMS = typename LOOP_CORE_PARAMETERS::SAC_PARAMETERS;
    };

}// namespace

namespace rldemo {

    template<int OBS_DIM, int ACT_DIM>
    struct RLSwarmTrainer<OBS_DIM, ACT_DIM>::Impl {
        using ENVIRONMENT = typename RLEnv<OBS_DIM, ACT_DIM>::type;
        using CFG = Cfg<ENVIRONMENT>;
        using LOOP_CONFIG = typename CFG::LOOP_CONFIG;
        using LOOP_STATE = typename CFG::LOOP_STATE;
        using SAC_PARAMS = typename CFG::SAC_PARAMS;
        static constexpr int H = 64;

        using ObsMat = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS_DIM>>;
        using ActMat = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACT_DIM>>;
        using BatchInput = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, kM, OBS_DIM>>>;
        using BatchOutput = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, kM, ACT_DIM>>>;

        DEVICE trainDevice;
        DEVICE inferDevice;
        LOOP_STATE ts;

        decltype(ts.actor_critic.actor) snapshot;
        std::mutex mutex;

        BatchInput rolloutIn;
        BatchOutput rolloutOut;
        ObsMat addObs, addNextObs;
        ActMat addAction;
        std::mt19937 warmupRng;

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

    template<int OBS_DIM, int ACT_DIM>
    RLSwarmTrainer<OBS_DIM, ACT_DIM>::RLSwarmTrainer(unsigned int seed)
        : impl_(std::make_unique<Impl>(seed)) {}

    template<int OBS_DIM, int ACT_DIM>
    RLSwarmTrainer<OBS_DIM, ACT_DIM>::~RLSwarmTrainer() = default;

    template<int OBS_DIM, int ACT_DIM>
    void RLSwarmTrainer<OBS_DIM, ACT_DIM>::rolloutActions(const float* obs, float* actions, int m) {
        if (impl_->transitions.load() < Impl::CFG::LOOP_CORE_PARAMETERS::N_WARMUP_STEPS) {
            std::uniform_real_distribution<float> d(-1.f, 1.f);
            for (int i = 0; i < m * ACT_DIM; ++i) actions[i] = d(impl_->warmupRng);
            return;
        }
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < OBS_DIM; ++c)
                rlt::set(impl_->trainDevice, impl_->rolloutIn, obs[i * OBS_DIM + c], 0, i, c);
        rlt::evaluate(impl_->trainDevice, impl_->ts.actor_critic.actor, impl_->rolloutIn, impl_->rolloutOut,
                      impl_->ts.actor_buffers_eval, impl_->ts.rng, rlt::Mode<rlt::mode::Rollout<>>{});
        for (int i = 0; i < m; ++i)
            for (int a = 0; a < ACT_DIM; ++a)
                actions[i * ACT_DIM + a] = rlt::get(impl_->trainDevice, impl_->rolloutOut, 0, i, a);
    }

    template<int OBS_DIM, int ACT_DIM>
    void RLSwarmTrainer<OBS_DIM, ACT_DIM>::addTransitions(const float* obs, const float* actions, const float* rewards,
                                                          const float* nextObs, const unsigned char* truncated, int m) {
        auto& dev = impl_->trainDevice;
        for (int i = 0; i < m; ++i) {
            typename Impl::ENVIRONMENT::State st{};// cosmetic for Markovian SAC (gather never reads it)
            typename Impl::ENVIRONMENT::State nst{};
            for (int c = 0; c < OBS_DIM; ++c) {
                rlt::set(impl_->addObs, 0, c, obs[i * OBS_DIM + c]);
                rlt::set(impl_->addNextObs, 0, c, nextObs[i * OBS_DIM + c]);
            }
            for (int a = 0; a < ACT_DIM; ++a) rlt::set(impl_->addAction, 0, a, actions[i * ACT_DIM + a]);
            auto& rb = get(impl_->ts.off_policy_runner.replay_buffers, 0, i % kM);
            rlt::add(dev, rb, st, impl_->addObs, impl_->addObs, impl_->addAction, rewards[i],
                     nst, impl_->addNextObs, impl_->addNextObs, false, truncated[i] != 0);
        }
        impl_->transitions.fetch_add(m);
    }

    template<int OBS_DIM, int ACT_DIM>
    int RLSwarmTrainer<OBS_DIM, ACT_DIM>::update(int gradSteps) {
        if (impl_->transitions.load() < Impl::CFG::LOOP_CORE_PARAMETERS::N_WARMUP_STEPS_CRITIC) return 0;
        auto& dev = impl_->trainDevice;
        auto& ts = impl_->ts;
        using SAC_PARAMS = typename Impl::SAC_PARAMS;
        int performed = 0;
        for (int g = 0; g < gradSteps; ++g) {
            long step = impl_->gradStep.load();
            if (step >= Impl::CFG::LOOP_CORE_PARAMETERS::STEP_LIMIT) {
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

    template<int OBS_DIM, int ACT_DIM>
    void RLSwarmTrainer<OBS_DIM, ACT_DIM>::policyActionsDet(const float* obs, float* actions, int m) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < OBS_DIM; ++c)
                rlt::set(impl_->inferDevice, impl_->detIn, obs[i * OBS_DIM + c], 0, i, c);
        rlt::evaluate(impl_->inferDevice, impl_->snapshot, impl_->detIn, impl_->detOut,
                      impl_->detBuffer, impl_->detRng, rlt::Mode<rlt::mode::Evaluation<>>{});
        for (int i = 0; i < m; ++i)
            for (int a = 0; a < ACT_DIM; ++a)
                actions[i * ACT_DIM + a] = rlt::get(impl_->inferDevice, impl_->detOut, 0, i, a);
    }

    template<int OBS_DIM, int ACT_DIM>
    void RLSwarmTrainer<OBS_DIM, ACT_DIM>::extractWeights(float* out) const {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto& dev = impl_->inferDevice;
        const auto& mlp = impl_->snapshot.content;
        int p = 0;
        auto emit = [&](const auto& layer, int outDim, int inDim) {
            for (int o = 0; o < outDim; ++o)
                for (int i = 0; i < inDim; ++i)
                    out[p++] = static_cast<float>(rlt::get(dev, layer.weights.parameters, o, i));
            for (int o = 0; o < outDim; ++o)
                out[p++] = static_cast<float>(rlt::get(dev, layer.biases.parameters, o));
        };
        emit(mlp.input_layer, kHidden, OBS_DIM);
        emit(mlp.hidden_layers[0], kHidden, kHidden);
        emit(mlp.output_layer, 2 * ACT_DIM, kHidden);
    }

    template<int OBS_DIM, int ACT_DIM>
    long RLSwarmTrainer<OBS_DIM, ACT_DIM>::transitionsCollected() const { return impl_->transitions.load(); }
    template<int OBS_DIM, int ACT_DIM>
    bool RLSwarmTrainer<OBS_DIM, ACT_DIM>::pastWarmup() const { return impl_->transitions.load() >= Impl::CFG::LOOP_CORE_PARAMETERS::N_WARMUP_STEPS_CRITIC; }
    template<int OBS_DIM, int ACT_DIM>
    long RLSwarmTrainer<OBS_DIM, ACT_DIM>::gradientSteps() const { return impl_->gradStep.load(); }
    template<int OBS_DIM, int ACT_DIM>
    bool RLSwarmTrainer<OBS_DIM, ACT_DIM>::done() const { return impl_->done.load(); }
    template<int OBS_DIM, int ACT_DIM>
    long RLSwarmTrainer<OBS_DIM, ACT_DIM>::updateBudget() { return static_cast<long>(Cfg<typename RLEnv<OBS_DIM, ACT_DIM>::type>::LOOP_CORE_PARAMETERS::STEP_LIMIT); }
    template<int OBS_DIM, int ACT_DIM>
    long RLSwarmTrainer<OBS_DIM, ACT_DIM>::warmupTransitions() { return static_cast<long>(Cfg<typename RLEnv<OBS_DIM, ACT_DIM>::type>::LOOP_CORE_PARAMETERS::N_WARMUP_STEPS); }

    // ---- explicit instantiations (one per task's dims) ------------------------
    template class RLSwarmTrainer<3, 1>;// pendulum
    template class RLSwarmTrainer<6, 1>;// acrobot

}// namespace rldemo
