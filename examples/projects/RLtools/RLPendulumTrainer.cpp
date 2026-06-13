// RLtools-isolated implementation of RLPendulumTrainer.
//
// This is the ONLY translation unit that instantiates RLtools' template machinery.
// It configures a Soft-Actor-Critic (SAC) loop for the Pendulum swing-up task and
// exposes lock-free training + lock-guarded deterministic inference.
//
// Decoupling training from inference (the "snapshot" pattern):
//   The SAC training step rewrites the actor's weights and runs for ~milliseconds.
//   The render thread, meanwhile, needs to read those weights for a sub-millisecond
//   forward pass every frame. Guarding the whole training step with a mutex would
//   stall rendering. Instead, the trainer thread owns the live training state
//   lock-free and, every few steps, copies the actor into a shared `snapshot`
//   under a brief lock. The render thread evaluates that snapshot under the same
//   brief lock. (A same-type copy is required — copying into a different batch-size
//   actor trips RLtools' structure check — but a batch-100-typed actor evaluates
//   a batch-1 input just fine.)

#include "RLPendulumTrainer.hpp"

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

#include <atomic>
#include <mutex>

namespace rlt = rl_tools;

namespace {

    using DEVICE = rlt::devices::DefaultCPU;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using T = float;
    using TI = typename DEVICE::index_t;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;

    using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
    using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

    struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT> {
        struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM> {
            static constexpr TI ACTOR_BATCH_SIZE = 100;
            static constexpr TI CRITIC_BATCH_SIZE = 100;
        };
        static constexpr TI STEP_LIMIT = 5000;// SAC reliably solves the pendulum by here (verified); ~half the wait of 10k
        static constexpr TI ACTOR_NUM_LAYERS = 3;
        static constexpr TI ACTOR_HIDDEN_DIM = 64;
        static constexpr TI CRITIC_NUM_LAYERS = 3;
        static constexpr TI CRITIC_HIDDEN_DIM = 64;
    };

    using LOOP_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
    using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;
    using EVAL_ACTOR = LOOP_CONFIG::EVAL_ACTOR_TYPE;// batch-size-1 actor (used only for its Buffer type)

    static_assert(ENVIRONMENT::Observation::DIM == 3, "pendulum fourier observation is [cos, sin, thetaDot]");
    static_assert(ENVIRONMENT::ACTION_DIM == 1, "pendulum action is a scalar torque");

    constexpr long kSnapshotInterval = 25;// refresh the inference snapshot every N training steps (~0.2 s)

}// namespace

namespace rldemo {

    struct RLPendulumTrainer::Impl {
        DEVICE trainDevice;// owned by the trainer thread
        DEVICE inferDevice;// owned by the render thread
        LOOP_STATE ts;     // live training state (trainer thread, lock-free)

        // Shared inference snapshot of the actor (same type as the live actor so the
        // copy passes RLtools' structure check). Guarded by `mutex`.
        decltype(ts.actor_critic.actor) snapshot;
        std::mutex mutex;

        // Render-thread-private inference scratch.
        EVAL_ACTOR::template Buffer<true> evalBuffer;
        RNG inferRng;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, 3>>> input;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, 1>>> output;

        std::atomic<long> step{0};
        std::atomic<bool> done{false};

        explicit Impl(unsigned int seed) {
            rlt::malloc(trainDevice, ts);
            rlt::init(trainDevice, ts, seed);
            rlt::malloc(trainDevice, snapshot);
            rlt::malloc(inferDevice, evalBuffer);
            rlt::malloc(inferDevice, inferRng);
            rlt::init(inferDevice, inferRng, 0);
            rlt::malloc(inferDevice, input);
            rlt::malloc(inferDevice, output);
            // Seed the snapshot with the (random) initial policy so policyAction() is
            // always valid, even before the first training step.
            rlt::copy(trainDevice, trainDevice, ts.actor_critic.actor, snapshot);
        }

        ~Impl() {
            rlt::free(trainDevice, ts);
            rlt::free(trainDevice, snapshot);
            rlt::free(inferDevice, evalBuffer);
            rlt::free(inferDevice, inferRng);
            rlt::free(inferDevice, input);
            rlt::free(inferDevice, output);
        }
    };

    RLPendulumTrainer::RLPendulumTrainer(unsigned int seed)
        : impl_(std::make_unique<Impl>(seed)) {}

    RLPendulumTrainer::~RLPendulumTrainer() = default;

    bool RLPendulumTrainer::trainStep() {
        if (impl_->done.load()) return true;

        const bool finished = rlt::step(impl_->trainDevice, impl_->ts);
        const long s = static_cast<long>(impl_->ts.step);
        impl_->step.store(s);

        if (s % kSnapshotInterval == 0 || finished) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            rlt::copy(impl_->trainDevice, impl_->trainDevice, impl_->ts.actor_critic.actor, impl_->snapshot);
        }
        if (finished) impl_->done.store(true);
        return finished;
    }

    float RLPendulumTrainer::policyAction(float cosTheta, float sinTheta, float thetaDot) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        rlt::set(impl_->inferDevice, impl_->input, cosTheta, 0, 0, 0);
        rlt::set(impl_->inferDevice, impl_->input, sinTheta, 0, 0, 1);
        rlt::set(impl_->inferDevice, impl_->input, thetaDot, 0, 0, 2);
        rlt::evaluate(impl_->inferDevice, impl_->snapshot, impl_->input, impl_->output,
                      impl_->evalBuffer, impl_->inferRng, rlt::Mode<rlt::mode::Evaluation<>>{});
        return rlt::get(impl_->inferDevice, impl_->output, 0, 0, 0);
    }

    void RLPendulumTrainer::reset(unsigned int seed) {
        rlt::free(impl_->trainDevice, impl_->ts);
        rlt::malloc(impl_->trainDevice, impl_->ts);
        rlt::init(impl_->trainDevice, impl_->ts, seed);
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            rlt::copy(impl_->trainDevice, impl_->trainDevice, impl_->ts.actor_critic.actor, impl_->snapshot);
        }
        impl_->step.store(0);
        impl_->done.store(false);
    }

    long RLPendulumTrainer::stepCount() const { return impl_->step.load(); }
    bool RLPendulumTrainer::done() const { return impl_->done.load(); }
    long RLPendulumTrainer::stepLimit() { return static_cast<long>(LOOP_CORE_PARAMETERS::STEP_LIMIT); }
    long RLPendulumTrainer::warmupSteps() { return static_cast<long>(LOOP_CORE_PARAMETERS::N_WARMUP_STEPS); }

}// namespace rldemo
