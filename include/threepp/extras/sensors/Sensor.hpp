// Proprioceptive sensor suite — shared base contract.
//
// A Sensor rides the scene graph: it is attached to an Object3D and its
// measurement frame IS that node's world frame. Sensors are push-based —
// PhysxWorld drives them from its fixed-timestep step loop (see
// PhysxWorld::registerSensor), so each measurement is taken the instant the
// physics state is fresh (after fetchResults) and timestamped with the
// accumulated simulation time. This is the fidelity a future ArduPilot SITL
// lock-step integration needs: one clean sample per (sub)step.
//
// Read side is non-blocking and decoupled from the physics thread cadence:
//   latest()  — the most recent measurement (a snapshot; survives drain()).
//   drain(out) — moves every measurement accumulated since the last drain out
//                of a bounded ring buffer (oldest dropped on overflow).
//
// The generic machinery here (rate gating, the ring buffer, the reusable
// Gaussian NoiseModel) is deliberately sensor-agnostic so later sensors
// (JointEncoder, ForceTorque, GPS, …) reuse it. Only Sensor::sample() is
// sensor-specific.

#ifndef THREEPP_SENSORS_SENSOR_HPP
#define THREEPP_SENSORS_SENSOR_HPP

#include "threepp/math/Vector3.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace physx {
    class PxRigidActor;// forward-declared so this header stays PhysX-include-free
}

namespace threepp {

    class Object3D;
    class PhysxWorld;// forward-declared: only referenced by reference in the hooks below

    // ---------------------------------------------------------------------------
    // Deterministic PRNG (SplitMix64) + Gaussian draw. Self-contained so a fixed
    // seed reproduces bit-for-bit regardless of the platform's <random>
    // implementation (std::normal_distribution is NOT portable across stdlibs).
    // ---------------------------------------------------------------------------
    class SplitMix64 {
    public:
        explicit SplitMix64(std::uint64_t seed = 0) : state_(seed) {}

        std::uint64_t next() {
            std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        }

        // Uniform in [0, 1).
        double nextDouble() {
            // Top 53 bits -> double with 53-bit mantissa resolution.
            return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
        }

    private:
        std::uint64_t state_;
    };

    /**
     * Per-axis noise configuration shared by every sensor. All densities are
     * continuous-time so a measurement's noise is invariant to the sampling
     * rate (a real IMU spec sheet is quoted this way):
     *
     *   whiteNoiseDensity  units [X]/sqrt(Hz)      (e.g. gyro rad/s/sqrt(Hz))
     *       per-sample stddev = density / sqrt(dt)
     *   randomWalk         units [X]/(s*sqrt(Hz))  (bias-instability drive)
     *       per-sample bias increment stddev = randomWalk * sqrt(dt)
     *   constantBias       units [X]               (fixed turn-on offset)
     *
     * where [X] is the measured quantity's unit and dt the seconds since the
     * previous sample. Determinism: given `seed` and the sequence of apply()
     * calls, the produced noise is identical across runs.
     */
    struct NoiseModel {
        Vector3 whiteNoiseDensity{0.f, 0.f, 0.f};
        Vector3 randomWalk{0.f, 0.f, 0.f};
        Vector3 constantBias{0.f, 0.f, 0.f};
        std::uint64_t seed = 0;
    };

    /**
     * Stateful applier for a NoiseModel: owns the RNG stream and the current
     * random-walked bias. Reusable by any sensor axis-triplet. Deterministic
     * for a given model.seed + apply() call sequence. A zero model (all vectors
     * zero) returns the clean value bit-for-bit — the "perfect sensor" the
     * physics-truth tests rely on.
     */
    class GaussianNoise {
    public:
        GaussianNoise() = default;
        explicit GaussianNoise(const NoiseModel& model) { reset(model); }

        void reset(const NoiseModel& model) {
            model_ = model;
            rng_ = SplitMix64{model.seed};
            bias_.set(0.f, 0.f, 0.f);
            haveSpare_ = false;
        }

        // Corrupt a clean per-axis measurement sampled over `dt` seconds.
        Vector3 apply(const Vector3& clean, double dt) {
            const double sqrtDt = (dt > 0.0) ? std::sqrt(dt) : 0.0;
            Vector3 out = clean;
            for (int i = 0; i < 3; ++i) {
                // Integrate the bias random walk first (b_{k+1} = b_k + rw*sqrt(dt)*N).
                if (model_.randomWalk[i] != 0.f && sqrtDt > 0.0) {
                    bias_[i] += static_cast<float>(model_.randomWalk[i] * sqrtDt * gaussian());
                }
                float white = 0.f;
                if (model_.whiteNoiseDensity[i] != 0.f && sqrtDt > 0.0) {
                    white = static_cast<float>(model_.whiteNoiseDensity[i] / sqrtDt * gaussian());
                }
                out[i] += model_.constantBias[i] + bias_[i] + white;
            }
            return out;
        }

        // Current random-walked bias (excludes the constant term). For debugging.
        [[nodiscard]] const Vector3& bias() const { return bias_; }

    private:
        // Standard-normal draw via Box-Muller; caches the paired value.
        double gaussian() {
            if (haveSpare_) {
                haveSpare_ = false;
                return spare_;
            }
            const double u1 = rng_.nextDouble();// [0, 1)
            const double u2 = rng_.nextDouble();
            const double r = std::sqrt(-2.0 * std::log(1.0 - u1));// 1-u1 in (0, 1]
            const double theta = 6.283185307179586 * u2;
            spare_ = r * std::sin(theta);
            haveSpare_ = true;
            return r * std::cos(theta);
        }

        NoiseModel model_{};
        SplitMix64 rng_{0};
        Vector3 bias_{0.f, 0.f, 0.f};
        double spare_ = 0.0;
        bool haveSpare_ = false;
    };

    /**
     * Bounded single-producer/single-consumer ring of measurements. Fixed
     * capacity; when full the oldest sample is dropped so the newest always
     * lands (a stalled reader never blocks the physics step). Not thread-safe —
     * PhysxWorld drives push() and the caller drains from the same context.
     */
    template<class T>
    class SensorRing {
    public:
        explicit SensorRing(std::size_t capacity = 2048)
            : cap_(capacity ? capacity : 1), buf_(cap_) {}

        void push(const T& v) {
            buf_[head_] = v;
            head_ = (head_ + 1) % cap_;
            if (size_ < cap_) {
                ++size_;
            } else {
                tail_ = (tail_ + 1) % cap_;// overwrote the oldest
            }
            last_ = v;
            hasLast_ = true;
        }

        [[nodiscard]] bool empty() const { return size_ == 0; }
        [[nodiscard]] std::size_t size() const { return size_; }
        [[nodiscard]] std::size_t capacity() const { return cap_; }

        // Most recent sample ever pushed; persists across drain(). nullopt until
        // the first push (or after clear()).
        [[nodiscard]] std::optional<T> latest() const {
            return hasLast_ ? std::optional<T>(last_) : std::nullopt;
        }

        // Move all buffered samples (oldest-first) into out; empties the ring.
        void drain(std::vector<T>& out) {
            out.clear();
            out.reserve(size_);
            for (std::size_t i = 0, idx = tail_; i < size_; ++i, idx = (idx + 1) % cap_) {
                out.push_back(buf_[idx]);
            }
            tail_ = head_;
            size_ = 0;
        }

        void clear() {
            head_ = tail_ = size_ = 0;
            hasLast_ = false;
        }

    private:
        std::size_t cap_;
        std::vector<T> buf_;
        std::size_t head_ = 0;
        std::size_t tail_ = 0;
        std::size_t size_ = 0;
        T last_{};
        bool hasLast_ = false;
    };

    /**
     * Base class for a scene-graph-attached, physics-driven sensor.
     *
     * Ownership: the sensor holds a NON-owning pointer to its attachment node;
     * that node must outlive the sensor. Register the sensor with a PhysxWorld
     * (world.registerSensor(&s)) and it is sampled from the world's step loop;
     * unregister (or destroy the world) before the sensor dies.
     */
    class Sensor {
    public:
        explicit Sensor(Object3D& node, double rateHz = 0.0)
            : node_(&node), rateHz_(rateHz) {}

        virtual ~Sensor() = default;

        Sensor(const Sensor&) = delete;
        Sensor& operator=(const Sensor&) = delete;

        [[nodiscard]] Object3D* node() const { return node_; }

        // Target sample rate in Hz. 0 (default) = sample every physics substep.
        [[nodiscard]] double rateHz() const { return rateHz_; }
        void setRateHz(double hz) { rateHz_ = hz; }

        // Hooks invoked by PhysxWorld::registerSensor / unregisterSensor. The
        // default is a no-op; a concrete sensor resolves its rigid body / caches
        // world state here and should throw on an invalid attachment (so the
        // error surfaces at registration, not mid-step).
        virtual void onRegister(PhysxWorld& /*world*/) {}
        virtual void onUnregister() {}

        // Called by PhysxWorld::removeActor just BEFORE the actor is released. A
        // sensor that cached this actor at registration must drop it here —
        // otherwise the next substep samples freed memory. The sensor stays
        // registered and simply goes quiet, mirroring how a removed actor leaves
        // its InstancedMesh slot nulled rather than reshuffling the list: whether
        // to unregister or re-attach is the caller's call.
        virtual void onActorRemoved(::physx::PxRigidActor* /*actor*/) {}

        // Driven by PhysxWorld once per fixed substep with the substep dt and the
        // accumulated sim time. Rate-gates, then calls sample() with the true
        // elapsed time since the previous emitted sample (so a sensor running
        // slower than the physics rate still finite-differences over the correct
        // interval).
        //
        // The gate schedules against a fixed period accumulated from the first
        // sample, NOT against the time the previous sample landed on. Samples can
        // only land on substep boundaries, so chasing the actual emission time
        // rounds every interval up to a whole substep and the error compounds:
        // at 240 Hz physics a 100 Hz request (period 10 ms) would fire every 3rd
        // substep = 12.5 ms = 80 Hz. Accumulating the due time instead keeps the
        // long-run average exact and leaves only sub-substep jitter, which is
        // unavoidable.
        void tick(double dt, double simTime) {
            const bool gated = rateHz_ > 0.0;
            if (gated && hasLast_ && simTime + 1e-9 < nextDue_) return;

            const double sampleDt = hasLast_ ? (simTime - lastSampleTime_) : dt;
            sample(sampleDt, simTime);
            lastSampleTime_ = simTime;

            if (gated) {
                const double period = 1.0 / rateHz_;
                nextDue_ = hasLast_ ? nextDue_ + period : simTime + period;
                // If the physics rate is slower than the requested rate we can
                // never catch up; resync rather than let the due time fall
                // arbitrarily far behind (which would pin the gate open and, on
                // a later slowdown, burst).
                if (nextDue_ <= simTime) nextDue_ = simTime + period;
            }
            hasLast_ = true;
        }

        // Produce one measurement covering `dt` seconds, timestamped `simTime`.
        virtual void sample(double dt, double simTime) = 0;

    protected:
        // Concrete sensors call this on reset so the next tick behaves like the
        // first (no finite-difference spike, rate gate re-armed).
        void resetTiming() {
            hasLast_ = false;
            lastSampleTime_ = 0.0;
            nextDue_ = 0.0;
        }

    private:
        Object3D* node_;
        double rateHz_;
        bool hasLast_ = false;
        double lastSampleTime_ = 0.0;
        double nextDue_ = 0.0;// next scheduled sample time (rate-gated sensors)
    };

}// namespace threepp

#endif// THREEPP_SENSORS_SENSOR_HPP
