// Joint position/velocity encoder — the proprioceptive sensor a robot arm or a
// legged robot reports its configuration with (Gazebo's joint-state sensor,
// ROS's sensor_msgs/JointState minus effort).
//
// Articulation already exposes exact joint positions via jointPositions(). What
// this adds is the part that makes it a SENSOR rather than a getter:
//
//   quantization  a real encoder reports whole ticks, not a real number. This
//                 is the dominant error term at low speed and it is what makes
//                 differentiated velocity noisy near zero — the classic
//                 "velocity chatters at standstill" behaviour that a controller
//                 tuned against exact state will not survive on hardware.
//   noise         per-sample white noise + bias random walk, shared with the
//                 rest of the suite (see NoiseModel).
//   rate gating   encoders are read on a bus at a fixed rate, usually well below
//                 the physics rate.
//   buffering     latest()/drain(), so a controller running at its own cadence
//                 sees every reading rather than the last one.
//
// The measured quantity is the inbound joint of an articulation link: an angle
// in radians for a revolute joint, a displacement in metres for a prismatic one.
// Units follow the joint, so `resolution` is rad/tick or m/tick to match.

#ifndef THREEPP_SENSORS_JOINTENCODER_HPP
#define THREEPP_SENSORS_JOINTENCODER_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/Sensor.hpp"
#include "threepp/math/MathUtils.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace threepp {

    /**
     * One encoder reading. `t` is the accumulated simulation time (s) at the end
     * of the sampled substep. Units follow the joint: rad and rad/s for a
     * revolute joint, m and m/s for a prismatic one.
     */
    struct JointSample {
        double t = 0.0;
        float position = 0.f;
        float velocity = 0.f;
    };

    class JointEncoder: public Sensor {

    public:
        /**
         * Quantization step — radians (revolute) or metres (prismatic) per encoder
         * tick. 0 (the default) is an ideal continuous encoder.
         *
         * For a rotary encoder quoted in counts per revolution use
         * setCountsPerRev(); for a quadrature encoder remember the usual 4x
         * decoding, i.e. counts = 4 * lines.
         */
        float resolution = 0.f;

        /**
         * Per-sample position noise. Only the X component of each Vector3 is
         * used (this is a scalar sensor); the rest of the NoiseModel semantics
         * are unchanged — whiteNoiseDensity is rad/sqrt(Hz), randomWalk drives
         * the slow zero drift, constantBias is a fixed mounting offset.
         *
         * Default is a clean encoder: the quantization above is the only error.
         */
        NoiseModel positionNoise{};

        /**
         * How velocity is produced.
         *
         * true (default) — differentiate the QUANTIZED, noisy position, which is
         * what a real encoder-fed controller actually computes and reproduces the
         * standstill chatter that exact velocity hides.
         *
         * false — report the simulator's true joint velocity (still noise-
         * corrupted if velocityNoise is set). Use when the encoder stands in for
         * a resolver or a motor-side tachometer, or when the controller under
         * test is fed velocity from elsewhere.
         */
        bool differentiateVelocity = true;

        /// Applied to the velocity channel only when differentiateVelocity is false.
        NoiseModel velocityNoise{};

        /**
         * @param node   The attachment node — normally the mesh bound to this
         *               joint's child link. Its world frame is not used by the
         *               measurement (a joint is measured in joint space), but it
         *               keeps the sensor sited in the scene graph like every
         *               other Sensor, and gives tooling something to draw.
         * @param link   The articulation link whose INBOUND joint is measured.
         *               Must not be the root link (the root has no joint).
         * @param rateHz Sample rate (Hz); 0 = every physics substep.
         * @param bufferCapacity Ring-buffer depth (oldest dropped on overflow).
         */
        JointEncoder(Object3D& node, const ArticulationLink& link,
                     double rateHz = 0.0, std::size_t bufferCapacity = 2048)
            : Sensor(node, rateHz), link_(link), ring_(bufferCapacity) {
            if (link_.isRoot()) {
                throw std::invalid_argument(
                        "JointEncoder: the root link has no inbound joint to measure. "
                        "Attach the encoder to a child link.");
            }
        }

        /// Set `resolution` from a rotary encoder's counts per revolution.
        void setCountsPerRev(int counts) {
            if (counts <= 0) throw std::invalid_argument("JointEncoder: countsPerRev must be > 0");
            resolution = math::TWO_PI / static_cast<float>(counts);
        }

        void onRegister(PhysxWorld& world) override {
            // The per-joint CPU getters this reads are not synced under the
            // direct-GPU path — they would return stale values forever. Fail at
            // registration rather than silently record garbage.
            if (world.directGpuEnabled()) {
                throw std::runtime_error(
                        "JointEncoder: not valid under direct_gpu — per-joint CPU state is not "
                        "synced. Read joint state through PhysxGpuBatch instead.");
            }
            reset();
        }

        /// Re-arm: clear the buffer and the differentiation history, and re-seed
        /// the noise from the current configs. Call after an episode reset.
        void reset() {
            hasPrevPos_ = false;
            prevPos_ = 0.f;
            posNoiseState_.reset(positionNoise);
            velNoiseState_.reset(velocityNoise);
            ring_.clear();
            resetTiming();
        }

        void sample(double dt, double simTime) override {

            // Noise first, then quantization: the shaft angle is exact, the
            // electrical read is noisy, and the tick boundary is the last thing
            // applied — so noise smaller than half a tick is mostly swallowed,
            // exactly as on hardware.
            float pos = link_.jointPosition();
            pos = posNoiseState_.apply(Vector3(pos, 0.f, 0.f), dt).x;
            pos = quantize(pos);

            float vel;
            if (differentiateVelocity) {
                // First sample after a reset has no predecessor: report 0 rather
                // than differencing against an undefined previous position.
                vel = (hasPrevPos_ && dt > 0.0) ? static_cast<float>((pos - prevPos_) / dt) : 0.f;
            } else {
                vel = velNoiseState_.apply(Vector3(link_.jointVelocity(), 0.f, 0.f), dt).x;
            }
            prevPos_ = pos;
            hasPrevPos_ = true;

            ring_.push(JointSample{simTime, pos, vel});
        }

        // --- read side (non-blocking) --------------------------------------

        [[nodiscard]] std::optional<JointSample> latest() const { return ring_.latest(); }
        void drain(std::vector<JointSample>& out) { ring_.drain(out); }
        [[nodiscard]] std::size_t available() const { return ring_.size(); }

    private:
        [[nodiscard]] float quantize(float v) const {
            if (resolution <= 0.f) return v;
            return std::round(v / resolution) * resolution;
        }

        ArticulationLink link_;

        bool hasPrevPos_ = false;
        float prevPos_ = 0.f;

        GaussianNoise posNoiseState_;
        GaussianNoise velNoiseState_;
        SensorRing<JointSample> ring_;
    };

}// namespace threepp

#endif// THREEPP_SENSORS_JOINTENCODER_HPP
