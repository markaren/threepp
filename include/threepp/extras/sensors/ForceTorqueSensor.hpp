// Force/torque sensor — the wrench transmitted through an articulation joint.
//
// The third of the Gazebo-parity trio (with JointEncoder and ContactSensor).
// A real F/T sensor is a load cell bolted between two links, and it measures
// exactly this: the six-component wrench the parent link applies to the child
// through their joint. It is what force control, admittance control, collision
// detection without contact sensing, and payload estimation all read.
//
// PhysX reports this as the articulation cache's linkIncomingJointForce, so the
// measurement is the SOLVER's constraint force — the same quantity a load cell
// would see, not a finite-difference estimate.
//
// The same instrument also bolts across a plain authored Joint (a weld, a
// hinge between two rigid bodies): there the wrench is the constraint force
// PxConstraint::getForce reports, the maximal-coordinate twin of the quantity
// above. Which source feeds the sensor is fixed at construction; the noise
// model, rate gate and buffering are identical either way.
//
// Frame: PhysX reports the articulation wrench in the CHILD JOINT FRAME of the
// measured link's inbound joint (see
// PxArticulationJointReducedCoordinate::getChildPose), which is the joint
// frame this library orients with its X axis along the hinge/slide axis. A
// plain joint's constraint force is reported in WORLD axes. Magnitudes are of
// course frame-independent, and are usually what a threshold is written
// against.

#ifndef THREEPP_SENSORS_FORCETORQUESENSOR_HPP
#define THREEPP_SENSORS_FORCETORQUESENSOR_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/Joint.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/Sensor.hpp"
#include "threepp/math/Vector3.hpp"

#include <PxPhysicsAPI.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace threepp {

    /**
     * One six-component wrench reading, in the measured joint's child frame.
     * `t` is the accumulated simulation time (s) at the end of the sampled
     * substep.
     */
    struct WrenchSample {
        double t = 0.0;
        Vector3 force{0.f, 0.f, 0.f}; //!< N
        Vector3 torque{0.f, 0.f, 0.f};//!< N*m
    };

    class ForceTorqueSensor: public Sensor {

    public:
        /// Per-axis noise, in the joint's child frame. Defaults to a perfect
        /// sensor; a real load cell is worth modelling with a small white-noise
        /// density plus a constant bias (the tare error).
        NoiseModel forceNoise{};
        NoiseModel torqueNoise{};

        /**
         * @param node   The attachment node — normally the mesh bound to the
         *               measured link, i.e. where the load cell physically sits.
         * @param art    The articulation the link belongs to. Must outlive the
         *               sensor: the sensor holds a state cache belonging to it,
         *               released on unregister/destruction.
         * @param link   The link whose INBOUND joint is measured. Must not be the
         *               root (PhysX reports a zero wrench for the root, since
         *               there is no incoming joint to transmit one).
         * @param rateHz Sample rate (Hz); 0 = every physics substep.
         * @param bufferCapacity Ring-buffer depth (oldest dropped on overflow).
         */
        ForceTorqueSensor(Object3D& node, Articulation& art, const ArticulationLink& link,
                          double rateHz = 0.0, std::size_t bufferCapacity = 2048)
            : Sensor(node, rateHz), art_(&art), ring_(bufferCapacity) {
            if (link.isRoot()) {
                throw std::invalid_argument(
                        "ForceTorqueSensor: the root link has no inbound joint, so no wrench is "
                        "transmitted through one. Measure a child link.");
            }
            auto* raw = link.raw();
            if (!raw) throw std::invalid_argument("ForceTorqueSensor: link has no PhysX link");
            linkIndex_ = raw->getLinkIndex();
        }

        /**
         * The same load cell across a plain authored Joint: the wrench is the
         * solver's constraint force on that joint (Joint::reactionForce), in
         * world axes. No articulation, no cache — abandonCache() is a no-op on
         * this form. The joint must outlive the sensor; in the editor that
         * ordering is the play controller's (sensors stop before the physics
         * session that owns the joints).
         */
        ForceTorqueSensor(Object3D& node, const Joint& joint,
                          double rateHz = 0.0, std::size_t bufferCapacity = 2048)
            : Sensor(node, rateHz), joint_(&joint), ring_(bufferCapacity) {}

        ~ForceTorqueSensor() override { releaseCache(); }

        void onRegister(PhysxWorld& world) override {
            if (world.directGpuEnabled()) {
                throw std::runtime_error(
                        "ForceTorqueSensor: not valid under direct_gpu - the CPU-side constraint "
                        "state is not synced. Read link forces through PhysxGpuBatch instead.");
            }
            // The plain-joint form has no cache lifecycle at all: the wrench
            // is read straight off the constraint each sample.
            if (joint_) {
                reset();
                return;
            }
            if (!art_->finalized()) {
                throw std::runtime_error(
                        "ForceTorqueSensor: finalize() the articulation before registering - the "
                        "state cache can only be created once it belongs to a scene.");
            }
            if (!cache_) cache_ = art_->rawArt()->createCache();
            if (!cache_) throw std::runtime_error("ForceTorqueSensor: createCache failed");
            reset();
        }

        void onUnregister() override {
            // The cache belongs to the articulation; drop it here so a sensor that
            // outlives its articulation cannot release it afterwards.
            releaseCache();
        }

        // Drop the cache WITHOUT releasing it. For the one caller that knows the
        // PhysX SDK has already been torn down (so the cache memory is gone with
        // it): calling release() then would touch a freed allocator. A host that
        // stops its physics world before this sensor - which is the editor's
        // session stop order - uses this instead of the destructor's release().
        // See ForceTorqueSensor's use in PhysxSensorPlaySession.
        void abandonCache() { cache_ = nullptr; }

        /// Re-arm: clear the buffer and re-seed the noise from the current configs.
        void reset() {
            forceNoiseState_.reset(forceNoise);
            torqueNoiseState_.reset(torqueNoise);
            ring_.clear();
            resetTiming();
        }

        void sample(double dt, double simTime) override {
            using namespace ::physx;

            Vector3 force, torque;
            if (joint_) {
                joint_->reactionForce(force, torque);
            } else {
                if (!cache_) return;// not registered

                art_->rawArt()->copyInternalStateToCache(
                        *cache_, PxArticulationCacheFlag::eLINK_INCOMING_JOINT_FORCE);

                const PxSpatialForce& w = cache_->linkIncomingJointForce[linkIndex_];
                force.set(w.force.x, w.force.y, w.force.z);
                torque.set(w.torque.x, w.torque.y, w.torque.z);
            }

            force = forceNoiseState_.apply(force, dt);
            torque = torqueNoiseState_.apply(torque, dt);

            ring_.push(WrenchSample{simTime, force, torque});
        }

        // --- read side (non-blocking) --------------------------------------

        [[nodiscard]] std::optional<WrenchSample> latest() const { return ring_.latest(); }
        void drain(std::vector<WrenchSample>& out) { ring_.drain(out); }
        [[nodiscard]] std::size_t available() const { return ring_.size(); }

    private:
        void releaseCache() {
            if (cache_) {
                cache_->release();
                cache_ = nullptr;
            }
        }

        // Exactly one of these is set (see the two constructors): the
        // articulation + cache + link index, or the plain joint.
        Articulation* art_ = nullptr;
        ::physx::PxArticulationCache* cache_ = nullptr;
        ::physx::PxU32 linkIndex_ = 0;
        const Joint* joint_ = nullptr;

        GaussianNoise forceNoiseState_;
        GaussianNoise torqueNoiseState_;
        SensorRing<WrenchSample> ring_;
    };

}// namespace threepp

#endif// THREEPP_SENSORS_FORCETORQUESENSOR_HPP
