// Inertial Measurement Unit (gyroscope + accelerometer).
//
// Attach an Imu to an Object3D that is rigidly fixed to (a child of, or) a
// PhysX rigid body managed by a PhysxWorld, register it, and step the world.
// Each fixed substep the Imu measures, in the SENSOR (attachment node) frame:
//
//   angularVelocity   the body's angular velocity            [rad/s]
//   linearAcceleration the SPECIFIC FORCE at the sensor point [m/s^2]
//
// Specific force is what a real accelerometer reads — proper acceleration minus
// gravity, so a level sensor at rest reads +g on its up axis, and a sensor in
// free fall reads ~0:
//
//   f_sensor = R_ws^-1 * (a_point - g)
//
// where R_ws is the sensor's world orientation, g is the world gravity vector
// (read live from the PhysxWorld, threepp is Y-up so g = (0, -9.81, 0) by
// default), and a_point is the proper acceleration of the sensor point on the
// body including the lever-arm terms when the sensor is offset from the body's
// centre of mass:
//
//   a_point = a_com + alpha x r + omega x (omega x r)
//
// a_com (CoM linear acceleration) and alpha (angular acceleration) are obtained
// by finite-differencing the PhysX body's linear / angular velocity across
// consecutive samples; r is the CoM->sensor lever arm. The FIRST sample after
// attach/reset treats the previous velocity as the current one, so a_com and
// alpha are 0 rather than an impulse spike from an undefined difference.
//
// Frames: gyro and accel are expressed in the attachment node's world frame.
// The body->sensor offset is captured once at registration from the node's then
// world pose vs the actor pose, so sampling does not depend on the visual
// binding having been synced yet this substep (it hasn't — sensors sample
// mid-step, the visuals sync at step() end).

#ifndef THREEPP_SENSORS_IMU_HPP
#define THREEPP_SENSORS_IMU_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/Sensor.hpp"
#include "threepp/math/Vector3.hpp"

#include <PxPhysicsAPI.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace threepp {

    /**
     * One IMU measurement. `t` is the accumulated simulation time (s) at the end
     * of the sampled substep. Angular velocity is rad/s, linear acceleration is
     * specific force in m/s^2, both in the sensor (attachment node) frame.
     */
    struct ImuSample {
        double t = 0.0;
        Vector3 angularVelocity{0.f, 0.f, 0.f};
        Vector3 linearAcceleration{0.f, 0.f, 0.f};
    };

    /**
     * A named IMU noise model taken from a datasheet: a gyroscope and an
     * accelerometer NoiseModel. Only the white-noise densities are datasheet
     * figures. The parts below publish no rate random walk in the units of
     * NoiseModel::randomWalk (an "in-run bias stability" is a different
     * Allan-variance regime), so a preset leaves randomWalk at zero and says so
     * here rather than inventing a number; set it yourself for a drifting bias.
     * The seeds are the sensor's long-standing defaults, so seeded replays keep
     * their meaning across the change of densities.
     */
    struct ImuModel {
        const char* name = "";
        NoiseModel gyro;
        NoiseModel accel;

        // TDK InvenSense ICM-42688-P (datasheet DS-000347): gyroscope noise
        // density 2.8 mdps/sqrt(Hz) = 4.8869e-5 rad/s/sqrt(Hz), accelerometer
        // noise density 70 ug/sqrt(Hz) = 6.8647e-4 m/s^2/sqrt(Hz).
        static ImuModel ICM42688P() {
            ImuModel m;
            m.name = "ICM-42688-P";
            m.gyro = NoiseModel{
                    /*whiteNoiseDensity*/ Vector3(4.8869e-5f, 4.8869e-5f, 4.8869e-5f),
                    /*randomWalk*/ Vector3(0.f, 0.f, 0.f),
                    /*constantBias*/ Vector3(0.f, 0.f, 0.f),
                    /*seed*/ 0x9E3779B97F4A7C15ULL};
            m.accel = NoiseModel{
                    /*whiteNoiseDensity*/ Vector3(6.8647e-4f, 6.8647e-4f, 6.8647e-4f),
                    /*randomWalk*/ Vector3(0.f, 0.f, 0.f),
                    /*constantBias*/ Vector3(0.f, 0.f, 0.f),
                    /*seed*/ 0xBF58476D1CE4E5B9ULL};
            return m;
        }
    };

    class Imu: public Sensor {

    public:
        // Per-axis noise. Public so a domain-randomization loop can re-roll them;
        // call reset() afterwards to re-seed the RNG / re-arm the finite-difference.
        //
        // Defaults are the ICM-42688-P preset: datasheet white-noise densities,
        // no random walk (see ImuModel). The previous placeholder, 0.005
        // rad/s/sqrt(Hz) and 0.06 m/s^2/sqrt(Hz) with invented random walks, sat
        // two orders of magnitude above any current part; the sensor-conformance
        // table of the reproducibility paper found it, and this is the fix.
        // Set every field to 0 for a perfect (noiseless, unbiased) sensor.
        NoiseModel gyroNoise  = ImuModel::ICM42688P().gyro;
        NoiseModel accelNoise = ImuModel::ICM42688P().accel;

        /**
         * @param node   The attachment node; its world frame is the sensor frame.
         *               Must be, or be a descendant of, a mesh added to a
         *               PhysxWorld as a rigid body.
         * @param rateHz Sample rate (Hz); 0 = every physics substep.
         * @param bufferCapacity Ring-buffer depth (oldest dropped on overflow).
         */
        explicit Imu(Object3D& node, double rateHz = 0.0, std::size_t bufferCapacity = 2048)
            : Sensor(node, rateHz), ring_(bufferCapacity) {}

        // Resolve the rigid body from the attachment node's ancestry, capture the
        // rigid body->sensor offset, and seed the noise. Throws if no
        // PhysxWorld-managed body is found (so the mistake surfaces here, not
        // silently mid-simulation).
        void onRegister(PhysxWorld& world) override {
            using namespace ::physx;
            world_ = &world;
            actor_ = world.findActor(node());
            if (!actor_) {
                throw std::invalid_argument(
                        "Imu: the attachment Object3D has no PhysxWorld-managed rigid body in "
                        "its ancestry. Attach the IMU to a mesh added via world.add(...) (or a "
                        "child of it) before registering.");
            }
            // Dynamic bodies and articulation links are PxRigidBody (have velocity
            // + CoM); a static collider is not — treat it as a body at rest.
            body_ = actor_->is<PxRigidBody>();

            // Capture the constant body->sensor rigid offset from the node's world
            // pose at registration (consistent with the actor pose before stepping).
            node()->updateWorldMatrix(true, false);
            Vector3 sp, ss;
            Quaternion sq;
            node()->matrixWorld->decompose(sp, sq, ss);
            const PxTransform bodyPose = actor_->getGlobalPose();
            const PxQuat qSensorWorld(static_cast<float>(sq.x), static_cast<float>(sq.y),
                                      static_cast<float>(sq.z), static_cast<float>(sq.w));
            const PxVec3 pSensorWorld(sp.x, sp.y, sp.z);
            qOffset_ = bodyPose.q.getConjugate() * qSensorWorld;
            pOffset_ = bodyPose.q.rotateInv(pSensorWorld - bodyPose.p);

            reset();
        }

        void onUnregister() override {
            world_ = nullptr;
            actor_ = nullptr;
            body_ = nullptr;
        }

        // The body this IMU rides is being released: go inert rather than keep a
        // dangling pointer. Still registered, so re-attaching is a matter of
        // unregister/register once a new body exists.
        void onActorRemoved(::physx::PxRigidActor* actor) override {
            if (actor != actor_) return;
            actor_ = nullptr;
            body_ = nullptr;
            hasPrevVel_ = false;
        }

        // True once registered against a live rigid body. False before
        // registration, after unregistering, and after the body was removed —
        // sampling is a no-op in all three cases.
        [[nodiscard]] bool attached() const { return actor_ != nullptr; }

        // Re-arm: clear the finite-difference history + buffer and re-seed the
        // noise from the current gyroNoise / accelNoise configs. Call after an
        // episode reset (teleport / velocity zeroing) or after changing noise.
        void reset() {
            hasPrevVel_ = false;
            prevLinVel_ = ::physx::PxVec3(0.f);
            prevAngVel_ = ::physx::PxVec3(0.f);
            gyroNoiseState_.reset(gyroNoise);
            accelNoiseState_.reset(accelNoise);
            ring_.clear();
            resetTiming();
        }

        void sample(double dt, double simTime) override {
            using namespace ::physx;

            // No live body: unregistered, or the body was removed under us.
            // tick() is public, so this is reachable without a PhysxWorld at all.
            if (!actor_) return;

            const PxTransform bodyPose = actor_->getGlobalPose();

            PxVec3 omega(0.f);// world angular velocity
            PxVec3 vCom(0.f); // world CoM linear velocity
            PxVec3 comLocal(0.f);
            if (body_) {
                omega = body_->getAngularVelocity();
                vCom = body_->getLinearVelocity();
                comLocal = body_->getCMassLocalPose().p;
            }

            // Finite-difference accelerations. First sample: prev := current so
            // the differences are 0 (no start-up spike).
            if (!hasPrevVel_) {
                prevLinVel_ = vCom;
                prevAngVel_ = omega;
                hasPrevVel_ = true;
            }
            const float invDt = (dt > 0.0) ? static_cast<float>(1.0 / dt) : 0.f;
            const PxVec3 aCom = (vCom - prevLinVel_) * invDt;
            const PxVec3 alpha = (omega - prevAngVel_) * invDt;
            prevLinVel_ = vCom;
            prevAngVel_ = omega;

            // Sensor world pose (from the captured rigid offset) and lever arm.
            const PxQuat qSensor = bodyPose.q * qOffset_;
            const PxVec3 pSensor = bodyPose.p + bodyPose.q.rotate(pOffset_);
            const PxVec3 pCom = bodyPose.p + bodyPose.q.rotate(comLocal);
            const PxVec3 r = pSensor - pCom;

            // Proper acceleration of the sensor point, then specific force.
            const PxVec3 aPoint = aCom + alpha.cross(r) + omega.cross(omega.cross(r));
            const Vector3 g = world_ ? world_->settings().gravity : Vector3(0.f, -9.81f, 0.f);
            const PxVec3 gPx(g.x, g.y, g.z);
            const PxVec3 fWorld = aPoint - gPx;

            // Express in the sensor frame: R_ws^-1 * v.
            const PxVec3 gyroS = qSensor.rotateInv(omega);
            const PxVec3 accelS = qSensor.rotateInv(fWorld);

            Vector3 gyro(gyroS.x, gyroS.y, gyroS.z);
            Vector3 accel(accelS.x, accelS.y, accelS.z);
            gyro = gyroNoiseState_.apply(gyro, dt);
            accel = accelNoiseState_.apply(accel, dt);

            ring_.push(ImuSample{simTime, gyro, accel});
        }

        // --- read side (non-blocking) --------------------------------------

        // Most recent measurement, or nullopt if none yet. Survives drain().
        [[nodiscard]] std::optional<ImuSample> latest() const { return ring_.latest(); }

        // Move all buffered measurements (oldest-first) into out; empties the buffer.
        void drain(std::vector<ImuSample>& out) { ring_.drain(out); }

        [[nodiscard]] std::size_t available() const { return ring_.size(); }

    private:
        PhysxWorld* world_ = nullptr;
        ::physx::PxRigidActor* actor_ = nullptr;
        ::physx::PxRigidBody* body_ = nullptr;// null for a static attachment

        // Constant rigid body->sensor offset, captured at registration.
        ::physx::PxQuat qOffset_{::physx::PxIdentity};
        ::physx::PxVec3 pOffset_{0.f, 0.f, 0.f};

        // Finite-difference state.
        bool hasPrevVel_ = false;
        ::physx::PxVec3 prevLinVel_{0.f, 0.f, 0.f};
        ::physx::PxVec3 prevAngVel_{0.f, 0.f, 0.f};

        GaussianNoise gyroNoiseState_;
        GaussianNoise accelNoiseState_;
        SensorRing<ImuSample> ring_;
    };

}// namespace threepp

#endif// THREEPP_SENSORS_IMU_HPP
