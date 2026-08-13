// PhysX-backed quadcopter for the ArduPilot SITL JSON backend.
//
// Lock-step contract: SITL sends one servo packet per physics frame and waits
// for our state reply, so one ServoInput = exactly one deterministic PhysX
// substep. PhysxWorld::step() is an accumulator, so we configure
// fixedTimestep = 1/frame_rate, maxSubSteps = 1, smoothTimestep = off and
// always step by exactly fixedTimestep — the accumulator then drains to
// precisely zero every call (same float added and subtracted). Because
// Settings are constructor-only, the QuadSim is built lazily by main() once
// the first packet has told us SITL's frame rate.
//
// The IMU reply reuses threepp::Imu, which computes exactly what SITL wants
// (specific force incl. lever-arm terms, gyro, per-substep, in the attachment
// node's frame) — with every noise field zeroed for a perfect sensor.
//
// Motor layout — ArduCopter X-frame, body FRD (x fwd, y right, z down);
// threepp drone-local axes in parentheses (forward = -Z, right = +X):
//
//   pwm[0]  motor 1  front-right (+x,+y)  ( a,0,-a)  CCW  yaw +
//   pwm[1]  motor 2  back-left   (-x,-y)  (-a,0, a)  CCW  yaw +
//   pwm[2]  motor 3  front-left  (+x,-y)  (-a,0,-a)  CW   yaw -
//   pwm[3]  motor 4  back-right  (-x,+y)  ( a,0, a)  CW   yaw -
//
// (a = armLength/sqrt(2); a CCW prop, seen from above, puts a clockwise
// reaction torque on the frame = positive FRD yaw.) Roll/pitch moments arise
// naturally from the thrust offsets; only the yaw reaction is explicit.

#ifndef THREEPP_EXAMPLE_SITL_QUADSIM_HPP
#define THREEPP_EXAMPLE_SITL_QUADSIM_HPP

#include "FrameConv.hpp"
#include "SitlBridge.hpp"

#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/Imu.hpp"
#include "threepp/objects/Mesh.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace sitl {

    struct QuadParams {
        float mass = 1.5f;      // [kg]
        float armLength = 0.25f;// hub -> motor [m]
        float inertiaRoll = 0.02f, inertiaPitch = 0.02f, inertiaYaw = 0.035f;// [kg m^2]
        float thrustToWeight = 4.f;// total max thrust / weight
        // Thrust curve matched to ArduPilot's default MOT_THST_EXPO = 0.65:
        // T = Tmax * ((1-expo)*t + expo*t^2). The autopilot linearizes its
        // outputs assuming this curve, so matching it keeps the attitude-loop
        // gain constant across throttle — mismatch shows up as wobble,
        // especially in descent.
        float thrustExpo = 0.65f;
        float motorTau = 0.02f;// first-order motor lag [s]; 50 ms was visibly sluggish
        float yawTorqueCoef = 0.05f; // reaction torque = coef * thrust [m]
        // Aero drag acts toward the AIR's velocity, not toward zero, so wind
        // pushes the airframe: F = mass * dragRate * (wind - v). With no wind
        // this equals the old PhysX linearDamping 0.3.
        float dragRate = 0.3f;// [1/s]
        float angularDamping = 0.05f;
        float batteryVolts = 12.6f;
    };

    class QuadSim {
    public:
        /// `droneMesh` must carry Box geometry (the collider is inferred from
        /// it) and already sit at its spawn pose, resting on the ground.
        /// `terrain` becomes an exact static trimesh collider; null = infinite
        /// ground plane at y=0 (the selftest path).
        QuadSim(std::uint16_t frameRateHz, threepp::Mesh& droneMesh,
                threepp::Mesh* terrain, QuadParams params = {})
            : params_(params), frameRate_(frameRateHz),
              fixedDt_(1.f / static_cast<float>(frameRateHz)),
              imu_(droneMesh) {

            using namespace ::physx;

            threepp::PhysxWorld::Settings s;
            s.fixedTimestep = fixedDt_;
            s.maxSubSteps = 1;
            s.smoothTimestep = false;
            world_ = std::make_unique<threepp::PhysxWorld>(s);

            if (terrain) {
                world_->addStaticTrimesh(*terrain);
            } else {
                // PxPlaneGeometry's plane is the YZ plane (normal +X); rotate
                // +90 deg about Z so the normal becomes +Y (ground at y=0).
                world_->addStatic(PxPlaneGeometry(),
                                  PxTransform(PxVec3(0), PxQuat(PxHalfPi, PxVec3(0, 0, 1))));
            }

            droneMesh.updateMatrixWorld(true);
            body_ = world_->add(droneMesh, 100.f);
            body_->setMass(params_.mass);
            // Mass-space (drone-local) diagonal: local z is the FRD roll axis,
            // local x the pitch axis, local y the yaw axis.
            body_->setMassSpaceInertiaTensor(
                    PxVec3(params_.inertiaPitch, params_.inertiaYaw, params_.inertiaRoll));
            body_->setLinearDamping(0.f);// drag is explicit (wind-relative), see applyMotors
            body_->setAngularDamping(params_.angularDamping);
            body_->setSleepThreshold(0.f);// a hovering quad must never doze off
            home_ = body_->getGlobalPose();

            // Perfect IMU: SITL adds its own sensor noise via SIM_* parameters.
            imu_.gyroNoise = {};
            imu_.accelNoise = {};
            world_->registerSensor(&imu_);

            const float a = params_.armLength / std::sqrt(2.f);
            motorPosLocal_[0] = PxVec3(a, 0, -a); // M1 front-right
            motorPosLocal_[1] = PxVec3(-a, 0, a); // M2 back-left
            motorPosLocal_[2] = PxVec3(-a, 0, -a);// M3 front-left
            motorPosLocal_[3] = PxVec3(a, 0, a);  // M4 back-right
            yawSign_[0] = yawSign_[1] = 1.f;      // CCW
            yawSign_[2] = yawSign_[3] = -1.f;     // CW

            preHandle_ = world_->onPreSubstep([this](float dt) { applyMotors(dt); });
        }

        ~QuadSim() {
            world_->removeSubstepCallback(preHandle_);
            world_->unregisterSensor(&imu_);
        }

        [[nodiscard]] std::uint16_t frameRate() const { return frameRate_; }

        /// One SITL frame: latch throttle commands, run exactly one substep,
        /// and report the resulting state (NED / FRD, per the JSON protocol).
        void step(const ServoInput& in, FdmState& out) {
            using namespace ::physx;

            const float maxThrustPerMotor =
                    params_.thrustToWeight * params_.mass * 9.80665f / 4.f;
            for (int i = 0; i < 4; ++i) {
                cmd_[i] = std::clamp((static_cast<float>(in.pwm[i]) - 1000.f) / 1000.f, 0.f, 1.f);
            }
            maxThrust_ = maxThrustPerMotor;

            world_->step(fixedDt_);
            ++frames_;

            out.timestampSec = static_cast<double>(frames_) / frameRate_;

            const PxTransform pose = body_->getGlobalPose();
            const threepp::Vector3 relPos(pose.p.x - home_.p.x, pose.p.y - home_.p.y,
                                          pose.p.z - home_.p.z);
            frame::tpToNed(relPos, out.positionNed[0], out.positionNed[1], out.positionNed[2]);

            const PxVec3 vel = body_->getLinearVelocity();
            frame::tpToNed({vel.x, vel.y, vel.z},
                           out.velocityNed[0], out.velocityNed[1], out.velocityNed[2]);

            double qw, qx, qy, qz;
            frame::tpAttToNed(threepp::Quaternion(pose.q.x, pose.q.y, pose.q.z, pose.q.w),
                              qw, qx, qy, qz);
            frame::nedQuatToRpy(qw, qx, qy, qz,
                                out.attitudeRpy[0], out.attitudeRpy[1], out.attitudeRpy[2]);

            // threepp::Imu samples every substep in the drone-node frame; the
            // node frame relates to FRD by the same axis triple as world NED.
            if (const auto s = imu_.latest()) {
                frame::tpToNed(s->angularVelocity, out.gyro[0], out.gyro[1], out.gyro[2]);
                frame::tpToNed(s->linearAcceleration,
                               out.accelBody[0], out.accelBody[1], out.accelBody[2]);
            }

            // Wind report: lets SITL's airspeed sensor and wind estimator see
            // the same air the drag force uses.
            const PxVec3 wind = currentWind();
            const PxVec3 air = vel - wind;
            out.airspeed = air.magnitude();
            frame::tpToNed({wind.x, wind.y, wind.z},
                           out.windNed[0], out.windNed[1], out.windNed[2]);

            out.rangefinderM = raycastAgl(pose);
            out.batteryV = params_.batteryVolts;
            out.batteryA = 4.0 * (throttle_[0] + throttle_[1] + throttle_[2] + throttle_[3]);
        }

        /// Steady wind (threepp world frame) + gustiness [0..1]. Gusts modulate
        /// magnitude and wobble direction deterministically from sim time.
        void setWind(const threepp::Vector3& windTp, float gustiness) {
            baseWind_ = ::physx::PxVec3(windTp.x, windTp.y, windTp.z);
            gustiness_ = std::clamp(gustiness, 0.f, 1.f);
        }

        /// The instantaneous (gusty) wind, threepp world frame — for the
        /// windsock and air-streak visuals, so they show the SAME air the
        /// airframe feels.
        [[nodiscard]] threepp::Vector3 windNow() const {
            const auto w = currentWind();
            return {w.x, w.y, w.z};
        }

        /// SITL restarted: back to the spawn pose, everything zeroed.
        void reset() {
            using namespace ::physx;
            body_->setGlobalPose(home_);
            body_->setLinearVelocity(PxVec3(0));
            body_->setAngularVelocity(PxVec3(0));
            for (auto& t : throttle_) t = 0.f;
            for (auto& c : cmd_) c = 0.f;
            imu_.reset();
            frames_ = 0;
        }

        /// Filtered throttle 0..1 of one motor (visual rotor spin + HUD).
        [[nodiscard]] float motorLevel(int i) const { return throttle_[i]; }

        [[nodiscard]] threepp::PhysxWorld& world() { return *world_; }

    private:
        /// Effective wind right now (base + deterministic gusts).
        [[nodiscard]] ::physx::PxVec3 currentWind() const {
            using namespace ::physx;
            if (gustiness_ <= 0.f || baseWind_.magnitudeSquared() < 1e-6f) return baseWind_;
            const float t = static_cast<float>(frames_) / frameRate_;
            const float mag = 1.f + gustiness_ * (0.45f * std::sin(0.9f * t) +
                                                  0.30f * std::sin(2.3f * t + 1.7f) +
                                                  0.20f * std::sin(5.1f * t + 0.4f));
            const float wobble = gustiness_ * 0.25f * std::sin(0.6f * t + 2.f);
            const float c = std::cos(wobble), s = std::sin(wobble);
            return PxVec3((baseWind_.x * c + baseWind_.z * s) * mag, baseWind_.y * mag,
                          (-baseWind_.x * s + baseWind_.z * c) * mag);
        }

        void applyMotors(float dt) {
            using namespace ::physx;
            const float alpha = std::clamp(dt / params_.motorTau, 0.f, 1.f);

            // Wind-relative aero drag (replaces PhysX linear damping).
            const PxVec3 airVel = currentWind() - body_->getLinearVelocity();
            body_->addForce(airVel * (params_.dragRate * params_.mass));

            float yawTorque = 0.f;
            for (int i = 0; i < 4; ++i) {
                throttle_[i] += (cmd_[i] - throttle_[i]) * alpha;
                const float t = throttle_[i];
                const float thrust =
                        maxThrust_ * ((1.f - params_.thrustExpo) * t + params_.thrustExpo * t * t);
                PxRigidBodyExt::addLocalForceAtLocalPos(*body_, PxVec3(0, thrust, 0),
                                                        motorPosLocal_[i]);
                yawTorque += yawSign_[i] * params_.yawTorqueCoef * thrust;
            }
            // FRD yaw torque (about down) = drone-local -Y; addTorque wants world.
            const PxVec3 worldTorque = body_->getGlobalPose().q.rotate(PxVec3(0, -yawTorque, 0));
            body_->addTorque(worldTorque);
        }

        /// Downward AGL for "rng_1"; NaN (omitted) when nothing below in range.
        [[nodiscard]] double raycastAgl(const ::physx::PxTransform& pose) const {
            using namespace ::physx;
            PxRaycastBuffer hit;
            PxQueryFilterData fd;
            fd.flags = PxQueryFlag::eSTATIC;// terrain only, never the drone itself
            if (world_->scene().raycast(pose.p, PxVec3(0, -1, 0), 200.f, hit,
                                        PxHitFlag::eDEFAULT, fd) &&
                hit.hasBlock) {
                return hit.block.distance;
            }
            return NAN;
        }

        QuadParams params_;
        std::uint16_t frameRate_;
        float fixedDt_;
        std::unique_ptr<threepp::PhysxWorld> world_;
        ::physx::PxRigidDynamic* body_ = nullptr;
        ::physx::PxTransform home_;
        threepp::Imu imu_;
        threepp::PhysxWorld::SubstepHandle preHandle_{};

        float cmd_[4] = {}, throttle_[4] = {};
        ::physx::PxVec3 baseWind_{0.f, 0.f, 0.f};
        float gustiness_ = 0.f;
        float maxThrust_ = 0.f;
        float yawSign_[4] = {};
        ::physx::PxVec3 motorPosLocal_[4];
        std::uint64_t frames_ = 0;
    };

}// namespace sitl

#endif// THREEPP_EXAMPLE_SITL_QUADSIM_HPP
