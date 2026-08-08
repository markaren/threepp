// A joint between two rigid bodies (or one body and the world), built on the
// PhysX joint extensions and expressed in threepp terms: one world-space joint
// frame (origin at the anchor, X along the hinge/slide axis — the same
// convention Articulation::addLink uses), plus limits, a PD drive and a break
// threshold.
//
// One implementation for almost every type: a PxD6Joint configured per type,
// rather than one PhysX class per type. The D6 carries the same limit/drive
// plumbing for every axis, so fixed/revolute/prismatic/spherical differ only
// in which motions unlock — one code path to get right instead of four that
// drift. The exception is Distance, which PhysX models as its own joint (a
// tether between two anchors constrains a length, not a frame).
//
// This complements Articulation, it does not compete with it: an articulation
// is a reduced-coordinate CHAIN (a robot), while a Joint is one maximal-
// coordinate constraint between two otherwise independent actors — a door on
// its frame, a trailer on its hitch, a pendulum on the world. Either side may
// be an articulation link (a PxArticulationLink is a PxRigidActor), which is
// how a prop attaches to a robot's gripper.
//
// Lifetime: a Joint must be destroyed BEFORE the PhysxWorld it was created in.
// Releasing the world's PxPhysics releases every joint with it, and this class
// releases the joint again in its destructor — so the session that owns both
// clears its joints first (see PhysicsPlaySession::stop()).

#ifndef THREEPP_PHYSX_JOINT_HPP
#define THREEPP_PHYSX_JOINT_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>

#include <stdexcept>

namespace threepp {

    class Joint {
    public:
        enum class Type {
            Fixed,    // welds the two frames together
            Revolute, // hinge: rotation about the frame's X axis
            Prismatic,// slider: translation along the frame's X axis
            Spherical,// ball socket: all three rotations, optional swing cone
            Distance  // tether: keeps the anchors within [min, max] metres
        };

        struct Params {
            Type type = Type::Fixed;

            // Motion limits. Revolute: radians about X. Prismatic: metres
            // along X. Spherical: `limited` turns the swing cone on (coneY /
            // coneZ half-angles in radians; twist stays free). Distance
            // ignores `limited` — lower/upper ARE the constraint (min/max
            // metres; a min of 0 leaves only the max enforced).
            bool limited = false;
            float lower = 0.f;
            float upper = 0.f;
            float coneY = 0.f;
            float coneZ = 0.f;

            // PD drive toward `target` (radians/metres along the motion axis),
            // plus a feed-forward `velocity`. Zero stiffness AND damping means
            // no drive at all — and each half gates its own input: the drive
            // force is stiffness·(target−x) + damping·(velocity−v), so a
            // `target` needs stiffness > 0 to act and a `velocity` needs
            // damping > 0. Force mode, not acceleration, so the numbers mean
            // the same as ArticulationConfig's. A driven Spherical joint is a
            // SLERP spring back to the joint frame's rest orientation.
            // Distance uses stiffness/damping as its tether spring instead.
            float stiffness = 0.f;
            float damping = 0.f;
            float maxForce = PX_MAX_F32;
            float target = 0.f;
            float velocity = 0.f;

            // The constraint breaks (and the bodies separate for good) when
            // the solver needs more than this. Zero or negative = unbreakable.
            float breakForce = 0.f;
            float breakTorque = 0.f;

            // Whether the two jointed bodies still collide with each other.
            // Off by default: bodies meeting at a joint overlap at the anchor,
            // and contacts there fight the constraint.
            bool collide = false;
        };

        // `frame` is the joint frame in WORLD space: origin at the anchor, X
        // along the hinge/slide axis. Either actor may be null, meaning "the
        // world" — but not both, which would constrain nothing.
        Joint(PhysxWorld& world, ::physx::PxRigidActor* a, ::physx::PxRigidActor* b,
              const ::physx::PxTransform& frame, const Params& params)
            : type_(params.type) {
            using namespace ::physx;

            if (!a && !b) throw std::invalid_argument("Joint: both sides are the world");
            if (a == b) throw std::invalid_argument("Joint: both sides are the same actor");

            // A null actor's "local" frame is the world frame itself.
            const PxTransform frameA = a ? a->getGlobalPose().transformInv(frame) : frame;
            const PxTransform frameB = b ? b->getGlobalPose().transformInv(frame) : frame;

            if (params.type == Type::Distance) {
                auto* joint = PxDistanceJointCreate(world.physics(), a, frameA, b, frameB);
                if (!joint) throw std::runtime_error("Joint: PxDistanceJointCreate failed");
                const float minD = std::max(params.lower, 0.f);
                const float maxD = std::max(params.upper, minD);
                joint->setMinDistance(minD);
                joint->setMaxDistance(maxD);
                joint->setDistanceJointFlag(PxDistanceJointFlag::eMIN_DISTANCE_ENABLED, minD > 0.f);
                joint->setDistanceJointFlag(PxDistanceJointFlag::eMAX_DISTANCE_ENABLED, true);
                if (params.stiffness > 0.f || params.damping > 0.f) {
                    joint->setStiffness(params.stiffness);
                    joint->setDamping(params.damping);
                    joint->setDistanceJointFlag(PxDistanceJointFlag::eSPRING_ENABLED, true);
                }
                joint_ = joint;
            } else {
                auto* joint = PxD6JointCreate(world.physics(), a, frameA, b, frameB);
                if (!joint) throw std::runtime_error("Joint: PxD6JointCreate failed");

                // Every motion starts eLOCKED (the D6 default), i.e. Fixed;
                // the other types unlock theirs.
                switch (params.type) {
                    case Type::Revolute:
                        joint->setMotion(PxD6Axis::eTWIST,
                                         params.limited ? PxD6Motion::eLIMITED : PxD6Motion::eFREE);
                        if (params.limited)
                            joint->setTwistLimit(PxJointAngularLimitPair(params.lower, params.upper));
                        break;
                    case Type::Prismatic:
                        joint->setMotion(PxD6Axis::eX,
                                         params.limited ? PxD6Motion::eLIMITED : PxD6Motion::eFREE);
                        if (params.limited)
                            joint->setLinearLimit(PxD6Axis::eX,
                                                  PxJointLinearLimitPair(world.physics().getTolerancesScale(),
                                                                         params.lower, params.upper));
                        break;
                    case Type::Spherical:
                        joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
                        joint->setMotion(PxD6Axis::eSWING1,
                                         params.limited ? PxD6Motion::eLIMITED : PxD6Motion::eFREE);
                        joint->setMotion(PxD6Axis::eSWING2,
                                         params.limited ? PxD6Motion::eLIMITED : PxD6Motion::eFREE);
                        if (params.limited)
                            joint->setSwingLimit(PxJointLimitCone(params.coneY, params.coneZ));
                        break;
                    case Type::Fixed:
                    case Type::Distance:
                        break;
                }

                if (params.stiffness > 0.f || params.damping > 0.f) {
                    const PxD6JointDrive drive(params.stiffness, params.damping,
                                               params.maxForce > 0.f ? params.maxForce : PX_MAX_F32,
                                               false);
                    switch (params.type) {
                        case Type::Revolute:
                            joint->setDrive(PxD6Drive::eTWIST, drive);
                            joint->setDrivePosition(
                                    PxTransform(PxQuat(params.target, PxVec3(1, 0, 0))));
                            joint->setDriveVelocity(PxVec3(PxZero), PxVec3(params.velocity, 0, 0));
                            break;
                        case Type::Prismatic:
                            joint->setDrive(PxD6Drive::eX, drive);
                            joint->setDrivePosition(PxTransform(PxVec3(params.target, 0, 0)));
                            joint->setDriveVelocity(PxVec3(params.velocity, 0, 0), PxVec3(PxZero));
                            break;
                        case Type::Spherical:
                            joint->setDrive(PxD6Drive::eSLERP, drive);
                            joint->setDrivePosition(PxTransform(PxIdentity));
                            break;
                        case Type::Fixed:
                        case Type::Distance:
                            break;// nothing to drive
                    }
                }
                joint_ = joint;
            }

            if (params.breakForce > 0.f || params.breakTorque > 0.f) {
                joint_->setBreakForce(params.breakForce > 0.f ? params.breakForce : PX_MAX_F32,
                                      params.breakTorque > 0.f ? params.breakTorque : PX_MAX_F32);
            }
            joint_->setConstraintFlag(::physx::PxConstraintFlag::eCOLLISION_ENABLED, params.collide);
            // Always eligible for the scene's debug visualization (the editor's
            // physics debug view draws joint frames and limits from the same
            // render buffer as the colliders). Free while the scene's
            // visualization scale is zero, which is the off state.
            joint_->setConstraintFlag(::physx::PxConstraintFlag::eVISUALIZATION, true);
        }

        ~Joint() { detach(); }

        // Release the constraint NOW and forget it.
        //
        // Exists because "destroy the Joint before its world" is not something
        // its owner can always promise. Releasing a PxPhysics destroys every
        // joint it made, so a Joint that outlives its world holds a dangling
        // PxJoint and its destructor releases it a second time - a crash that
        // arrives whenever the last reference happens to go, which for a
        // scripted joint is whenever the collector next runs. Whoever DOES
        // know the world is still alive (PhysicsPlaySession::stop, which runs
        // before the world is dropped) calls this, and every reference still
        // outstanding is left holding something inert.
        //
        // Idempotent. After it, attached() is false and the accessors raise
        // rather than dereference.
        void detach() {
            if (!joint_) return;

            // Wake both sides BEFORE the constraint goes.
            //
            // A body held still by a joint falls asleep - that is PhysX doing
            // its job. Releasing the joint does not wake it, so a grasp that
            // let go left the part hanging in mid-air until something else
            // happened to touch it: the constraint was genuinely gone and the
            // object still did not fall, which reads as "release() did
            // nothing" and is the most confusing possible symptom. Their
            // equilibrium just changed; they are entitled to be re-solved.
            ::physx::PxRigidActor* actors[2] = {nullptr, nullptr};
            joint_->getActors(actors[0], actors[1]);
            for (auto* actor : actors) {
                if (!actor) continue;// that side is the world
                if (auto* link = actor->is<::physx::PxArticulationLink>()) {
                    // A link does not sleep on its own account - the whole
                    // articulation does.
                    link->getArticulation().wakeUp();
                } else if (auto* body = actor->is<::physx::PxRigidDynamic>()) {
                    // Kinematic bodies are driven, not solved; waking one is
                    // meaningless and PhysX warns about it.
                    if (!body->getRigidBodyFlags().isSet(::physx::PxRigidBodyFlag::eKINEMATIC)) {
                        body->wakeUp();
                    }
                }
            }

            joint_->release();
            joint_ = nullptr;
        }

        // False once detach() (or the world's teardown) has taken the
        // constraint away.
        [[nodiscard]] bool attached() const { return joint_ != nullptr; }
        Joint(const Joint&) = delete;
        Joint& operator=(const Joint&) = delete;

        // Retarget the drive along the motion axis (radians for Revolute,
        // metres for Prismatic; ignored by the other types, whose drive has no
        // scalar axis). For a script nudging a door or stepping a slider.
        void setDriveTarget(float value) {
            using namespace ::physx;
            auto* d6 = live()->is<PxD6Joint>();
            if (!d6) return;
            if (type_ == Type::Revolute) {
                d6->setDrivePosition(PxTransform(PxQuat(value, PxVec3(1, 0, 0))));
            } else if (type_ == Type::Prismatic) {
                d6->setDrivePosition(PxTransform(PxVec3(value, 0, 0)));
            }
        }

        void setDriveVelocity(float value) {
            using namespace ::physx;
            auto* d6 = live()->is<PxD6Joint>();
            if (!d6) return;
            if (type_ == Type::Revolute) {
                d6->setDriveVelocity(PxVec3(PxZero), PxVec3(value, 0, 0));
            } else if (type_ == Type::Prismatic) {
                d6->setDriveVelocity(PxVec3(value, 0, 0), PxVec3(PxZero));
            }
        }

        // True once the solver exceeded the break threshold; the constraint no
        // longer acts and never comes back.
        [[nodiscard]] bool broken() const {
            return live()->getConstraintFlags().isSet(::physx::PxConstraintFlag::eBROKEN);
        }

        // --- joint-space state, for encoders and scripts -------------------
        // The scalar coordinate along the motion axis: the twist angle in
        // radians (Revolute — and Spherical, whose twist is the one scalar it
        // has), the X displacement in metres (Prismatic), the anchor distance
        // (Distance). Zero for Fixed, which has no coordinate.
        [[nodiscard]] float position() const {
            using namespace ::physx;
            switch (type_) {
                case Type::Revolute:
                case Type::Spherical:
                    return live()->is<PxD6Joint>()->getTwistAngle();
                case Type::Prismatic:
                    return live()->getRelativeTransform().p.x;
                case Type::Distance:
                    return live()->is<PxDistanceJoint>()->getDistance();
                case Type::Fixed:
                    break;
            }
            return 0.f;
        }

        // Its rate: rad/s or m/s. PhysX reports the relative velocities in
        // actor0's CONSTRAINT frame, whose X is the motion axis — so the X
        // component is the joint rate directly. Distance projects the linear
        // velocity onto the anchor separation instead (the rate the tether
        // pays out at).
        [[nodiscard]] float velocity() const {
            using namespace ::physx;
            switch (type_) {
                case Type::Revolute:
                case Type::Spherical:
                    return live()->getRelativeAngularVelocity().x;
                case Type::Prismatic:
                    return live()->getRelativeLinearVelocity().x;
                case Type::Distance: {
                    const PxVec3 p = live()->getRelativeTransform().p;
                    const float d = p.magnitude();
                    if (d < 1e-6f) return 0.f;
                    return live()->getRelativeLinearVelocity().dot(p) / d;
                }
                case Type::Fixed:
                    break;
            }
            return 0.f;
        }

        // The force and torque the solver applied to hold the constraint on
        // the last step — what a force/torque sensor bolted across the joint
        // reads. Zero before the first step, and zero once broken: a broken
        // joint transmits nothing. (PxConstraint::getForce does NOT say zero
        // then — the broken constraint leaves the solver's active set and the
        // buffer freezes on the breaking step's value, a stale reading that
        // would otherwise be reported as live load forever. breakWrench()
        // is where that final value lives, on purpose.)
        void reactionForce(Vector3& force, Vector3& torque) const {
            if (broken()) {
                latchBreakWrench();
                force.set(0, 0, 0);
                torque.set(0, 0, 0);
                return;
            }
            ::physx::PxVec3 f(::physx::PxZero), t(::physx::PxZero);
            if (auto* constraint = live()->getConstraint()) constraint->getForce(f, t);
            force.set(f.x, f.y, f.z);
            torque.set(t.x, t.y, t.z);
        }

        // The wrench the solver applied on the BREAKING step — the true
        // failure load, necessarily past the authored threshold. Zero until
        // the joint breaks. Latched from the world's break callback when a
        // session is wired for it (see PhysxWorld::watchConstraintBreaks);
        // a standalone reader is covered by the lazy latch below either way.
        void breakWrench(Vector3& force, Vector3& torque) const {
            if (broken()) latchBreakWrench();
            force.copy(breakForce_);
            torque.copy(breakTorque_);
        }

        // Stamp the failure load, first writer wins. Called by whoever hears
        // the break callback with the event's wrench; harmless to call again.
        void latchBreakWrench(const Vector3& force, const Vector3& torque) const {
            if (breakLatched_) return;
            breakLatched_ = true;
            breakForce_.copy(force);
            breakTorque_.copy(torque);
        }

        [[nodiscard]] Type type() const { return type_; }
        [[nodiscard]] ::physx::PxJoint* raw() const { return joint_; }

    private:
        // The lazy half of the latch, for a reader with no break callback
        // wired: after the breaking step PxConstraint::getForce holds that
        // step's value — frozen, because a broken constraint is never solved
        // again — so the first post-break read still recovers the failure
        // load. Measured (ForceTorqueSensor_test pins this path), not
        // promised by PhysX, which is why the callback latch above is
        // preferred when available.
        void latchBreakWrench() const {
            if (breakLatched_) return;
            ::physx::PxVec3 f(::physx::PxZero), t(::physx::PxZero);
            if (auto* constraint = live()->getConstraint()) constraint->getForce(f, t);
            breakLatched_ = true;
            breakForce_.set(f.x, f.y, f.z);
            breakTorque_.set(t.x, t.y, t.z);
        }

        // The joint, or a raise. A detached Joint is a live C++ object with
        // nothing behind it, and saying so is much better than the alternative
        // - which is reading through a pointer PhysX freed.
        [[nodiscard]] ::physx::PxJoint* live() const {
            if (!joint_) {
                throw std::runtime_error(
                        "Joint: this constraint has been released - its world was torn down, "
                        "or it was detached explicitly");
            }
            return joint_;
        }

        ::physx::PxJoint* joint_ = nullptr;
        Type type_;
        // The latch is a measurement cache, not object state — mutable so the
        // const read paths above can fill it on first sight of the break.
        mutable bool breakLatched_ = false;
        mutable Vector3 breakForce_;
        mutable Vector3 breakTorque_;
    };

}// namespace threepp

#endif// THREEPP_PHYSX_JOINT_HPP
