// Contact sensor — "is this body touching anything, where, and how hard".
//
// The proprioceptive half of a bumper, a foot-contact switch, or a gripper's
// grasp detector. For legged locomotion this is the observation that tells a
// policy which feet are loaded; for manipulation it is what says the grasp
// closed on something.
//
// Two channels, because they answer different questions and behave differently:
//
//   inContact   a LATCHED state, true from the substep a touch is reported until
//               the substep it is reported lost. This is what a bumper switch
//               does, and it is the channel to trust for "am I standing on it",
//               because PhysX stops re-reporting a resting contact once the pair
//               falls asleep — points and force go quiet while the touch itself
//               is very much still there.
//   points/force what was actually observed during this sample interval. Rich
//               while the contact is live and changing, empty once asleep.
//
// Contact points are captured from inside fetchResults(), in the same substep
// that produced them, so a sensor sampling at the physics rate sees each
// substep's manifold exactly once. Under rate gating the observations of every
// substep since the previous sample are aggregated into one reading.

#ifndef THREEPP_SENSORS_CONTACTSENSOR_HPP
#define THREEPP_SENSORS_CONTACTSENSOR_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/Sensor.hpp"
#include "threepp/math/Vector3.hpp"

#include <PxPhysicsAPI.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace threepp {

    /// One manifold point, in world space, oriented relative to the sensor body.
    struct ContactPoint {
        Vector3 position{0.f, 0.f, 0.f};
        /// Unit normal pointing INTO the sensor's body (i.e. the direction the
        /// other body is pushing it).
        Vector3 normal{0.f, 0.f, 0.f};
        /// Normal impulse magnitude applied at this point over the substep (N*s).
        float impulse = 0.f;
        /// The actor this point was against. Non-owning, and only valid until
        /// that actor is released — compare it, do not dereference it later.
        ::physx::PxRigidActor* other = nullptr;
    };

    /**
     * One contact reading. `t` is the accumulated simulation time (s) at the end
     * of the sampled substep.
     */
    struct ContactSample {
        static constexpr std::size_t maxPoints = 8;

        double t = 0.0;
        /// Latched touch state — see the header comment. Survives sleeping.
        bool inContact = false;
        /// Set on the sample where the latch changed, so an edge is never missed
        /// even if the touch began and ended inside one sample interval.
        bool touchBegan = false;
        bool touchEnded = false;
        /// Total contact force over the interval (N): sum of normal impulses
        /// divided by the elapsed time. Zero while asleep even if inContact.
        Vector3 force{0.f, 0.f, 0.f};
        /// Manifold points observed this interval, capped at maxPoints.
        std::uint32_t pointCount = 0;
        /// How many were seen before the cap — pointCount == min(this, maxPoints).
        std::uint32_t observedPoints = 0;
        std::array<ContactPoint, maxPoints> points{};
    };

    class ContactSensor: public Sensor {

    public:
        /**
         * @param node   The attachment node; the rigid body found in its ancestry
         *               is the body whose contacts are reported.
         * @param rateHz Sample rate (Hz); 0 = every physics substep.
         * @param bufferCapacity Ring-buffer depth (oldest dropped on overflow).
         *               A ContactSample is comparatively fat (it inlines its
         *               manifold rather than allocating), so this defaults well
         *               below the IMU's.
         */
        explicit ContactSensor(Object3D& node, double rateHz = 0.0, std::size_t bufferCapacity = 256)
            : Sensor(node, rateHz), ring_(bufferCapacity) {}

        ~ContactSensor() override {
            // Registered-but-not-unregistered is a caller error, but leaving a
            // live callback pointing at a destroyed sensor turns it into a crash
            // in the step loop, so unhook defensively.
            detach();
        }

        void onRegister(PhysxWorld& world) override {
            world_ = &world;
            actor_ = world.findActor(node());
            if (!actor_) {
                throw std::invalid_argument(
                        "ContactSensor: the attachment Object3D has no PhysxWorld-managed rigid "
                        "body in its ancestry. Attach the sensor to a mesh added via world.add(...) "
                        "(or a child of it) before registering.");
            }
            watch_ = world.watchContacts(actor_, [this](const ContactEvent& e) { accumulate(e); });
            reset();
        }

        void onUnregister() override {
            detach();
        }

        void onActorRemoved(::physx::PxRigidActor* actor) override {
            if (actor == actor_) {
                // The world drops its own watcher for this actor; just forget ours
                // so we do not try to unwatch a handle it already discarded.
                watch_ = 0;
                actor_ = nullptr;
                touching_.clear();
                clearPending();
                return;
            }
            // Something we were touching went away. PhysX does not follow a
            // removal with a TOUCH_LOST, so without this the sensor would report
            // a permanent touch against a body that no longer exists.
            if (releaseTouch(actor, /*all*/ true)) pendingEnded_ = true;
        }

        /// True once registered against a live rigid body.
        [[nodiscard]] bool attached() const { return actor_ != nullptr; }

        /// Re-arm: clear the buffer, the latch and any pending observations.
        void reset() {
            touching_.clear();
            clearPending();
            ring_.clear();
            resetTiming();
        }

        void sample(double dt, double simTime) override {
            ContactSample s;
            s.t = simTime;
            s.inContact = inContact();
            s.touchBegan = pendingBegan_;
            s.touchEnded = pendingEnded_;
            s.pointCount = static_cast<std::uint32_t>(pendingCount_);
            s.observedPoints = pendingObserved_;
            for (std::size_t i = 0; i < pendingCount_; ++i) s.points[i] = pending_[i];

            // Impulse (N*s) accumulated over the interval -> mean force (N).
            if (dt > 0.0) {
                s.force = pendingImpulse_;
                s.force.multiplyScalar(static_cast<float>(1.0 / dt));
            }

            ring_.push(s);
            clearPending();
        }

        // --- read side (non-blocking) --------------------------------------

        [[nodiscard]] std::optional<ContactSample> latest() const { return ring_.latest(); }
        void drain(std::vector<ContactSample>& out) { ring_.drain(out); }
        [[nodiscard]] std::size_t available() const { return ring_.size(); }

        /// Current latched touch state, without going through the buffer — the
        /// cheap read for a control loop that only wants a foot-down boolean.
        [[nodiscard]] bool inContact() const { return !touching_.empty(); }

        /// How many distinct bodies are currently being touched.
        [[nodiscard]] std::size_t touchCount() const { return touching_.size(); }

    private:
        // Currently-touching bodies, refcounted by shape pair: one body can touch
        // this one through several shapes, and a LOST on one of them must not
        // clear a touch the others are still holding. A plain boolean latch got
        // this wrong in both directions — it cleared while still touching, and it
        // never cleared when the other body was removed outright.
        struct Touch {
            ::physx::PxRigidActor* actor;
            int pairs;
        };

        void acquireTouch(::physx::PxRigidActor* other) {
            for (auto& t: touching_) {
                if (t.actor == other) {
                    ++t.pairs;
                    return;
                }
            }
            touching_.push_back({other, 1});
        }

        // Drop one shape pair against `other` (or every pair, when the body is
        // gone outright). Returns true if the last pair went away.
        bool releaseTouch(const ::physx::PxRigidActor* other, bool all) {
            for (auto it = touching_.begin(); it != touching_.end(); ++it) {
                if (it->actor != other) continue;
                if (all || --it->pairs <= 0) {
                    touching_.erase(it);
                    return true;
                }
                return false;
            }
            return false;
        }

        void detach() {
            if (world_ && watch_) world_->unwatchContacts(watch_);
            // Leave the actor's reporting bit alone: another sensor or a caller's
            // own watchContacts() may still need it, and PhysxWorld exposes
            // setContactReporting() for turning it off deliberately.
            watch_ = 0;
            world_ = nullptr;
            actor_ = nullptr;
            // No longer being told about touches, so any latched state is stale.
            touching_.clear();
        }

        void clearPending() {
            pendingCount_ = 0;
            pendingObserved_ = 0;
            pendingBegan_ = false;
            pendingEnded_ = false;
            pendingImpulse_.set(0.f, 0.f, 0.f);
        }

        // Called from inside fetchResults(), once per reported pair per substep.
        void accumulate(const ContactEvent& e) {
            if (e.touchFound) {
                acquireTouch(e.other);
                pendingBegan_ = true;
            }
            if (e.touchLost) {
                if (releaseTouch(e.other, /*all*/ false)) pendingEnded_ = true;
            }

            for (::physx::PxU32 i = 0; i < e.pointCount; ++i) {
                const auto& p = e.points[i];
                ++pendingObserved_;

                // PhysX expresses a pair's normal and impulse relative to the
                // pair's FIRST actor; flip when this sensor is the second so
                // every reading is oriented the same way regardless of the order
                // PhysX happened to put the pair in.
                const float sign = e.selfIsFirst ? 1.f : -1.f;
                const Vector3 n(p.normal.x * sign, p.normal.y * sign, p.normal.z * sign);
                const Vector3 imp(p.impulse.x * sign, p.impulse.y * sign, p.impulse.z * sign);

                pendingImpulse_.add(imp);

                if (pendingCount_ < ContactSample::maxPoints) {
                    ContactPoint& out = pending_[pendingCount_++];
                    out.position.set(p.position.x, p.position.y, p.position.z);
                    out.normal = n;
                    out.impulse = imp.length();
                    out.other = e.other;
                }
            }
        }

        PhysxWorld* world_ = nullptr;
        ::physx::PxRigidActor* actor_ = nullptr;
        PhysxWorld::ContactHandle watch_ = 0;

        std::vector<Touch> touching_;

        // Observations accumulated since the last sample().
        std::array<ContactPoint, ContactSample::maxPoints> pending_{};
        std::size_t pendingCount_ = 0;
        std::uint32_t pendingObserved_ = 0;
        bool pendingBegan_ = false;
        bool pendingEnded_ = false;
        Vector3 pendingImpulse_{0.f, 0.f, 0.f};

        SensorRing<ContactSample> ring_;
    };

}// namespace threepp

#endif// THREEPP_SENSORS_CONTACTSENSOR_HPP
