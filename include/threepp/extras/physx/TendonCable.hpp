// A tendon that behaves like a cable: routed over via points, pull-only, with real
// tension and optional routing friction.
//
// WHY THIS EXISTS ALONGSIDE ArticulationTendon.hpp. PhysX ships two tendon primitives
// and neither is a routed cable. Measured, not inferred (python/examples/tendon_probe.py,
// each number reproducible from that script):
//
//   * A spatial tendon's interior attachments set the LENGTH but exert NO FORCE, exactly
//     as PxArticulationTendon.h:405-407 says. Measured on a two-link finger bent 0.9 rad:
//     the generalized force it produces matches the gradient of the routed length taken
//     with the via point FROZEN in world space to 0.07 deg, and sits 21.06 deg away from
//     a true routed cable. So a flexor threaded through A1-A5 gets no transverse pulley
//     reaction, and its moment arms distal of the first joint are fabricated.
//   * A fixed tendon is joint-space only: L = sum(c_i q_i), no geometry at all.
//
// This class is the third option and it is exact. At every via point it applies the
// force a frictionless pulley actually applies -- the vector sum of the two adjacent
// segment tensions -- so the generalized force is -T dL/dq by virtual work, with no
// approximation anywhere. Measured against the analytic gradient on the same rig:
// 0.02% of torque magnitude and 0.001 deg of direction.
//
// WHAT THAT BUYS, beyond correctness:
//   * PULL-ONLY for free. Tension is clamped at zero in one line, so a slack cable
//     produces literally nothing and can never push a finger open. PhysX's spring is
//     bilateral and needs the limit-only construction to fake this.
//   * A REAL TENSION NUMBER. PhysX exposes no tendon force readback at all --
//     PxArticulationCacheFlag has no tendon entry and neither does the direct-GPU read
//     enum -- so with an SDK tendon the value an RL policy observes can only ever be a
//     model prediction. Here the tension IS the applied quantity.
//   * ROUTING FRICTION, the thing the mechanism is actually known for: pulling harder
//     on a cable does not cleanly mean more torque at the joint, because each pulley
//     takes an unknown share on the way. See setFriction().
//   * Any body layout, any number of via points, no rebuild to re-route.
//
// WHAT IT COSTS. The forces go in through PxRigidBody::addForce/addTorque, which PhysX
// rejects under PxSceneFlag::eENABLE_DIRECT_GPU_API. So a cable-driven hand runs on the
// CPU path; batched GPU RL over thousands of envs would need the SDK's own tendons (and
// their fidelity caveats) or a GPU implementation of this. Stated plainly because it is
// the one real trade.
//
// CONTROL MODES. Both are unilateral; pick by which end of the drivetrain you model.
//   Tension: the command IS the cable tension, i.e. an ideal motor with a torque loop
//            closed around it. Cannot go unstable -- there is no stiffness to explode.
//   Length:  the command is the SPOOLED length, and tension follows from how far the
//            route is stretched past it, T = k*(L - L_cmd) + c*Ldot, floored at zero.
//            This is a real series-elastic drivetrain: cable stretch is a property of
//            actual tendons, not an artefact. Explicit, so k is bounded by the substep.
//
// LIFETIME. The cable registers a pre-substep callback on the world and unregisters in
// its destructor; it must not outlive the world or the links it routes over.
#ifndef THREEPP_PHYSX_TENDONCABLE_HPP
#define THREEPP_PHYSX_TENDONCABLE_HPP

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace threepp {

    class TendonCable {
    public:
        enum class Mode {
            Tension,// the command is the cable tension (N)
            Length  // the command is the spooled length (m); tension follows from stretch
        };

        explicit TendonCable(PhysxWorld& world, Mode mode = Mode::Tension)
            : world_(world), mode_(mode) {
            // Registered pre-substep so the route is evaluated at the true physics rate,
            // not the frame rate: a cable sampled once per rendered frame would apply a
            // stale direction through every substep of a catch-up burst.
            handle_ = world_.onPreSubstep([this](float dt) { apply(dt); });
        }
        ~TendonCable() { world_.removeSubstepCallback(handle_); }
        TendonCable(const TendonCable&) = delete;
        TendonCable& operator=(const TendonCable&) = delete;

        // Add a via point on `link` at `localOffset` in the link's ACTOR frame, in order
        // from the actuator end to the insertion. The first and last are the anchor and
        // the insertion; everything between is a pulley.
        void addViaPoint(const ArticulationLink& link, const Vector3& localOffset) {
            links_.push_back(link);
            local_.push_back(localOffset);
        }

        // --- commands ---------------------------------------------------------------
        // Tension mode: the cable tension in newtons. Negative is clamped to zero rather
        // than rejected, because a policy or a PD loop will ask for negative tension and
        // the physical answer is "the cable goes slack", not "error".
        void setTension(float t) { cmdTension_ = std::max(0.f, t); }
        // Length mode: the spooled length in metres. Pull the cable by REDUCING it.
        void setSpoolLength(float l) { spool_ = l; }
        void setStiffness(float k) { k_ = k; }
        void setDamping(float c) { c_ = c; }

        // Coulomb friction at the pulleys, as the capstan coefficient mu.
        //
        // A cable wrapping a pulley through angle theta comes out the far side carrying
        // T*exp(-mu*theta) when it is being pulled in -- the pulley eats the difference.
        // This is why a tendon hand is hard to control precisely: the tension that
        // reaches the fingertip is not the tension the motor applied, and the shortfall
        // depends on the posture through the wrap angles. mu = 0 (the default) is the
        // ideal frictionless cable, and is the configuration the exactness measurement
        // above was made in. Tendon-in-sheath values are typically 0.1-0.4.
        void setFriction(float mu) { mu_ = std::max(0.f, mu); }

        // --- state ------------------------------------------------------------------
        // Total routed length, recomputed from the live link poses.
        [[nodiscard]] float length() const {
            const auto p = points();
            float l = 0.f;
            for (std::size_t i = 1; i < p.size(); ++i) l += (p[i] - p[i - 1]).length();
            return l;
        }
        // Tension at the ACTUATOR end, in newtons. Genuinely the applied value, not a
        // reconstruction: this is the number that went into the last substep.
        [[nodiscard]] float tension() const { return tension_; }
        // Tension at the INSERTION end. Differs from tension() only when mu > 0, and the
        // ratio between them is exactly what the routing swallowed.
        [[nodiscard]] float tipTension() const { return tipTension_; }
        [[nodiscard]] Mode mode() const { return mode_; }
        [[nodiscard]] std::size_t numViaPoints() const { return links_.size(); }

    private:
        [[nodiscard]] std::vector<Vector3> points() const {
            std::vector<Vector3> p;
            p.reserve(links_.size());
            for (std::size_t i = 0; i < links_.size(); ++i) p.push_back(links_[i].worldPoint(local_[i]));
            return p;
        }

        void apply(float dt) {
            if (links_.size() < 2) return;
            const auto p = points();
            const std::size_t n = p.size();

            std::vector<Vector3> seg(n - 1);
            float len = 0.f;
            for (std::size_t i = 0; i + 1 < n; ++i) {
                Vector3 d = p[i + 1] - p[i];
                const float m = d.length();
                len += m;
                if (m > 1e-9f) d.divideScalar(m);
                seg[i] = d;
            }

            float T;
            if (mode_ == Mode::Tension) {
                T = cmdTension_;
            } else {
                // Rate from the length itself rather than from link velocities: it is the
                // derivative of the quantity the spring actually acts on, so the damper
                // can never fight a motion the spring does not see.
                const float rate = (haveLast_ && dt > 0.f) ? (len - lastLen_) / dt : 0.f;
                T = std::max(0.f, k_ * (len - spool_) + c_ * rate);
            }
            lastLen_ = len;
            haveLast_ = true;
            tension_ = T;

            // Per-segment tension. Frictionless, every segment carries T. With friction,
            // each pulley sheds a factor exp(-mu*theta) where theta is the turn angle
            // between the incoming and outgoing segments, so the insertion sees less than
            // the motor applied.
            std::vector<float> segT(n - 1, T);
            if (mu_ > 0.f) {
                for (std::size_t i = 1; i + 1 < n; ++i) {
                    const float c = std::clamp(seg[i - 1].dot(seg[i]), -1.f, 1.f);
                    const float theta = std::acos(c);// turn angle at via point i
                    segT[i] = segT[i - 1] * std::exp(-mu_ * theta);
                }
            }
            tipTension_ = segT.back();

            // The free-body diagram of a frictionless pulley: the anchor is pulled toward
            // the next point, the insertion toward the previous, and every via point takes
            // the vector sum of its two adjacent segment tensions. With equal segment
            // tensions these sum to zero, so the cable injects no net momentum, and the
            // generalized force is exactly -T dL/dq by virtual work.
            //
            // addForceAtPos, not addForce: a force through the centre of mass produces no
            // torque about the link's own joint and would drive nothing.
            const auto scaled = [](Vector3 v, float s) { return v.multiplyScalar(s); };
            links_[0].addForceAtPos(scaled(seg[0], segT[0]), p[0]);
            for (std::size_t i = 1; i + 1 < n; ++i)
                links_[i].addForceAtPos(scaled(seg[i], segT[i]) - scaled(seg[i - 1], segT[i - 1]), p[i]);
            links_[n - 1].addForceAtPos(scaled(seg[n - 2], -segT[n - 2]), p[n - 1]);
        }

        PhysxWorld& world_;
        Mode mode_;
        PhysxWorld::SubstepHandle handle_{};
        std::vector<ArticulationLink> links_;
        std::vector<Vector3> local_;
        float cmdTension_{0.f};
        float spool_{0.f}, k_{0.f}, c_{0.f}, mu_{0.f};
        float lastLen_{0.f}, tension_{0.f}, tipTension_{0.f};
        bool haveLast_{false};
    };

}// namespace threepp

#endif
