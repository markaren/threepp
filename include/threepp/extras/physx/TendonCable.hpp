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
#include <numbers>
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
            nodes_.push_back({Node::Via, link, localOffset, Vector3(0, 0, 1), 0.f, Vector3(1, 0, 0), false});
        }

        // Add a WRAP: between the previous and the next via point the cable runs around
        // a cylinder of `radius` centred at `localCentre` on `link`, with its axis along
        // `localAxis` (both in the link's actor frame). Put the cylinder on the joint,
        // co-axial with the hinge, and the radius is the tendon's standoff.
        //
        // WHY A WRAP AND NOT MORE VIA POINTS. A via point is welded to its link, so a
        // cable strung between via points on either side of a flexing joint chords
        // straight across it, and once the distal point swings behind the joint axis the
        // moment REVERSES. Measured on the index finger of this hand: an FDP whose MCP
        // moment arm ran +9.98 mm extended to -5.03 mm at 89 deg of flexion -- a flexor
        // that extends the knuckle, in exactly the posture a grasp lives in. No number of
        // via points fixes it, because the real tendon's contact point SLIDES along the
        // pulley as the joint moves, and a welded point cannot slide.
        //
        // Resolved by discretising the arc into via points regenerated every substep, so
        // the exact free-body force law above is reused unchanged: the wrap adds
        // geometry, not a second force model. The moment arm about the cylinder's own
        // axis then comes out at the radius and STAYS there through the full range, which
        // is the textbook result for a wrapped tendon and the reason biomechanics quotes
        // a pulley's moment arm as a single number.
        // `sideHint` (link frame, from the centre outward) says WHICH SIDE of the
        // cylinder the cable runs on -- volar for a flexor, dorsal for an extensor. It is
        // required rather than inferred because the two are not distinguishable from the
        // endpoints alone, and "take the shorter way round" is simply wrong for a tendon:
        // a sheath holds it on one definite side whether or not that side is shorter.
        // Inferring it put this hand's extensor on the volar side of every joint, which
        // measured as an extensor with a +7.53 mm FLEXION arm at the MCP.
        // `sheathed` forces contact even where the free path would clear the pulley.
        // Leave it FALSE for a flexor: one runs on the concave side, lifts off its pulley
        // and bowstrings, and forcing it to follow lengthens its path by r*theta and turns
        // it into an extensor (measured, 58.5 -> 73.2 mm over 105 deg). Set it TRUE for an
        // extensor, which runs on the convex side and is genuinely held against the joint
        // by its sheath -- without it the wrap DISENGAGES at deep flexion, the chord takes
        // over and the moment arm reverses, so the extensor flexes the joint it is meant
        // to straighten and the finger cannot come back out of a fist.
        void addWrap(const ArticulationLink& link, const Vector3& localCentre,
                     const Vector3& localAxis, float radius, const Vector3& sideHint,
                     bool sheathed = false) {
            if (nodes_.empty())
                throw std::runtime_error("TendonCable.add_wrap: a wrap needs a via point before it");
            nodes_.push_back({Node::Wrap, link, localCentre, localAxis, radius, sideHint, sheathed});
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
        [[nodiscard]] std::size_t numNodes() const { return nodes_.size(); }
        // Points in the RESOLVED path, wrap arcs included -- so it grows as a joint flexes.
        [[nodiscard]] std::size_t numPathPoints() const { return points().size(); }

        // The whole resolved path in world space, actuator end first. This is what the
        // cable IS at this instant: via points plus whatever arc a wrap contributed. Draw
        // it and the routing stops being a table of numbers -- a cable on the wrong side
        // of a joint, or one cutting a corner it should wrap, is obvious on sight and
        // very hard to see in a moment-arm column.
        [[nodiscard]] std::vector<Vector3> path() const { return points(); }

    private:
        struct Node {
            enum Kind { Via,
                        Wrap } kind;
            ArticulationLink link;
            Vector3 local;  // via point, or cylinder centre
            Vector3 axis;   // cylinder axis (wrap only)
            float radius;   // wrap only
            Vector3 side;   // which side of the cylinder the cable runs on (wrap only)
            bool sheathed;  // wrap even where the free path would clear the pulley
        };

        // Resolve the node list into the actual cable path: a flat list of world points,
        // each tagged with the link it presses on. Wrap arcs become extra points on the
        // wrapped link, so everything downstream sees one uniform polyline.
        void resolve(std::vector<Vector3>& pts, std::vector<const ArticulationLink*>& owner) const {
            pts.clear();
            owner.clear();
            for (std::size_t i = 0; i < nodes_.size(); ++i) {
                const Node& nd = nodes_[i];
                if (nd.kind == Node::Via) {
                    pts.push_back(nd.link.worldPoint(nd.local));
                    owner.push_back(&nd.link);
                    continue;
                }
                // A wrap sits between the previous resolved point and the next via node.
                if (pts.empty() || i + 1 >= nodes_.size()) continue;
                const Node& nxt = nodes_[i + 1];
                if (nxt.kind != Node::Via) continue;
                arc(nd, pts.back(), nxt.link.worldPoint(nxt.local), pts, owner);
            }
        }

        // Planar tangent-arc-tangent around one cylinder. Everything is solved in the
        // plane through the cylinder centre perpendicular to its axis, which is exact for
        // a hinge because the cable and both neighbours lie in that plane by construction.
        static void arc(const Node& w, const Vector3& P, const Vector3& Q,
                        std::vector<Vector3>& pts, std::vector<const ArticulationLink*>& owner) {
            const Vector3 C = w.link.worldPoint(w.local);
            Vector3 n = w.link.worldPoint(w.local + w.axis) - C;// the axis, rotated into world
            const float nl = n.length();
            if (nl < 1e-9f) return;
            n.divideScalar(nl);

            // In-plane components of the two neighbours relative to the centre.
            auto flat = [&](const Vector3& X) {
                Vector3 d = X - C;
                Vector3 along = n;
                along.multiplyScalar(d.dot(n));
                return d - along;
            };
            const Vector3 a = flat(P), b = flat(Q);
            const float ra = a.length(), rb = b.length();
            const float r = w.radius;
            // A neighbour inside the cylinder has no tangent line; leave the path straight
            // rather than emitting a NaN.
            if (ra <= r * 1.001f || rb <= r * 1.001f) return;

            // Signed angles in a right-handed 2D basis (u, v) spanning the plane.
            Vector3 u = a;
            u.divideScalar(ra);
            Vector3 v = n.clone().cross(u);
            const float angB = std::atan2(b.dot(v), b.dot(u));
            const float ta = std::acos(std::clamp(r / ra, -1.f, 1.f));// tangent offset from a
            const float tb = std::acos(std::clamp(r / rb, -1.f, 1.f));
            constexpr float kPi = 3.14159265358979323846f;

            // The side the sheath holds the cable on, in the same 2D basis.
            Vector3 hw = w.link.worldPoint(w.local + w.side) - C;
            const float hx = hw.dot(u), hy = hw.dot(v);
            const float hang = std::atan2(hy, hx);

            // Two ways round; take the one whose arc actually sits on that side.
            float bestT1 = 0.f, bestSweep = 0.f, bestScore = -2.f;
            bool found = false;
            // Tangent construction. With the basis anchored so P sits at angle 0, the two
            // tangent points from P are at +-ta and those from Q at angB +- tb. A cable
            // that leaves P at +ta must travel COUNTER-clockwise and arrive at angB - tb;
            // the clockwise branch is the mirror. Getting these two signs backwards makes
            // both branches fail the consistency test below, and the wrap then silently
            // does nothing at all -- the cable stays a chord and every moment arm reverts
            // to the welded-via-point behaviour, with no error anywhere.
            for (const float sg : {1.f, -1.f}) {
                const float t1 = sg * ta;
                const float t2 = angB - sg * tb;
                // The two tangent PAIRS are the choice; the sweep direction is not free
                // once a pair is picked. Between two tangent points of the same pair a
                // cable can never wrap more than half the circle, so the short way is the
                // only physical one -- and finger joints top out near 115 deg, well
                // inside that. Normalising the other way (into each branch's own
                // direction) made the selection take the LONG way round the far side, and
                // the moment arm came out at the pulley radius with the WRONG SIGN.
                float sweep = std::fmod(t2 - t1 + kPi, 2.f * kPi);
                if (sweep < 0.f) sweep += 2.f * kPi;
                sweep -= kPi;
                float mid = t1 + 0.5f * sweep;
                float d = mid - hang;
                while (d > kPi) d -= 2.f * kPi;
                while (d < -kPi) d += 2.f * kPi;
                const float score = std::cos(d);// +1 when the arc midpoint is on the hinted side
                if (score > bestScore) {
                    bestScore = score;
                    bestT1 = t1;
                    bestSweep = sweep;
                    found = true;
                }
            }
            if (!found) return;
            const float t1 = bestT1;
            const float sweep = bestSweep;

            // Contact only where the free path would actually cut through the pulley --
            // unless the caller says the tendon is SHEATHED there and cannot leave it.
            // This is not an optimisation, it is the physics, and it is what separates a
            // flexor from an extensor. A cable on the CONCAVE side of a closing joint
            // (a flexor, volar) lifts OFF its pulley and bowstrings: its chord shortens
            // as the joint flexes, which is why pulling it flexes at all, and the pulley
            // acts as a CAP on how far it can bow rather than as a surface it follows.
            // Force it to wrap anyway and the volar path LENGTHENS by r*theta, turning
            // the flexor into an extensor -- measured, on the reference geometry: path
            // 58.5 -> 73.2 mm over 105 deg of flexion, moment arm +8 mm where the chord
            // gives -8. A cable on the CONVEX side (an extensor, dorsal) genuinely does
            // wrap, and there this branch engages and holds the arm at the radius.
            if (!w.sheathed && distancePointSegment(C, P, Q) >= r) return;

            const int steps = std::max(2, static_cast<int>(std::ceil(std::fabs(sweep) / 0.25f)));
            for (int k = 0; k <= steps; ++k) {
                const float th = t1 + sweep * (static_cast<float>(k) / static_cast<float>(steps));
                Vector3 uu = u;
                uu.multiplyScalar(r * std::cos(th));
                Vector3 vv = v;
                vv.multiplyScalar(r * std::sin(th));
                pts.push_back(C + uu + vv);
                owner.push_back(&w.link);
            }
        }

        static float distancePointSegment(const Vector3& C, const Vector3& P, const Vector3& Q) {
            Vector3 d = Q - P;
            const float dd = d.dot(d);
            if (dd < 1e-12f) return (C - P).length();
            const float t = std::clamp((C - P).dot(d) / dd, 0.f, 1.f);
            Vector3 s = d;
            s.multiplyScalar(t);
            return (C - (P + s)).length();
        }

        [[nodiscard]] std::vector<Vector3> points() const {
            std::vector<Vector3> p;
            std::vector<const ArticulationLink*> o;
            resolve(p, o);
            return p;
        }

        void apply(float dt) {
            if (nodes_.size() < 2) return;
            std::vector<Vector3> p;
            std::vector<const ArticulationLink*> owner;
            resolve(p, owner);
            const std::size_t n = p.size();
            if (n < 2) return;

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
            const auto push = [&](std::size_t i, const Vector3& f) {
                const_cast<ArticulationLink*>(owner[i])->addForceAtPos(f, p[i]);
            };
            push(0, scaled(seg[0], segT[0]));
            for (std::size_t i = 1; i + 1 < n; ++i)
                push(i, scaled(seg[i], segT[i]) - scaled(seg[i - 1], segT[i - 1]));
            push(n - 1, scaled(seg[n - 2], -segT[n - 2]));
        }

        PhysxWorld& world_;
        Mode mode_;
        PhysxWorld::SubstepHandle handle_{};
        std::vector<Node> nodes_;
        float cmdTension_{0.f};
        float spool_{0.f}, k_{0.f}, c_{0.f}, mu_{0.f};
        float lastLen_{0.f}, tension_{0.f}, tipTension_{0.f};
        bool haveLast_{false};
    };

}// namespace threepp

#endif
