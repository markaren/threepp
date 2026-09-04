// PhysX articulation tendons — the SDK's two coupling elements, wrapped.
//
// A tendon-driven robot puts its motors in the base and pulls the moving parts
// with cables (the reason a dexterous hand can exist at all: a motor at a finger
// joint would be mass at the fastest-moving end of the arm). PhysX models that
// with two very different primitives, and choosing between them is the whole
// design decision, so both are here and the difference is spelled out.
//
// ---------------------------------------------------------------------------
// SpatialTendon — a geometric cable
// ---------------------------------------------------------------------------
// A tree of attachment points, each pinned at an offset in some link's ACTOR
// frame. Length is the coefficient-weighted sum of the segment lengths along the
// path from the root attachment to a leaf:
//
//   L_leaf = sum over the path  c_a * || p_a - p_parent(a) ||
//
// and the spring acts on L + offset against the leaf's restLength.
//
// THE CAVEAT THAT DECIDES EVERYTHING. Per PxArticulationTendon.h:401-408, a
// sub-tendon "applies forces at the leaf, and an equal but opposing force at the
// root", and attachments in between "do not exert any force on the articulation,
// but define the geometry of the tendon from which the length is computed".
//
// So the line of action is the ROOT-to-LEAF CHORD, not the routed polyline. A
// real cable pressing on the A2 pulley of a proximal phalanx pushes that phalanx
// — that transverse reaction is where a flexor's moment arm at the joint it
// spans comes from. A single multi-attachment spatial tendon does not produce
// it: routing changes the length sum (hence the magnitude) but never the
// direction. Every moment arm distal of the first joint is therefore fabricated.
//
// The fix, when the routing must be mechanically real, is one two-attachment
// tendon PER SEGMENT: with exactly two attachments the chord IS the segment, so
// each applies a correct equal-and-opposite pair at its own two links, and a
// via-point link receives the vector sum of its two adjacent segment forces —
// which is the free-body diagram of a frictionless pulley. See makeSegment().
//
// PULL-ONLY. The spring is bilateral: with stiffness > 0 and L < restLength it
// PUSHES, i.e. a flexor that shoves the finger open. A cable cannot do that. The
// unilateral construction uses the SDK's own one-sided limit machinery instead:
//
//   setStiffness(0)                      // no bilateral spring at all
//   setDamping(0)                        // see the damping note below
//   setLimitStiffness(k)                 // this is the cable's axial stiffness
//   leaf->setLimitParameters({-FLT_MAX, l_taut})
//
// Force only when L > l_taut; exactly zero when slack. That the limits are a
// pair of independently gated one-sided constraints is what makes the SDK's own
// documented default — (PX_MAX_F32, -PX_MAX_F32), described at
// PxArticulationTendon.h:90 as "an invalid configuration that can only work if
// stiffness is zero" — survivable at all.
//
// TWO THINGS THE HEADER DOES NOT SETTLE, both measured by the probe in
// python/examples/tendon_probe.py rather than assumed here:
//   * Do the LIMITS see `offset`? The spring block says rest is compared against
//     "accumulated length plus the tendon offset" (:528-529); the limit block
//     says limits act on "the accumulated length" (:551). If limits ignore
//     offset then setOffset cannot actuate a limit-based cable and the taut
//     length is the motor instead.
//   * Is `damping` still bilateral when stiffness is 0? It is documented as
//     acting on "both the tendon length and tendon-length limits" (:307), which
//     would make it resist shortening as well as lengthening — a viscous strut,
//     not a cable. Default to 0 and put damping in the joint, where its sign is
//     unambiguous.
//
// ---------------------------------------------------------------------------
// FixedTendon — a joint-space coupling
// ---------------------------------------------------------------------------
// No geometry at all: length is a linear combination of joint positions,
// L = sum(c_i * q_i), so a spring on L couples the joints it spans. This is the
// honest way to express a mechanically coupled linkage (the DIP/PIP coupling a
// real extensor hood enforces) and the only one of the two that survives the
// direct-GPU path cleanly. Its "moment arms" are the coefficients — prescribed,
// not emergent, which is exactly the trade against SpatialTendon.
//
// The joints must be directly connected in the articulation (:466-471).
//
// RECIPCOEFFICIENT. The header calls it "the scale that the tendon's response is
// multiplied by when applying to this tendon joint" and notes it is "commonly
// expected to be 1/coefficient" (:210-215). Power balance says otherwise: with
// L = sum(c_i q_i), F * Ldot = sum(tau_i * qdot_i) forces tau_i = F * c_i, so
// the energetically consistent multiplier is c_i, NOT 1/c_i. The two differ by
// c^2 — four orders of magnitude at a 10 mm moment arm — so this is measured,
// not assumed; see the probe.
//
// ---------------------------------------------------------------------------
// LIFETIME. A tendon belongs to its articulation: "When an articulation is
// released, its attached tendons are automatically released." These are
// non-owning handles, valid only while the articulation lives. Creation is
// forbidden once the articulation is in a scene (:416, :481), so build every
// tendon BEFORE Articulation::finalize().
#ifndef THREEPP_PHYSX_ARTICULATIONTENDON_HPP
#define THREEPP_PHYSX_ARTICULATIONTENDON_HPP

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/math/Vector3.hpp"

#include <PxPhysicsAPI.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace threepp {

    // One attachment point of a spatial tendon. Non-owning; released with its tendon.
    //
    // restLength and the limits are LEAF-ONLY properties — PhysX silently ignores
    // them on an interior attachment ("Setting this on non-leaf attachments has no
    // effect", PxArticulationTendon.h:66, :91). Silently, so these setters check
    // isLeaf() and throw instead of letting a mis-built routing look configured.
    class SpatialAttachment {
    public:
        explicit SpatialAttachment(::physx::PxArticulationAttachment* a) : a_(a) {}

        void setRestLength(float l) {
            leafOnly("set_rest_length");
            a_->setRestLength(l);
        }
        [[nodiscard]] float restLength() const { return a_->getRestLength(); }

        // The cable's taut length: force only once the tendon is longer than this.
        // Pull the cable by REDUCING it.
        void setTautLength(float l) {
            leafOnly("set_taut_length");
            ::physx::PxArticulationTendonLimit lim;
            lim.lowLimit = -std::numeric_limits<float>::max();// never active: a cable cannot push
            lim.highLimit = l;
            a_->setLimitParameters(lim);
        }
        void setLimits(float low, float high) {
            leafOnly("set_limits");
            ::physx::PxArticulationTendonLimit lim;
            lim.lowLimit = low;
            lim.highLimit = high;
            a_->setLimitParameters(lim);
        }
        [[nodiscard]] std::pair<float, float> limits() const {
            const auto l = a_->getLimitParameters();
            return {l.lowLimit, l.highLimit};
        }

        void setCoefficient(float c) { a_->setCoefficient(c); }
        [[nodiscard]] float coefficient() const { return a_->getCoefficient(); }
        void setRelativeOffset(const Vector3& v) { a_->setRelativeOffset(toPxVec3(v)); }
        [[nodiscard]] Vector3 relativeOffset() const { return fromPxVec3(a_->getRelativeOffset()); }
        [[nodiscard]] bool isLeaf() const { return a_->isLeaf(); }

        [[nodiscard]] ::physx::PxArticulationAttachment* raw() const { return a_; }

    private:
        void leafOnly(const char* what) const {
            if (!a_->isLeaf())
                throw std::runtime_error(std::string("SpatialAttachment.") + what +
                                         ": PhysX ignores this on a non-leaf attachment - "
                                         "only a leaf defines a sub-tendon with a rest length and limits");
        }
        ::physx::PxArticulationAttachment* a_;
    };

    // A spatial (geometric) tendon. Create before Articulation::finalize().
    class SpatialTendon {
    public:
        explicit SpatialTendon(Articulation& art) {
            if (art.finalized())
                throw std::runtime_error(
                        "SpatialTendon: the articulation is already in a scene - PhysX forbids creating a "
                        "tendon on a scene-resident articulation (PxArticulationTendon.h:416). Build every "
                        "tendon before Articulation.finalize().");
            t_ = art.rawArt()->createSpatialTendon();
            if (!t_) throw std::runtime_error("SpatialTendon: createSpatialTendon failed");
        }

        // Add an attachment on `link` at `localOffset` in the link's ACTOR frame.
        // `parent` = nullptr makes it the root. `coefficient` scales this segment's
        // contribution to the accumulated length.
        SpatialAttachment addAttachment(const SpatialAttachment* parent, const ArticulationLink& link,
                                        const Vector3& localOffset, float coefficient = 1.f) {
            auto* a = t_->createAttachment(parent ? parent->raw() : nullptr, coefficient,
                                           toPxVec3(localOffset), link.raw());
            if (!a) throw std::runtime_error("SpatialTendon.add_attachment: createAttachment failed");
            return SpatialAttachment(a);
        }

        // Bilateral spring on the length. Leave at 0 for a cable — see the header
        // note; a nonzero stiffness makes the tendon push when it is short.
        void setStiffness(float k) { t_->setStiffness(k); }
        [[nodiscard]] float stiffness() const { return t_->getStiffness(); }

        // Documented as acting on BOTH the spring and the limits (:307), so it is
        // not known to be one-sided. Leave at 0 until measured.
        void setDamping(float d) { t_->setDamping(d); }
        [[nodiscard]] float damping() const { return t_->getDamping(); }

        // The axial stiffness of the cable in the pull-only construction.
        void setLimitStiffness(float k) { t_->setLimitStiffness(k); }
        [[nodiscard]] float limitStiffness() const { return t_->getLimitStiffness(); }

        // The actuator. Added to the accumulated length, so INCREASING it makes the
        // tendon act shorter, i.e. pulls. The only tendon setter that carries an
        // autowake flag and the only one mirrored into the direct-GPU actuation
        // struct (PxGpuSpatialTendonData) — both signs that it is the intended
        // per-frame motor.
        void setOffset(float o, bool autowake = true) { t_->setOffset(o, autowake); }
        [[nodiscard]] float offset() const { return t_->getOffset(); }

        [[nodiscard]] std::size_t numAttachments() const { return t_->getNbAttachments(); }
        [[nodiscard]] ::physx::PxArticulationSpatialTendon* raw() const { return t_; }

    private:
        ::physx::PxArticulationSpatialTendon* t_{};
    };

    // A fixed (joint-space) tendon. Create before Articulation::finalize().
    class FixedTendon {
    public:
        explicit FixedTendon(Articulation& art) {
            if (art.finalized())
                throw std::runtime_error(
                        "FixedTendon: the articulation is already in a scene - PhysX forbids creating a "
                        "tendon on a scene-resident articulation (PxArticulationTendon.h:481). Build every "
                        "tendon before Articulation.finalize().");
            t_ = art.rawArt()->createFixedTendon();
            if (!t_) throw std::runtime_error("FixedTendon: createFixedTendon failed");
        }

        // Bind one joint DOF into the tendon. `link` names its INBOUND joint (the
        // same convention addLink uses), so the root link cannot be a tendon joint.
        //
        // `coefficient` is the c_i in L = sum(c_i q_i) — dimensionally a moment arm
        // for a revolute DOF. `recipCoefficient` scales the response applied back to
        // this DOF; see the header note on why c (not 1/c) is the energetically
        // consistent choice, and why it is measured rather than assumed. Defaults to
        // coefficient.
        class TendonJoint {
        public:
            explicit TendonJoint(::physx::PxArticulationTendonJoint* j) : j_(j) {}
            void setCoefficient(::physx::PxArticulationAxis::Enum axis, float c, float recip) {
                j_->setCoefficient(axis, c, recip);
            }
            [[nodiscard]] ::physx::PxArticulationTendonJoint* raw() const { return j_; }

        private:
            ::physx::PxArticulationTendonJoint* j_;
        };

        TendonJoint addJoint(const TendonJoint* parent, const ArticulationLink& link,
                             float coefficient, float recipCoefficient) {
            if (link.isRoot())
                throw std::runtime_error(
                        "FixedTendon.add_joint: the root link has no inbound joint to bind");
            auto* j = t_->createTendonJoint(parent ? parent->raw() : nullptr, link.axis(),
                                            coefficient, recipCoefficient, link.raw());
            if (!j)
                throw std::runtime_error(
                        "FixedTendon.add_joint: createTendonJoint failed - the joints of a fixed tendon must "
                        "be directly connected in the articulation, and the axis must be neither LOCKED nor "
                        "part of a fixed joint (PxArticulationTendon.h:466-471, :490-492)");
            return TendonJoint(j);
        }

        void setStiffness(float k) { t_->setStiffness(k); }
        [[nodiscard]] float stiffness() const { return t_->getStiffness(); }
        void setDamping(float d) { t_->setDamping(d); }
        [[nodiscard]] float damping() const { return t_->getDamping(); }
        void setLimitStiffness(float k) { t_->setLimitStiffness(k); }
        [[nodiscard]] float limitStiffness() const { return t_->getLimitStiffness(); }
        void setOffset(float o, bool autowake = true) { t_->setOffset(o, autowake); }
        [[nodiscard]] float offset() const { return t_->getOffset(); }
        void setRestLength(float l) { t_->setRestLength(l); }
        [[nodiscard]] float restLength() const { return t_->getRestLength(); }

        // The SDK's default limit parameters are (PX_MAX_F32, -PX_MAX_F32), which
        // PxArticulationTendon.h:90 itself calls "an invalid configuration that can only
        // work if stiffness is zero". A spring-driven fixed tendon therefore has to be
        // given real limits before it will do anything; leaving them at the default is a
        // silent no-op, not an error. openLimits() is the "never clamp" setting for a
        // tendon that wants only its spring.
        void setLimits(float low, float high) {
            ::physx::PxArticulationTendonLimit lim;
            lim.lowLimit = low;
            lim.highLimit = high;
            t_->setLimitParameters(lim);
        }
        void openLimits() { setLimits(-1e30f, 1e30f); }
        [[nodiscard]] std::pair<float, float> limits() const {
            const auto l = t_->getLimitParameters();
            return {l.lowLimit, l.highLimit};
        }
        [[nodiscard]] std::size_t numJoints() const { return t_->getNbTendonJoints(); }
        [[nodiscard]] ::physx::PxArticulationFixedTendon* raw() const { return t_; }

    private:
        ::physx::PxArticulationFixedTendon* t_{};
    };

}// namespace threepp

#endif
