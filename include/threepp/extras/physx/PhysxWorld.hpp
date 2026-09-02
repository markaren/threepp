
#ifndef THREEPP_PHYSX_WORLD_HPP
#define THREEPP_PHYSX_WORLD_HPP

#include <PxPhysicsAPI.h>
#include <cudamanager/PxCudaContext.h>
#include <cudamanager/PxCudaContextManager.h>

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/Sensor.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    struct SoftBodyTetBind {
        ::physx::PxU32 i0, i1, i2, i3;
        float w0, w1, w2, w3;
    };

    class SoftBody;// defined in PhysxSoftBody.hpp, included at the bottom of this file

    inline ::physx::PxVec3 toPxVec3(const Vector3& v) {
        return {v.x, v.y, v.z};
    }

    /**
     * One reported touch involving a watched actor, delivered from inside
     * fetchResults() — i.e. while the substep that produced it is still current.
     *
     * `self` is the watched actor and `other` is whatever it touched (null only
     * if that actor was removed mid-simulation, in which case the event is
     * dropped before it gets here). Contact points are PhysX's own manifold
     * points, in world space, valid ONLY for the duration of the callback —
     * copy anything you need to keep.
     *
     * Normals and impulses follow PhysX's pair convention, which is expressed
     * relative to the pair's first actor, not to `self`. `selfIsFirst` says
     * which one `self` was, so a consumer that wants everything oriented
     * relative to its own body can flip accordingly.
     */
    struct ContactEvent {
        ::physx::PxRigidActor* self = nullptr;
        ::physx::PxRigidActor* other = nullptr;
        const ::physx::PxContactPairPoint* points = nullptr;
        ::physx::PxU32 pointCount = 0;
        bool selfIsFirst = true;
        bool touchFound = false;// this pair started touching in this substep
        bool touchLost = false; // this pair stopped touching in this substep
    };

    /**
     * One reported overlap involving a watched actor and a TRIGGER shape,
     * delivered from inside fetchResults() exactly as a ContactEvent is.
     *
     * A trigger volume generates no contacts at all — that is the whole point of
     * it — so there is no manifold here, no normal and no impulse: PhysX reports
     * only that the two started or stopped overlapping.
     *
     * `self` is the watched actor and `other` is the one on the far side.
     * `selfIsTrigger` says which of the two the watcher was, since BOTH sides of
     * a trigger pair are dispatched: the volume wants to know who walked in, and
     * whoever walked in wants to know it did.
     *
     * PhysX reports TOUCH_FOUND and TOUCH_LOST only — there is no PERSISTS for a
     * trigger (see PxPairFlag::eNOTIFY_TOUCH_PERSISTS, "Triggers do not support
     * this event"), so an overlap that has already been reported goes quiet until
     * it ends. That makes the found/lost pair the ONLY state there is, and losing
     * one of them loses the state.
     */
    struct TriggerEvent {
        ::physx::PxRigidActor* self = nullptr;
        ::physx::PxRigidActor* other = nullptr;
        bool selfIsTrigger = true;
        bool touchFound = false;// this pair started overlapping in this substep
        bool touchLost = false; // this pair stopped overlapping in this substep
    };

    /**
     * A joint (constraint) broke this substep: the solver needed more than its
     * break force/torque, the constraint stopped acting for good, and this is
     * the one notification PhysX gives about it. `joint` is the PxJoint whose
     * setBreakForce armed the break — compare it against your own records
     * (e.g. Joint::raw()) to name it; do not dereference beyond identity if
     * you cannot prove it is yours.
     *
     * `force` / `torque` (world axes) are the wrench the solver applied on the
     * BREAKING step — the true failure load, necessarily past the threshold
     * that armed the break. Captured here because this callback is the one
     * point where that value is defined: it fires inside fetchResults() of the
     * breaking substep, while PxConstraint::getForce still holds that step's
     * solver result. Afterwards the broken constraint leaves the solver's
     * active set and its force buffer is never written again (a stale read,
     * not a promise — do not rely on it).
     */
    struct ConstraintBreakEvent {
        ::physx::PxJoint* joint = nullptr;
        Vector3 force;
        Vector3 torque;
    };

    /**
     * Routes PhysX contact notifications to per-actor callbacks.
     *
     * Owned by PhysxWorld and installed as the scene's simulation event
     * callback. Contact reporting is OPT-IN per actor (see
     * PhysxWorld::watchContacts): PhysX only generates these notifications for
     * pairs whose filter data asks for them, so a world with no contact
     * watchers pays nothing beyond the branch in the filter shader.
     *
     * TRIGGER notifications (see PhysxWorld::watchTriggers) are the other half,
     * and they are NOT opt-in: the stock filter shader gives every pair
     * involving a trigger shape PxPairFlag::eTRIGGER_DEFAULT unconditionally, so
     * a scene with a trigger volume in it reports that volume's overlaps whether
     * anything is listening or not. What is opt-in is the delivery below.
     */
    class ContactDispatcher: public ::physx::PxSimulationEventCallback {

    public:
        using Handle = std::size_t;

        // Largest manifold PhysX will be asked to unpack per pair. A box on a
        // plane produces 4; 16 covers convex-on-mesh comfortably. Extra points
        // beyond this are counted but not individually reported.
        static constexpr ::physx::PxU32 maxPointsPerPair = 16;

        Handle add(::physx::PxRigidActor* watch, std::function<void(const ContactEvent&)> fn) {
            const Handle h = next_++;
            entries_.push_back({h, watch, std::move(fn)});
            return h;
        }

        void remove(Handle handle) {
            entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                          [handle](const Entry& e) { return e.handle == handle; }),
                           entries_.end());
        }

        Handle addTrigger(::physx::PxRigidActor* watch, std::function<void(const TriggerEvent&)> fn) {
            const Handle h = next_++;
            triggers_.push_back({h, watch, std::move(fn)});
            return h;
        }

        void removeTrigger(Handle handle) {
            triggers_.erase(std::remove_if(triggers_.begin(), triggers_.end(),
                                           [handle](const TriggerEntry& e) { return e.handle == handle; }),
                            triggers_.end());
        }

        // Constraint breaks have no per-actor key: a subscriber hears about
        // EVERY break and matches the PxJoint* against its own records. There
        // is rarely more than one subscriber (a session mapping joints back to
        // scene nodes), so filtering here would buy nothing.
        Handle addBreak(std::function<void(const ConstraintBreakEvent&)> fn) {
            const Handle h = next_++;
            breaks_.push_back({h, std::move(fn)});
            return h;
        }

        void removeBreak(Handle handle) {
            breaks_.erase(std::remove_if(breaks_.begin(), breaks_.end(),
                                         [handle](const BreakEntry& e) { return e.handle == handle; }),
                          breaks_.end());
        }

        // Drop every watcher of `actor` — called when the actor is released, so
        // a stale watch entry cannot match a recycled pointer later. Both kinds:
        // one released actor invalidates every entry naming it.
        void forget(const ::physx::PxRigidActor* actor) {
            entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                          [actor](const Entry& e) { return e.watch == actor; }),
                           entries_.end());
            triggers_.erase(std::remove_if(triggers_.begin(), triggers_.end(),
                                           [actor](const TriggerEntry& e) { return e.watch == actor; }),
                            triggers_.end());
        }

        [[nodiscard]] bool empty() const {
            return entries_.empty() && triggers_.empty() && breaks_.empty();
        }

        void onContact(const ::physx::PxContactPairHeader& header,
                       const ::physx::PxContactPair* pairs, ::physx::PxU32 nbPairs) override {
            using namespace ::physx;
            if (entries_.empty()) return;
            // Either actor deleted during simulate(): header.actors[] dangles.
            if (header.flags & (PxContactPairHeaderFlag::eREMOVED_ACTOR_0 |
                                PxContactPairHeaderFlag::eREMOVED_ACTOR_1)) {
                return;
            }

            // Contact pairs always involve rigid actors, so the downcast from the
            // header's PxActor* is safe.
            auto* a0 = static_cast<PxRigidActor*>(header.actors[0]);
            auto* a1 = static_cast<PxRigidActor*>(header.actors[1]);

            for (PxU32 i = 0; i < nbPairs; ++i) {
                const PxContactPair& cp = pairs[i];
                if (cp.flags & (PxContactPairFlag::eREMOVED_SHAPE_0 |
                                PxContactPairFlag::eREMOVED_SHAPE_1)) {
                    continue;
                }

                const bool found = cp.events.isSet(PxPairFlag::eNOTIFY_TOUCH_FOUND);
                const bool lost = cp.events.isSet(PxPairFlag::eNOTIFY_TOUCH_LOST);
                const bool persists = cp.events.isSet(PxPairFlag::eNOTIFY_TOUCH_PERSISTS);
                if (!found && !lost && !persists) continue;

                // A lost touch has no manifold left to extract.
                const PxU32 n = lost ? 0u : cp.extractContacts(pointBuf_, maxPointsPerPair);

                for (const auto& e: entries_) {
                    const bool isFirst = (e.watch == a0);
                    if (!isFirst && e.watch != a1) continue;
                    ContactEvent ev;
                    ev.self = e.watch;
                    ev.other = isFirst ? a1 : a0;
                    ev.points = pointBuf_;
                    ev.pointCount = n;
                    ev.selfIsFirst = isFirst;
                    ev.touchFound = found;
                    ev.touchLost = lost;
                    e.fn(ev);
                }
            }
        }

        // Trigger overlaps. Unlike onContact this is not gated on a filter bit —
        // the stock shader flags every trigger pair for reporting — so the
        // early-out on an empty watcher list is what a world with triggers but no
        // listeners pays.
        void onTrigger(::physx::PxTriggerPair* pairs, ::physx::PxU32 count) override {
            using namespace ::physx;
            if (triggers_.empty()) return;

            for (PxU32 i = 0; i < count; ++i) {
                const PxTriggerPair& tp = pairs[i];
                // Either shape deleted during simulate(): the actor pointers are
                // only good for identity comparison, and the event is about a
                // shape that no longer exists.
                if (tp.flags & (PxTriggerPairFlag::eREMOVED_SHAPE_TRIGGER |
                                PxTriggerPairFlag::eREMOVED_SHAPE_OTHER)) {
                    continue;
                }

                const bool found = tp.status == PxPairFlag::eNOTIFY_TOUCH_FOUND;
                const bool lost = tp.status == PxPairFlag::eNOTIFY_TOUCH_LOST;
                if (!found && !lost) continue;// PERSISTS does not exist for triggers

                // Trigger pairs are always rigid actors, so the downcasts are safe.
                auto* trigger = static_cast<PxRigidActor*>(tp.triggerActor);
                auto* other = static_cast<PxRigidActor*>(tp.otherActor);

                // BOTH sides are dispatched. The volume wants to know who walked
                // in; whoever walked in is not itself a trigger and would
                // otherwise never hear about it.
                for (const auto& e: triggers_) {
                    const bool isTrigger = (e.watch == trigger);
                    if (!isTrigger && e.watch != other) continue;
                    TriggerEvent ev;
                    ev.self = e.watch;
                    ev.other = isTrigger ? other : trigger;
                    ev.selfIsTrigger = isTrigger;
                    ev.touchFound = found;
                    ev.touchLost = lost;
                    e.fn(ev);
                }
            }
        }

        // Constraint breaks. Reporting is armed by setBreakForce alone (no
        // filter bit, no scene parameter), so like onTrigger the early-out on
        // an empty subscriber list is the whole cost of not listening.
        void onConstraintBreak(::physx::PxConstraintInfo* constraints, ::physx::PxU32 count) override {
            using namespace ::physx;
            if (breaks_.empty()) return;

            for (PxU32 i = 0; i < count; ++i) {
                // externalReference is typed by `type`: only a joint's is a
                // PxJoint*. Anything else (a custom constraint) is skipped
                // rather than mis-cast.
                if (constraints[i].type != PxConstraintExtIDs::eJOINT) continue;
                ConstraintBreakEvent ev;
                ev.joint = static_cast<PxJoint*>(constraints[i].externalReference);
                // The breaking step's solver wrench — the failure load. Read
                // NOW, while fetchResults keeps it current (see the event doc);
                // after this callback the buffer merely goes stale.
                if (auto* constraint = ev.joint->getConstraint()) {
                    PxVec3 f(PxZero), t(PxZero);
                    constraint->getForce(f, t);
                    ev.force.set(f.x, f.y, f.z);
                    ev.torque.set(t.x, t.y, t.z);
                }
                for (const auto& e : breaks_) e.fn(ev);
            }
        }

        // Unused halves of the interface. PhysX requires all of them.
        void onWake(::physx::PxActor**, ::physx::PxU32) override {}
        void onSleep(::physx::PxActor**, ::physx::PxU32) override {}
        void onAdvance(const ::physx::PxRigidBody* const*, const ::physx::PxTransform*,
                       const ::physx::PxU32) override {}

    private:
        struct Entry {
            Handle handle;
            ::physx::PxRigidActor* watch;
            std::function<void(const ContactEvent&)> fn;
        };
        struct TriggerEntry {
            Handle handle;
            ::physx::PxRigidActor* watch;
            std::function<void(const TriggerEvent&)> fn;
        };
        struct BreakEntry {
            Handle handle;
            std::function<void(const ConstraintBreakEvent&)> fn;
        };
        std::vector<Entry> entries_;
        std::vector<TriggerEntry> triggers_;
        std::vector<BreakEntry> breaks_;
        // Shared by both lists, so a handle names exactly one entry whichever
        // kind it is and unwatch cannot cross the streams.
        Handle next_ = 1;// 0 reserved as "no handle"
        ::physx::PxContactPairPoint pointBuf_[maxPointsPerPair]{};
    };

    // Bit set in a shape's SIMULATION filter data (word3) to request contact
    // notifications for pairs it takes part in. word3 is used because the stock
    // filter shader's group mechanism reads word0/word1, and nothing in threepp
    // writes simulation filter data otherwise — so setting this cannot change
    // which pairs collide, only whether they are reported.
    inline constexpr ::physx::PxU32 kContactReportFilterBit = 1u << 31;

    /**
     * The stock filter shader plus opt-in contact reporting.
     *
     * Delegating to PxDefaultSimulationFilterShader (rather than reimplementing
     * it) keeps collision behaviour bit-identical to before this existed: the
     * only change is the extra pair flags, and only for pairs where one side
     * asked for them.
     *
     * Trigger pairs are LEFT ALONE. The stock shader already assigns them
     * PxPairFlag::eTRIGGER_DEFAULT (found | lost | detect-discrete), which is
     * what makes trigger reporting work here with no opt-in bit at all — and it
     * is also the complete set of flags a trigger pair may carry.
     * Sc::TriggerInteraction::setTriggerFlags masks whatever it is handed down
     * to found|lost, and warns once under PX_CHECKED that eNOTIFY_TOUCH_PERSISTS
     * is not supported for triggers. So OR-ing the contact-report set onto a
     * trigger pair is not a crash, it is a request for something the SDK drops —
     * and a warning in any checked build. A body watched for CONTACTS that walks
     * into a trigger volume is an ordinary thing to author (the same script can
     * define both callbacks), so the pair is common enough to get right rather
     * than tolerate. Which pair is which is not the shape flags but
     * PxFilterObjectIsTrigger on the attributes — the same test the stock shader
     * makes one line earlier.
     */
    inline ::physx::PxFilterFlags contactReportFilterShader(
            ::physx::PxFilterObjectAttributes attributes0, ::physx::PxFilterData filterData0,
            ::physx::PxFilterObjectAttributes attributes1, ::physx::PxFilterData filterData1,
            ::physx::PxPairFlags& pairFlags, const void* constantBlock,
            ::physx::PxU32 constantBlockSize) {
        using namespace ::physx;
        const PxFilterFlags flags = PxDefaultSimulationFilterShader(
                attributes0, filterData0, attributes1, filterData1,
                pairFlags, constantBlock, constantBlockSize);

        const bool trigger = PxFilterObjectIsTrigger(attributes0) ||
                             PxFilterObjectIsTrigger(attributes1);

        if (!trigger && ((filterData0.word3 | filterData1.word3) & kContactReportFilterBit)) {
            pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
                         PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
                         PxPairFlag::eNOTIFY_TOUCH_LOST |
                         PxPairFlag::eNOTIFY_CONTACT_POINTS;
        }
        return flags;
    }

    inline ::physx::PxQuat toPxQuat(const Quaternion& q) {
        return {q.x, q.y, q.z, q.w};
    }

    inline ::physx::PxTransform toPxTransform(const Vector3& pos, const Quaternion& rot = Quaternion()) {
        return ::physx::PxTransform(toPxVec3(pos), toPxQuat(rot));
    }

    inline Vector3 fromPxVec3(const ::physx::PxVec3& v) {
        return {v.x, v.y, v.z};
    }

    inline Quaternion fromPxQuat(const ::physx::PxQuat& q) {
        return Quaternion(q.x, q.y, q.z, q.w);
    }

    // World-space pose + scale of an object, as PhysX wants it: a rigid pose
    // (PhysX actors carry no scale) plus the decomposed world scale for baking
    // into shape dimensions.
    //
    // updateWorldMatrix, not updateMatrixWorld: the latter trusts the parent's
    // cached matrixWorld, which is stale (identity) for a body added before
    // the first render - the actor would spawn at the object's LOCAL
    // coordinates. This helper exists so that comment lives ONCE.
    struct WorldPlacement {
        ::physx::PxTransform pose;
        Vector3 scale{1, 1, 1};
    };

    inline WorldPlacement worldPlacement(Object3D& object) {
        object.updateWorldMatrix(true, false);
        Vector3 position, scale;
        Quaternion rotation;
        object.matrixWorld->decompose(position, rotation, scale);
        return {toPxTransform(position, rotation), scale};
    }

    namespace physx_detail {

        // Box/Sphere/Capsule collider inferred from a mesh geometry. THE one
        // copy — PhysxWorld and the articulation builders all use this.
        // localPose corrects the capsule axis (threepp capsule is Y-aligned;
        // PhysX capsule is X-aligned).
        struct InferredShape {
            ::physx::PxGeometryHolder geom;
            ::physx::PxTransform localPose{::physx::PxIdentity};
            bool valid = true;
        };

        // `scale` is the mesh's decomposed WORLD scale, applied to the analytic
        // dimensions here because a PhysX primitive cannot be scaled after the
        // fact. A sphere has one radius, so a non-uniform scale keeps the
        // largest component; a capsule scales its radius from the lateral pair
        // and its height from Y (the threepp capsule axis).
        inline InferredShape inferShape(const BufferGeometry& geometry,
                                        const Vector3& scale = Vector3(1.f, 1.f, 1.f)) {
            using namespace ::physx;
            InferredShape out;
            if (auto box = dynamic_cast<const BoxGeometry*>(&geometry)) {
                out.geom = PxBoxGeometry(box->width * 0.5f * std::abs(scale.x),
                                         box->height * 0.5f * std::abs(scale.y),
                                         box->depth * 0.5f * std::abs(scale.z));
                return out;
            }
            if (auto sph = dynamic_cast<const SphereGeometry*>(&geometry)) {
                out.geom = PxSphereGeometry(sph->radius * std::max({std::abs(scale.x), std::abs(scale.y), std::abs(scale.z)}));
                return out;
            }
            if (auto cap = dynamic_cast<const CapsuleGeometry*>(&geometry)) {
                out.geom = PxCapsuleGeometry(cap->radius * std::max(std::abs(scale.x), std::abs(scale.z)),
                                             cap->length * 0.5f * std::abs(scale.y));
                // PhysX capsule axis is X; threepp capsule axis is Y. Rotate -PI/2 about Z.
                out.localPose = PxTransform(PxQuat(-PxHalfPi, PxVec3(0, 0, 1)));
                return out;
            }
            out.valid = false;
            return out;
        }

    }// namespace physx_detail

    // Interpolate between two PxTransforms. Position is linear; orientation uses
    // shortest-arc nlerp, which is visually indistinguishable from slerp for the
    // small angles between fixed-timestep substeps and cheaper. Used by
    // PhysxWorld to interpolate bound visuals between physics ticks.
    inline ::physx::PxTransform lerpPxTransform(const ::physx::PxTransform& a,
                                                const ::physx::PxTransform& b,
                                                float t) {
        using namespace ::physx;
        const PxVec3 p = a.p * (1.f - t) + b.p * t;
        const float dot = a.q.x * b.q.x + a.q.y * b.q.y + a.q.z * b.q.z + a.q.w * b.q.w;
        const float s = (dot < 0.f) ? -1.f : 1.f;// pick shortest arc
        PxQuat q(
                a.q.x * (1.f - t) + b.q.x * t * s,
                a.q.y * (1.f - t) + b.q.y * t * s,
                a.q.z * (1.f - t) + b.q.z * t * s,
                a.q.w * (1.f - t) + b.q.w * t * s);
        q.normalize();
        return PxTransform(p, q);
    }

    // Owns PhysX foundation/physics/scene. Single-scene wrapper aimed at scene-graph
    // integration; advanced use (multiple scenes, custom filter shaders, GPU dynamics)
    // can drop down to physics() / scene() / foundation() and ignore the helpers.
    //
    // Vehicle-ready: PxVehicle2 can register its update via onPreSubstep, then
    // raycast / sweep through scene() and write torques on the vehicle's wheels.
    class PhysxWorld {

    public:
        struct Settings {
            Vector3 gravity{0, -9.81f, 0};
            float fixedTimestep = 1.f / 60.f;
            int maxSubSteps = 4;
            // Low-pass the dt handed to step() before it drives the fixed-timestep
            // accumulator. DEFAULT OFF — measured to HURT smoothness on the real
            // vsync path (see below).
            //
            // The interpolation in step() is textbook "fix your timestep" (Fiedler
            // 2004): render lerp(prevSubstep, curSubstep, accumulator/dt). Fed the
            // RAW frame dt, that renders the sim state at a CONSTANT latency behind
            // wall-clock (~one fixed step), which is inherently smooth — per-frame dt
            // jitter is exactly absorbed by the alpha. Low-passing dt breaks that:
            // the accumulator then advances in a FICTIONAL (smoothed) timeline while
            // the frame is displayed at real wall-clock time, so the shown pose no
            // longer matches where the body should be at display time — the mapping
            // wobbles = judder. Measured on a chase-cammed vehicle under FIFO vsync:
            // enabling this smoothing raised per-frame along-track velocity JERK ~11x
            // (mean 0.25 -> 2.8 m/s^2, max 4.6 -> 22) vs raw dt. A CONSTANT dt is a
            // no-op either way, so fixed-step/deterministic callers are byte-unchanged.
            // The accumulator's maxSubSteps guard already bounds hitch catch-up, so
            // raw dt needs no low-pass to stay stable. Left as an opt-in only for
            // callers that (unlike the interpolated bindings) sample raw poses.
            bool smoothTimestep = false;
            unsigned numThreads = 2;
            // Enable GPU dynamics. Required for soft bodies (PxDeformableVolume),
            // particle systems, and GPU broadphase. Switches the scene to the TGS
            // solver. Needs a CUDA-capable GPU and the omniverse-physx GPU library
            // (gpu-library is copied next to the example by AddExample.cmake).
            bool enableGpuDynamics = false;
            // Enable the PhysX direct-GPU API (eENABLE_DIRECT_GPU_API). Implies
            // enableGpuDynamics. Lets articulation state be read/written as CUDA
            // device buffers in bulk (zero-copy to e.g. a torch CUDA tensor) — the
            // basis for GPU-resident vectorized RL (see PhysxGpuBatch). NOTE: when on,
            // the per-actor CPU getters/setters (getGlobalPose, getJointPosition,
            // setDriveTarget, ...) and the binding-sync step() are NOT valid; all
            // runtime state I/O must go through the direct-GPU batch instead.
            bool enableDirectGpu = false;
            // Use the TGS solver + PCM narrowphase + stabilization on a CPU world.
            // The GPU path (enableGpuDynamics) ALWAYS uses these; a pure-CPU world
            // otherwise defaults to PGS, no PCM, no stabilization. Set this so a CPU
            // world's CONTACT MODEL matches a GPU-trained policy — PGS-vs-TGS and the
            // PCM difference change the effective friction / penetration recovery a
            // policy sees, which is a primary sim-to-sim (GPU train -> CPU deploy)
            // transfer gap. No effect (already implied) when enableGpuDynamics is set.
            bool enableTgsPcm = false;
            // PxSceneFlag::eENABLE_ENHANCED_DETERMINISM. What it adds — and all
            // it adds: island-order invariance. WITHOUT it PhysX already
            // guarantees replay determinism (same scene, same insertion order,
            // same substeps, same binary → identical results; the replay_audit
            // harness proves that end-to-end through the sensor suite), but the
            // solver batches constraints ACROSS islands, so an unrelated actor
            // elsewhere in the scene can perturb this robot's impulses in the
            // last ULP. With it, each island solves alone: a prop added 50 m
            // away can no longer change a sensor log. Costs solver parallelism
            // (per NVIDIA, "at the expense of performance"); off by default,
            // opt in for recorded-dataset / replay work. CPU pipeline only —
            // GPU dynamics makes no determinism promise at all.
            bool enhancedDeterminism = false;
            // Existing CUDA context for PhysX to adopt (instead of creating its own).
            // Pass the host framework's context (e.g. PyTorch's device primary context)
            // so PhysX and that framework share ONE context — required for correctness
            // when both run CUDA work on the same device (a separate PhysX context leaves
            // cuBLAS/cuDNN unable to launch: CUBLAS_STATUS_INTERNAL_ERROR). Must be the
            // current context on this thread when the world is constructed.
            CUcontext cudaContext = nullptr;
            // GPU solver memory sizing (only used when enableGpuDynamics). The
            // defaults are tuned for a handful of bodies; thousands of articulations
            // need bigger pools or the GPU pipeline silently drops contacts / errors.
            unsigned gpuMaxRigidContacts = 1u << 20;
            unsigned gpuMaxRigidPatches = 1u << 20;
            unsigned gpuFoundLostPairsCapacity = 1u << 18;
            unsigned gpuTempBufferCapacityMB = 16;
            unsigned gpuHeapCapacityMB = 64;
        };

        PhysxWorld() : PhysxWorld(Settings{}) {}

        explicit PhysxWorld(Settings s)
            : settings_(s) {

            using namespace ::physx;

            // Direct-GPU implies GPU dynamics; normalize before any of the setup below
            // branches on enableGpuDynamics.
            if (settings_.enableDirectGpu) settings_.enableGpuDynamics = true;

            // Everything from here on can throw (most likely the CUDA context on a
            // machine without a CUDA GPU). A throwing constructor never runs the
            // destructor, and PxFoundation is one-instance-per-process — leak it and
            // every later PhysxWorld in the process dies with "Foundation object
            // exists already". So a failed construction tears down exactly what it
            // managed to build, via the same releaseCore() the destructor uses.
            try {
                constructCore();
            } catch (...) {
                releaseCore();
                throw;
            }
        }

    private:
        void constructCore() {

            using namespace ::physx;

            foundation_ = PxCreateFoundation(PX_PHYSICS_VERSION, allocator_, errorCallback_);
            if (!foundation_) {
                if (PxIsFoundationValid()) {
                    throw std::runtime_error(
                            "PxCreateFoundation failed: PhysX allows one foundation per process, "
                            "and another PhysxWorld is still alive. Destroy it before creating a "
                            "new one (in Python: del the old world and gc.collect(), or restart "
                            "the kernel).");
                }
                throw std::runtime_error("PxCreateFoundation failed");
            }

            // trackOutstandingAllocations=true matches the GPU samples; harmless when off.
            physics_ = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation_, PxTolerancesScale(),
                                       settings_.enableGpuDynamics);
            if (!physics_) throw std::runtime_error("PxCreatePhysics failed");

            if (!PxInitExtensions(*physics_, nullptr)) {
                throw std::runtime_error("PxInitExtensions failed");
            }
            extensionsInitialized_ = true;

            if (settings_.enableGpuDynamics) {
                PxCudaContextManagerDesc cudaDesc;
                // Adopt the caller's context (e.g. PyTorch's) when provided, so both
                // share a single CUDA context on the device.
                CUcontext sharedCtx = settings_.cudaContext;
                if (sharedCtx) cudaDesc.ctx = &sharedCtx;
                cuda_ = PxCreateCudaContextManager(*foundation_, cudaDesc, PxGetProfilerCallback());
                if (cuda_ && !cuda_->contextIsValid()) {
                    cuda_->release();
                    cuda_ = nullptr;
                }
                if (!cuda_) throw std::runtime_error("PhysxWorld: failed to create CUDA context (no CUDA GPU?)");
                {
                    PxScopedCudaLock _lock(*cuda_);
                    cuda_->getCudaContext()->streamCreate(&cudaCopyStream_, 0);
                }
            }

            dispatcher_ = PxDefaultCpuDispatcherCreate(settings_.numThreads);
            if (!dispatcher_) throw std::runtime_error("PxDefaultCpuDispatcherCreate failed");

            PxSceneDesc desc(physics_->getTolerancesScale());
            desc.gravity = toPxVec3(settings_.gravity);
            desc.cpuDispatcher = dispatcher_;
            // The stock shader plus opt-in contact reporting; behaviour is
            // unchanged for any pair that has not asked to be reported.
            desc.filterShader = contactReportFilterShader;
            desc.simulationEventCallback = &contacts_;
            // TGS solver + PCM narrowphase + stabilization. The GPU pipeline requires
            // these; a CPU world opts in via enableTgsPcm so its contact model matches
            // a GPU-trained policy (the GPU-only flags below stay gated separately).
            if (settings_.enableGpuDynamics || settings_.enableTgsPcm) {
                desc.solverType = PxSolverType::eTGS;
                desc.flags |= PxSceneFlag::eENABLE_PCM;
                desc.flags |= PxSceneFlag::eENABLE_STABILIZATION;
            }
            if (settings_.enhancedDeterminism) {
                desc.flags |= PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
            }
            if (settings_.enableGpuDynamics) {
                desc.cudaContextManager = cuda_;
                desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
                desc.broadPhaseType = PxBroadPhaseType::eGPU;
                desc.gpuMaxNumPartitions = 8;
                // Size the GPU solver pools for many articulations (RL swarms).
                desc.gpuDynamicsConfig.maxRigidContactCount = settings_.gpuMaxRigidContacts;
                desc.gpuDynamicsConfig.maxRigidPatchCount = settings_.gpuMaxRigidPatches;
                desc.gpuDynamicsConfig.foundLostPairsCapacity = settings_.gpuFoundLostPairsCapacity;
                desc.gpuDynamicsConfig.tempBufferCapacity =
                        static_cast<PxU32>(settings_.gpuTempBufferCapacityMB) * 1024u * 1024u;
                desc.gpuDynamicsConfig.heapCapacity =
                        static_cast<PxU32>(settings_.gpuHeapCapacityMB) * 1024u * 1024u;
                // Direct-GPU API: read/write actor + articulation state as CUDA buffers
                // with no CPU readback. Requires GPU dynamics + GPU broadphase (set above).
                if (settings_.enableDirectGpu) {
                    desc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
                }
            }
            scene_ = physics_->createScene(desc);
            if (!scene_) throw std::runtime_error("createScene failed");

            defaultMat_ = physics_->createMaterial(0.5f, 0.5f, 0.2f);
        }

        // The tail of ~PhysxWorld, shared with the constructor's failure path.
        // Safe to call on a partially-built world: every branch checks what it
        // releases, and PxCloseExtensions only runs if PxInitExtensions ran.
        void releaseCore() {
            using namespace ::physx;
            if (scene_) {
                scene_->release();
                scene_ = nullptr;
            }
            if (dispatcher_) {
                dispatcher_->release();
                dispatcher_ = nullptr;
            }
            if (cuda_ && cudaCopyStream_) {
                PxScopedCudaLock _lock(*cuda_);
                cuda_->getCudaContext()->streamDestroy(cudaCopyStream_);
                cudaCopyStream_ = nullptr;
            }
            if (physics_) {
                if (extensionsInitialized_) PxCloseExtensions();
                physics_->release();
                physics_ = nullptr;
            }
            if (cuda_) {
                cuda_->release();
                cuda_ = nullptr;
            }
            if (foundation_) {
                foundation_->release();
                foundation_ = nullptr;
            }
        }

    public:
        ~PhysxWorld() {
            using namespace ::physx;
            // A sensor still registered here holds a back-pointer to this world
            // (see onRegister). Sessions normally unregister before the world
            // goes, but the seam does not require that order — so notify the
            // stragglers first, while the scene and articulations their
            // onUnregister hooks touch are still alive. Without this, the
            // sensor's own destructor calls into freed memory (detach →
            // unwatchContacts).
            for (auto* s: sensors_) s->onUnregister();
            sensors_.clear();
            // Soft bodies must be released BEFORE the scene/physics/cuda context;
            // their destructor releases the PxDeformableVolume actor and frees pinned
            // host memory through the CUDA context.
            softBodies_.clear();
            for (auto& [_, entry] : cookCache_) {
                if (entry.mesh) entry.mesh->release();
            }
            cookCache_.clear();
            releaseCore();
        }

        PhysxWorld(const PhysxWorld&) = delete;
        PhysxWorld& operator=(const PhysxWorld&) = delete;

        // Variable-rate caller, fixed-rate physics. Pre/post substep hooks fire
        // around each fetchResults boundary. Visual bindings (Object3D / InstancedMesh)
        // are interpolated between the last two substep states using the leftover
        // accumulator fraction as alpha — this smooths out the 0-or-2-substeps-per-frame
        // pattern that variable real-frame dt produces under vsync jitter (the
        // classic "fix your timestep" problem; see Glenn Fiedler 2004).
        // Soft bodies are NOT interpolated — their GPU positions only get pulled
        // when at least one substep ran this call.
        void step(float dt) {
            // Delta-time smoothing (see Settings::smoothTimestep). Lazy-init to the
            // first dt so a constant timestep is an exact no-op; skip genuine
            // hitches so they neither pollute the average nor stall the catch-up.
            if (settings_.smoothTimestep && dt > 0.f && dt < 0.1f) {
                dtEma_ = (dtEma_ <= 0.f) ? dt : dtEma_ * 0.85f + dt * 0.15f;
                dt = dtEma_;
            }
            accumulator_ += dt;
            int steps = 0;
            while (accumulator_ >= settings_.fixedTimestep && steps < settings_.maxSubSteps) {
                snapshotPrevPoses();
                substep(settings_.fixedTimestep);
                accumulator_ -= settings_.fixedTimestep;
                ++steps;
            }
            if (steps >= settings_.maxSubSteps) {
                accumulator_ = 0;// avoid spiral of death on hitches
            }
            if (steps > 0) {
                syncSoftBodies();
            }
            // alpha = how far past the last completed substep we are. Always sync
            // rigid bindings, even on frames with 0 substeps — that's exactly when
            // alpha advancing produces the smoothing benefit.
            const float alpha = std::clamp(accumulator_ / settings_.fixedTimestep, 0.f, 1.f);
            syncRigidBindings(alpha);
        }

        // Substep hooks. Both return an opaque handle usable with
        // removeSubstepCallback(): a callback almost always captures `this` from
        // some longer-lived object (PhysxVehicle registers its update here), and
        // without a way to unregister, destroying that object left the world
        // calling into freed memory on the next step. Anything that registers
        // must unregister in its destructor.
        using SubstepHandle = std::size_t;

        SubstepHandle onPreSubstep(std::function<void(float)> cb) {
            const auto h = nextSubstepHandle_++;
            preSubstep_.push_back({h, std::move(cb)});
            return h;
        }

        SubstepHandle onPostSubstep(std::function<void(float)> cb) {
            const auto h = nextSubstepHandle_++;
            postSubstep_.push_back({h, std::move(cb)});
            return h;
        }

        // Unregister a pre/post substep callback. Safe to call with a stale or
        // already-removed handle (no-op), and safe to call from inside a
        // callback — the step loop iterates by index over a snapshot.
        void removeSubstepCallback(SubstepHandle handle) {
            const auto drop = [handle](std::vector<SubstepEntry>& v) {
                v.erase(std::remove_if(v.begin(), v.end(),
                                       [handle](const SubstepEntry& e) { return e.handle == handle; }),
                        v.end());
            };
            drop(preSubstep_);
            drop(postSubstep_);
        }

        // --- Proprioceptive sensors -------------------------------------------
        // A registered Sensor is sampled from the step loop once per fixed substep
        // (rate-gated by the sensor), the instant body states are fresh — see
        // Sensor.hpp. registerSensor calls the sensor's onRegister hook (where an
        // IMU resolves its rigid body and may throw on a bad attachment), so
        // register AFTER the body the sensor rides has been add()-ed.
        //
        // The sensor is a non-owning pointer: it must outlive its registration
        // (unregister it, or keep it alive, before it is destroyed). NOTE: sensors
        // are only driven by the interpolated step() path; the direct-GPU
        // simulateRaw() path does not sample them.
        void registerSensor(Sensor* sensor) {
            if (!sensor) return;
            if (std::find(sensors_.begin(), sensors_.end(), sensor) != sensors_.end()) return;
            sensor->onRegister(*this);// may throw (invalid attachment) — do this before storing
            sensors_.push_back(sensor);
        }

        void unregisterSensor(Sensor* sensor) {
            auto it = std::find(sensors_.begin(), sensors_.end(), sensor);
            if (it == sensors_.end()) return;
            sensors_.erase(it);
            sensor->onUnregister();
        }

        // --- Contact reporting -------------------------------------------------
        // Contacts are OFF by default: PhysX only notifies for pairs whose filter
        // data asks for it, so a world with no watchers pays nothing. Watching an
        // actor sets that bit on every one of its shapes and re-runs filtering, so
        // pairs already in flight pick the change up.
        //
        // The callback fires from inside fetchResults(), i.e. DURING substep() and
        // before sensors are ticked — so a sensor sampling in the same substep
        // sees contacts belonging to that substep. Do not add/remove actors from
        // inside it.
        //
        // Note on sleeping: PhysX stops issuing TOUCH_PERSISTS once a pair goes to
        // sleep, without a TOUCH_LOST. A resting contact therefore reports points
        // for a while and then goes quiet while still physically touching — track
        // the found/lost transitions if you need a steady "is touching" state
        // (ContactSensor does exactly that).
        using ContactHandle = ContactDispatcher::Handle;

        ContactHandle watchContacts(::physx::PxRigidActor* actor,
                                    std::function<void(const ContactEvent&)> cb) {
            using namespace ::physx;
            if (!actor || !cb) return 0;
            setContactReporting(actor, true);
            return contacts_.add(actor, std::move(cb));
        }

        // Stop delivering to this watcher. The reporting bit is left set on the
        // actor's shapes — several watchers may share an actor, and clearing it
        // per-watcher would silence the others. Use setContactReporting(actor,
        // false) to turn the actor's reporting off once nothing watches it.
        void unwatchContacts(ContactHandle handle) { contacts_.remove(handle); }

        // Set/clear the contact-report request on every shape of `actor`.
        void setContactReporting(::physx::PxRigidActor* actor, bool enabled) {
            using namespace ::physx;
            if (!actor) return;
            const PxU32 n = actor->getNbShapes();
            if (n == 0) return;
            std::vector<PxShape*> shapes(n);
            actor->getShapes(shapes.data(), n);
            for (auto* s: shapes) {
                if (!s) continue;
                PxFilterData fd = s->getSimulationFilterData();
                const PxU32 before = fd.word3;
                fd.word3 = enabled ? (fd.word3 | kContactReportFilterBit)
                                   : (fd.word3 & ~kContactReportFilterBit);
                if (fd.word3 != before) s->setSimulationFilterData(fd);
            }
            // Existing broad-phase pairs were filtered under the old data; without
            // this they would keep their old pair flags until they separate.
            scene_->resetFiltering(*actor);
        }

        // --- Trigger reporting -------------------------------------------------
        // A trigger volume is a shape carrying PxShapeFlag::eTRIGGER_SHAPE
        // INSTEAD of eSIMULATION_SHAPE (the two are mutually exclusive on one
        // shape): it generates no contacts, resolves nothing, and things pass
        // straight through it — what it does is report who is inside.
        //
        // Unlike contacts this needs no opt-in bit. The filter shader hands every
        // pair involving a trigger shape eTRIGGER_DEFAULT, so the notifications
        // exist as soon as the scene contains a trigger; watching is only about
        // delivery. Watch EITHER side: the volume, to hear who entered it, or the
        // entering body, to hear that it entered something. Both are dispatched
        // from one PxTriggerPair.
        //
        // The callback fires from inside fetchResults(), same rule as contacts:
        // copy what you need, and do not add or remove actors from in there.
        //
        // Two behaviours worth knowing, both PhysX's and neither worked around:
        //   * A trigger pair reports TOUCH_FOUND and TOUCH_LOST and nothing
        //     else — there is no PERSISTS event for a trigger. An overlap that is
        //     simply continuing produces no callbacks at all.
        //   * A dynamic actor resting inside a trigger goes to sleep, and a
        //     sleeping pair produces no further events. That is harmless here
        //     BECAUSE there is no persist event to lose: found has already been
        //     delivered, and the lost still arrives when the actor is woken and
        //     leaves. What would break is a design that re-derived "inside" from
        //     a per-step event; track the found/lost transitions instead.
        using TriggerHandle = ContactDispatcher::Handle;

        TriggerHandle watchTriggers(::physx::PxRigidActor* actor,
                                    std::function<void(const TriggerEvent&)> cb) {
            if (!actor || !cb) return 0;
            return contacts_.addTrigger(actor, std::move(cb));
        }

        void unwatchTriggers(TriggerHandle handle) { contacts_.removeTrigger(handle); }

        // Hear about every joint that BREAKS — exceeds the break force/torque
        // its creator set — in this world. No per-actor key: the subscriber
        // matches ConstraintBreakEvent::joint against its own joints (there is
        // no other way to name one). Reporting is armed by setBreakForce alone,
        // so an unbreakable joint never costs a notification. Delivered from
        // inside fetchResults(), i.e. during step(): queue, do not re-enter the
        // world from the callback.
        using BreakHandle = ContactDispatcher::Handle;

        BreakHandle watchConstraintBreaks(std::function<void(const ConstraintBreakEvent&)> cb) {
            if (!cb) return 0;
            return contacts_.addBreak(std::move(cb));
        }

        void unwatchConstraintBreaks(BreakHandle handle) { contacts_.removeBreak(handle); }

        // Turn every shape of `actor` into a trigger shape, or back into a
        // simulation shape. Returns false if any shape refused.
        //
        // PhysX rejects eTRIGGER_SHAPE on triangle-mesh and heightfield geometry:
        // "PxShape::setFlag(s): triangle mesh and heightfield triggers are not
        // supported!", verbatim from NpShape.cpp in 5.5. A trigger asks whether a
        // point is INSIDE it, and those two are surfaces. (Planes are refused by
        // the older documentation but accepted by 5.5 — measured, not assumed —
        // so they are not excluded here.)
        //
        // The check is ahead of the setFlag rather than after it for a reason
        // stronger than tidiness. The SDK's refusal is an error-stream message,
        // not a return value, and the two flags are mutually exclusive: the
        // caller must lower eSIMULATION_SHAPE before raising eTRIGGER_SHAPE, so
        // a refused raise leaves the shape with NEITHER — neither colliding nor
        // triggering, silently. Asking first is what keeps that state
        // unreachable. A caller that wants a trigger out of a triangle mesh must
        // cook something else (a convex hull) and pass THAT actor.
        static bool setTriggerShapes(::physx::PxRigidActor& actor, bool enabled) {
            using namespace ::physx;
            const PxU32 n = actor.getNbShapes();
            if (n == 0) return false;
            std::vector<PxShape*> shapes(n);
            actor.getShapes(shapes.data(), n);

            bool all = true;
            for (auto* s: shapes) {
                if (!s) continue;
                if (enabled && !canBeTrigger(s->getGeometry().getType())) {
                    all = false;
                    continue;
                }
                if (enabled) {
                    s->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
                    s->setFlag(PxShapeFlag::eTRIGGER_SHAPE, true);
                } else {
                    s->setFlag(PxShapeFlag::eTRIGGER_SHAPE, false);
                    s->setFlag(PxShapeFlag::eSIMULATION_SHAPE, true);
                }
            }
            return all;
        }

        // The geometries PhysX will accept as a trigger. See setTriggerShapes.
        static bool canBeTrigger(::physx::PxGeometryType::Enum type) {
            using namespace ::physx;
            return type != PxGeometryType::eTRIANGLEMESH &&
                   type != PxGeometryType::eHEIGHTFIELD;
        }

        [[nodiscard]] ContactDispatcher& contactDispatcher() { return contacts_; }

        // Accumulated simulation time (s) — sum of every fixed substep advanced so
        // far. This is the clock stamped onto sensor samples.
        [[nodiscard]] double simTime() const { return simTime_; }

        // How many fixed substeps have been advanced. simTime() divided by the
        // timestep, except that this stays exact if the timestep is ever changed
        // mid-run and costs no float division to ask for.
        [[nodiscard]] std::uint64_t substepCount() const { return substeps_; }

        // Resolve the PxRigidActor that governs `obj`: walk up the scene graph from
        // obj to the root and return the actor bound (via bind()/add()) to the
        // nearest ancestor (or obj itself). nullptr if none is managed here. Used
        // by sensors to map their attachment node to a rigid body.
        [[nodiscard]] ::physx::PxRigidActor* findActor(const Object3D* obj) const {
            for (const Object3D* o = obj; o != nullptr; o = o->parent) {
                for (const auto& b : objBindings_) {
                    if (b.obj == o) return b.actor;
                }
                for (const auto& a : associations_) {
                    if (a.obj == o) return a.actor;
                }
            }
            return nullptr;
        }

        // After each step, copy actor's world pose into Object3D.position/quaternion.
        void bind(Object3D& obj, ::physx::PxRigidActor& actor) {
            objBindings_.push_back({&obj, &actor});
        }

        // Make findActor resolve `obj` to `actor` WITHOUT mirroring the actor's
        // pose into it. For nodes that already follow the simulation by other
        // means — an articulated robot's visual links are driven by joint-space
        // mirroring, where a world-pose write-back would fight the kinematic
        // chain — but whose sensors (IMU, contact) still need to find the body
        // that governs them.
        void associate(Object3D& obj, ::physx::PxRigidActor& actor) {
            associations_.push_back({&obj, &actor});
        }

        void unassociate(Object3D& obj) {
            associations_.erase(
                    std::remove_if(associations_.begin(), associations_.end(),
                                   [&](const Association& a) { return a.obj == &obj; }),
                    associations_.end());
        }

        // Each instance i mirrors the pose of actors[i]. Per-instance scale is preserved
        // by decomposing the existing matrix; rotation/translation come from the actor.
        void bind(InstancedMesh& mesh, std::vector<::physx::PxRigidActor*> actors) {
            instBindings_.push_back({&mesh, std::move(actors)});
        }

        void unbind(Object3D& obj) {
            objBindings_.erase(
                    std::remove_if(objBindings_.begin(), objBindings_.end(),
                                   [&](const ObjBinding& b) { return b.obj == &obj; }),
                    objBindings_.end());
        }

        void unbind(InstancedMesh& mesh) {
            instBindings_.erase(
                    std::remove_if(instBindings_.begin(), instBindings_.end(),
                                   [&](const InstBinding& b) { return b.mesh == &mesh; }),
                    instBindings_.end());
        }

        // Remove a body (static or dynamic) from the world: drop any mesh binding that mirrors it, take it
        // out of the scene, and release it. The PxRigidActor* dangles afterwards — the caller must not reuse
        // a RigidBody handle to a removed actor. Use e.g. to rebuild geometry (a configurable staircase)
        // without recreating the whole world.
        void removeActor(::physx::PxRigidActor* actor) {
            if (!actor) return;
            objBindings_.erase(
                    std::remove_if(objBindings_.begin(), objBindings_.end(),
                                   [&](const ObjBinding& b) { return b.actor == actor; }),
                    objBindings_.end());
            // Associations name actors too, and a stale one would hand a sensor
            // a released body on the next findActor.
            associations_.erase(
                    std::remove_if(associations_.begin(), associations_.end(),
                                   [&](const Association& a) { return a.actor == actor; }),
                    associations_.end());
            // InstancedMesh bindings hold actor pointers too, and used to be
            // skipped here: the actor was released while instBindings_ still
            // named it, so the next sync dereferenced freed memory.
            //
            // Null the slot rather than erasing it — instance i mirrors
            // actors[i], so compacting the vector would silently repoint every
            // later instance at a different body. A nulled slot is skipped by
            // the sync loops, leaving that instance frozen at its last matrix;
            // hiding or moving it is the caller's call. A binding is dropped
            // only once every actor in it is gone.
            for (auto& b : instBindings_) {
                for (auto*& a : b.actors) {
                    if (a == actor) a = nullptr;
                }
            }
            instBindings_.erase(
                    std::remove_if(instBindings_.begin(), instBindings_.end(),
                                   [](const InstBinding& b) {
                                       return std::all_of(b.actors.begin(), b.actors.end(),
                                                          [](const ::physx::PxRigidActor* a) { return a == nullptr; });
                                   }),
                    instBindings_.end());
            // Registered sensors cache the actor they ride (resolved once at
            // registration), so they need the same treatment as the bindings
            // above: tell them before the release, or the next substep samples
            // freed memory.
            for (auto* s: sensors_) s->onActorRemoved(actor);
            // Same reasoning for contact watchers: a stale entry would otherwise
            // survive and could match a recycled pointer.
            contacts_.forget(actor);
            scene_->removeActor(*actor);
            actor->release();
        }

        // Low-level: explicit geometry + transform. Use when shape can't be inferred
        // (custom geometry, plane, trimesh) or when shape != mesh visuals.
        ::physx::PxRigidStatic* addStatic(const ::physx::PxGeometry& geom,
                                          const ::physx::PxTransform& tr,
                                          ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            if (!mat) mat = defaultMat_;
            PxRigidStatic* body = physics_->createRigidStatic(tr);
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            scene_->addActor(*body);
            return body;
        }

        ::physx::PxRigidDynamic* addDynamic(const ::physx::PxGeometry& geom,
                                            const ::physx::PxTransform& tr,
                                            float density,
                                            ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            if (!mat) mat = defaultMat_;
            PxRigidDynamic* body = physics_->createRigidDynamic(tr);
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            PxRigidBodyExt::updateMassAndInertia(*body, density);
            scene_->addActor(*body);
            return body;
        }

        // High-level: shape inferred from mesh.geometry(), pose from mesh.matrixWorld,
        // automatic transform sync. Supported geometries: Box, Sphere, Capsule.
        // For unsupported geometries, fall back to addDynamic / addStatic.
        ::physx::PxRigidDynamic* add(Mesh& mesh, float density, ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::add: mesh has no geometry");
            if (!mat) mat = defaultMat_;
            const auto pl = worldPlacement(mesh);
            auto inferred = physx_detail::inferShape(*g, pl.scale);
            if (!inferred.valid) {
                throw std::runtime_error(
                        "PhysxWorld::add: cannot infer a collider from '" + g->type() +
                        "'. Shape inference covers Box/Sphere/Capsule geometry; use "
                        "addDynamicConvex (convex hull) for a dynamic body, or "
                        "addStaticTrimesh (exact triangles) for a static collider.");
            }
            PxRigidDynamic* body = physics_->createRigidDynamic(pl.pose);
            PxShape* shape = physics_->createShape(inferred.geom.any(), *mat, true);
            shape->setLocalPose(inferred.localPose);
            body->attachShape(*shape);
            shape->release();
            PxRigidBodyExt::updateMassAndInertia(*body, density);
            scene_->addActor(*body);
            bind(mesh, *body);
            return body;
        }

        ::physx::PxRigidStatic* addStatic(Mesh& mesh, ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::addStatic: mesh has no geometry");
            if (!mat) mat = defaultMat_;
            const auto pl = worldPlacement(mesh);
            auto inferred = physx_detail::inferShape(*g, pl.scale);
            if (!inferred.valid) {
                throw std::runtime_error(
                        "PhysxWorld::addStatic: cannot infer a collider from '" + g->type() +
                        "'. Shape inference covers Box/Sphere/Capsule geometry; use "
                        "addStaticTrimesh (exact triangles), addDynamicConvex (convex hull), "
                        "or a thin BoxGeometry as a floor.");
            }
            PxRigidStatic* body = physics_->createRigidStatic(pl.pose);
            PxShape* shape = physics_->createShape(inferred.geom.any(), *mat, true);
            shape->setLocalPose(inferred.localPose);
            body->attachShape(*shape);
            shape->release();
            scene_->addActor(*body);
            return body;
        }

        // Cook a triangle mesh from arbitrary indexed geometry and add it as a static
        // collider. Returns nullptr if the geometry has no position attribute, fewer
        // than 3 vertices, or a non-triangle index buffer. Trimesh colliders are valid
        // only for static / kinematic actors — dynamics need a convex decomposition.
        ::physx::PxRigidStatic* addStaticTrimesh(
                const BufferGeometry& geometry,
                const ::physx::PxTransform& tr = ::physx::PxTransform(::physx::PxIdentity),
                const Vector3& scale = Vector3(1.f, 1.f, 1.f),
                ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            const auto* posAttr = geometry.getAttribute<float>("position");
            if (!posAttr) return nullptr;
            const auto& positions = posAttr->array();
            const PxU32 vertCount = static_cast<PxU32>(posAttr->count());
            if (vertCount < 3) return nullptr;

            // Build a u32 index buffer (copied from geometry index, or 0..N-1 for unindexed).
            std::vector<PxU32> indices;
            const auto* idxAttr = geometry.getIndex();
            if (idxAttr) {
                const auto& src = idxAttr->array();
                indices.assign(src.begin(), src.end());
            } else {
                indices.resize(vertCount);
                for (PxU32 i = 0; i < vertCount; ++i) indices[i] = i;
            }
            if (indices.size() < 3 || indices.size() % 3 != 0) return nullptr;

            PxTriangleMeshDesc desc;
            desc.points.count = vertCount;
            desc.points.stride = sizeof(float) * 3;
            desc.points.data = positions.data();
            desc.triangles.count = static_cast<PxU32>(indices.size() / 3);
            desc.triangles.stride = sizeof(PxU32) * 3;
            desc.triangles.data = indices.data();

            PxCookingParams params(physics_->getTolerancesScale());
            PxTriangleMesh* triMesh = PxCreateTriangleMesh(params, desc);
            if (!triMesh) return nullptr;

            if (!mat) mat = defaultMat_;
            PxTriangleMeshGeometry geom(triMesh, PxMeshScale(toPxVec3(scale)));
            PxRigidStatic* body = physics_->createRigidStatic(tr);
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            // Shape retains a reference; release our local one so the mesh is freed
            // when the shape (and therefore actor) goes away.
            triMesh->release();
            scene_->addActor(*body);
            return body;
        }

        // Build a static trimesh from mesh.geometry(); pose and scale come from
        // mesh.matrixWorld (decomposed). Convenience wrapper around the geometry overload.
        ::physx::PxRigidStatic* addStaticTrimesh(Mesh& mesh, ::physx::PxMaterial* mat = nullptr) {
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::addStaticTrimesh: mesh has no geometry");
            const auto pl = worldPlacement(mesh);
            return addStaticTrimesh(*g, pl.pose, pl.scale, mat);
        }

        // Walk the subtree and add every Mesh as its own static trimesh. Useful for
        // imported scenes (glTF tracks, environments) where the visual hierarchy is
        // also the collision geometry. Returns the created actors.
        // `filter` lets the caller skip meshes (e.g. dynamic obstacles); pass nullptr
        // to accept all.
        std::vector<::physx::PxRigidStatic*> addStaticTrimeshTree(
                Object3D& root,
                const std::function<bool(const Mesh&)>& filter = nullptr,
                ::physx::PxMaterial* mat = nullptr) {
            std::vector<::physx::PxRigidStatic*> out;
            root.updateWorldMatrix(true, true);
            root.traverseType<Mesh>([&](Mesh& m) {
                if (filter && !filter(m)) return;
                if (auto* body = addStaticTrimesh(m, mat)) out.push_back(body);
            });
            return out;
        }

        // Add a 2.5D height field as a static collider, in a Z-UP world.
        //
        // Why this exists next to addStaticTrimesh: a height field is watertight by
        // construction — one height per cell means a HOLE is not representable, and
        // neither is a near-vertical spike — which is exactly what a marching-cubes
        // bake of a scan is not. It is also ~4 bytes per sample against ~100 for the
        // same surface cooked as a triangle soup, and PhysX's height-field
        // narrowphase indexes the cell under a contact instead of walking a BVH.
        //
        // `heights` is row-major with the ROW index running along WORLD Y:
        //
        //     heights[iy * nx + ix]  is the surface z at
        //     (origin.x + ix*cell, origin.y + iy*cell, origin.z + heights[...])
        //
        // i.e. the natural NumPy layout of an (ny, nx) array. `cell` is the sample
        // spacing in metres, the same in x and y.
        //
        // Axis mapping. PhysX height fields are Y-UP in the shape's LOCAL frame:
        // the row index runs along local +x, the column index along local +z, the
        // height along local +y. Mapping local +y to world +z while keeping local
        // +x on world +x forces the third axis: local +z must go to world -y, or
        // the map is a reflection and no quaternion can express it. So the shape's
        // local pose is a QUARTER TURN ABOUT +X (local y -> world z, local z ->
        // world -y) and the columns are stored REVERSED (column c holds row
        // iy = ny-1-c), with the pose's translation moved to the far y edge to
        // compensate. Sample (r, c) then lands at world
        //     (origin.x + r*cell, origin.y + (ny-1-c)*cell, origin.z + h),
        // which is exactly the contract above. PhysxWorld_test's analytic-ramp
        // probe (an asymmetric z = 0.1x + 0.3 sin y) is what pins this down: a
        // transposed grid or a flipped axis fails it.
        //
        // Heights are quantised to int16: heightScale is picked from the actual
        // z range so the range spans 32000 counts (a 20 m range quantises at
        // 0.6 mm), with a 1e-4 m floor for a flat or near-flat field.
        //
        // `thickness` is how deep below the surface the caller wants the field to
        // stay solid. PhysX 5 has NO such knob: PxHeightFieldDesc::thickness was a
        // PhysX 3 field and is gone from the 5.5 header. The parameter is accepted
        // and validated (> 0) so callers can express the intent, but PhysX cannot
        // honour it, and a height field is NOT a solid volume. Measured, in
        // PhysxWorld_test: a 4 cm ball thrown down at 6 m/s (0.113 m of travel per
        // 1/60 s substep, no CCD) passes straight through this height field —
        // 12 probes of 12 — and through the identical surface cooked as a trimesh,
        // 12 of 12, at the same rest depth to four decimals. Discrete collision
        // detection is the limit, not the shape: nothing tunnels once the probe's
        // diameter exceeds its per-substep travel (a 16 cm ball at the same 6 m/s
        // holds on all 12). If a caller genuinely needs a fast small body to stop
        // here, the fix is CCD or a smaller substep, and it is the same fix for a
        // trimesh. What the height field buys over the bake is holes and spikes.
        //
        // Returns nullptr on a degenerate field (nx or ny < 2, cell <= 0, no finite
        // sample) or a cook failure.
        ::physx::PxRigidStatic* addStaticHeightField(
                const float* heights, int nx, int ny, float cell,
                const Vector3& origin, float thickness,
                ::physx::PxMaterial* mat) {
            using namespace ::physx;
            if (!heights || nx < 2 || ny < 2 || !(cell > 0.f) || !(thickness > 0.f)) return nullptr;

            const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

            // z range over the finite samples only; a NaN cell is pinned to the floor
            // rather than poisoning the quantisation for the whole field.
            float zMin = std::numeric_limits<float>::infinity();
            float zMax = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < n; ++i) {
                const float h = heights[i];
                if (!std::isfinite(h)) continue;
                zMin = std::min(zMin, h);
                zMax = std::max(zMax, h);
            }
            if (!std::isfinite(zMin) || !std::isfinite(zMax)) return nullptr;

            // 32000 of the int16's 32767 counts, so rounding can never overflow.
            const float heightScale = std::max((zMax - zMin) / 32000.f, 1e-4f);
            const float invScale = 1.f / heightScale;

            std::vector<PxHeightFieldSample> samples(n);
            for (int ix = 0; ix < nx; ++ix) {
                for (int c = 0; c < ny; ++c) {
                    const int iy = ny - 1 - c;// columns reversed: see the quarter turn above
                    const float h = heights[static_cast<std::size_t>(iy) * nx + ix];
                    const float q = (std::isfinite(h) ? h : zMin) - zMin;
                    auto& s = samples[static_cast<std::size_t>(ix) * ny + c];
                    s.height = static_cast<PxI16>(std::lround(q * invScale));
                    s.materialIndex0 = 0;
                    s.materialIndex1 = 0;
                }
            }

            PxHeightFieldDesc desc;
            desc.format = PxHeightFieldFormat::eS16_TM;
            desc.nbRows = static_cast<PxU32>(nx);
            desc.nbColumns = static_cast<PxU32>(ny);
            desc.samples.data = samples.data();
            desc.samples.stride = sizeof(PxHeightFieldSample);

            PxHeightField* hf = PxCreateHeightField(desc);
            if (!hf) return nullptr;

            if (!mat) mat = defaultMat_;
            PxHeightFieldGeometry geom(hf, PxMeshGeometryFlags(), heightScale, cell, cell);

            PxRigidStatic* body = physics_->createRigidStatic(PxTransform(PxIdentity));
            PxShape* shape = physics_->createShape(geom, *mat, true);
            shape->setLocalPose(PxTransform(
                    PxVec3(origin.x, origin.y + static_cast<float>(ny - 1) * cell, origin.z + zMin),
                    PxQuat(PxPiDivTwo, PxVec3(1.f, 0.f, 0.f))));
            body->attachShape(*shape);
            shape->release();
            hf->release();// the shape holds its own reference
            scene_->addActor(*body);
            return body;
        }

        // Vector overload. Same contract; `heights` must hold exactly nx*ny samples.
        // (Two explicit overloads rather than one signature with defaults after a
        // pointer: MSVC and GCC disagree about nested `= {}` defaults, and the house
        // fix is to spell the overloads out.)
        ::physx::PxRigidStatic* addStaticHeightField(
                const std::vector<float>& heights, int nx, int ny, float cell,
                const Vector3& origin, float thickness = 0.5f,
                ::physx::PxMaterial* mat = nullptr) {
            if (nx < 2 || ny < 2) return nullptr;
            if (heights.size() != static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)) {
                throw std::runtime_error(
                        "PhysxWorld::addStaticHeightField: heights holds " +
                        std::to_string(heights.size()) + " samples, expected nx*ny = " +
                        std::to_string(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)));
            }
            return addStaticHeightField(heights.data(), nx, ny, cell, origin, thickness, mat);
        }

        // Cook a convex hull from raw positions (tightly packed x,y,z floats).
        // The vertex-count limit is clamped to PhysX's hard ceiling of 255; 64 is
        // the sane default the editor cooks decomposed hulls at (a hull with more
        // than a few dozen planes buys precision no rigid-body contact needs and
        // slows narrowphase). Returns nullptr on a degenerate input (< 4 points)
        // or a cook failure; the caller owns the returned PxConvexMesh and must
        // release() it once its shapes are built.
        ::physx::PxConvexMesh* cookConvexHull(const float* positions, std::size_t vertCount,
                                              unsigned maxVerts = 64) {
            using namespace ::physx;
            if (!positions || vertCount < 4) return nullptr;

            PxConvexMeshDesc desc;
            desc.points.count = static_cast<PxU32>(vertCount);
            desc.points.stride = sizeof(float) * 3;
            desc.points.data = positions;
            desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
            // Without ePLANE_SHIFTING the cooker's vertexLimit floor is 8, and 64
            // is the ceiling for a GPU-compatible hull (the world may run GPU
            // dynamics) — so clamp into [8, 64] rather than PhysX's raw [4, 255].
            desc.vertexLimit = static_cast<PxU16>(std::clamp<unsigned>(maxVerts, 8u, 64u));

            PxCookingParams params(physics_->getTolerancesScale());
            return PxCreateConvexMesh(params, desc);
        }

        // Cook a convex hull from the mesh vertices and add it as a dynamic actor.
        // PhysX dynamics require convex shapes — for non-convex visuals (a traffic
        // cone, an L-bracket) the hull is the smallest convex envelope. Pose and
        // scale come from mesh.matrixWorld; the mesh is bound to the new actor so
        // it follows the simulation.
        ::physx::PxRigidDynamic* addDynamicConvex(
                Mesh& mesh, float density, ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::addDynamicConvex: mesh has no geometry");
            const auto* posAttr = g->getAttribute<float>("position");
            if (!posAttr) return nullptr;
            const auto& positions = posAttr->array();

            PxConvexMesh* convex = cookConvexHull(positions.data(), posAttr->count());
            if (!convex) return nullptr;

            if (!mat) mat = defaultMat_;
            const auto pl = worldPlacement(mesh);

            PxConvexMeshGeometry geom(convex, PxMeshScale(toPxVec3(pl.scale)));
            PxRigidDynamic* body = physics_->createRigidDynamic(pl.pose);
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            convex->release();
            PxRigidBodyExt::updateMassAndInertia(*body, density);
            scene_->addActor(*body);
            bind(mesh, *body);
            return body;
        }

        // One convex shape in a compound actor: a cooked hull, where it sits in
        // the actor's local frame, and the per-hull scale baked into that shape.
        // The mesh is NOT owned here — cook it with cookConvexHull, hand the parts
        // to addCompound, and release each hull afterwards (attachShape keeps its
        // own reference, so the actor survives the release).
        struct ConvexPart {
            ::physx::PxConvexMesh* mesh = nullptr;
            ::physx::PxTransform localPose{::physx::PxIdentity};
            Vector3 scale{1.f, 1.f, 1.f};
        };

        // Build ONE rigid actor (dynamic or static) carrying several convex
        // shapes, each at its own local pose. This is what a multi-hull collider
        // is: an imported prop whose sub-meshes each become a hull, or a single
        // concave mesh V-HACD split into convex pieces, all welded into one body
        // so they move as a rigid whole.
        //
        // For a dynamic body the mass properties come from the UNION of the
        // shapes via a single updateMassAndInertia after they all attach — the
        // inertia of the whole, not of any one piece. `pose` is the actor's world
        // transform; the parts' localPose/scale are relative to it. Binding (so a
        // visual follows the sim) is the caller's job — it holds the Object3D.
        ::physx::PxRigidActor* addCompound(const ::physx::PxTransform& pose,
                                           const std::vector<ConvexPart>& parts,
                                           bool dynamic, float density,
                                           ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            if (parts.empty()) return nullptr;
            if (!mat) mat = defaultMat_;

            PxRigidActor* actor = dynamic
                                          ? static_cast<PxRigidActor*>(physics_->createRigidDynamic(pose))
                                          : static_cast<PxRigidActor*>(physics_->createRigidStatic(pose));
            std::size_t attached = 0;
            for (const auto& part : parts) {
                if (!part.mesh) continue;
                PxConvexMeshGeometry geom(part.mesh, PxMeshScale(toPxVec3(part.scale)));
                PxShape* shape = physics_->createShape(geom, *mat, true);
                shape->setLocalPose(part.localPose);
                actor->attachShape(*shape);
                shape->release();
                ++attached;
            }
            if (attached == 0) {
                actor->release();
                return nullptr;
            }
            if (dynamic) {
                // Mass from the union of every hull, computed once — the whole
                // body's inertia, not a per-piece sum that ignores the offsets.
                PxRigidBodyExt::updateMassAndInertia(*static_cast<PxRigidDynamic*>(actor), density);
            }
            scene_->addActor(*actor);
            return actor;
        }

        // One PxRigidDynamic per instance. Initial pose taken from each instance matrix.
        std::vector<::physx::PxRigidActor*> add(InstancedMesh& mesh,
                                                float density,
                                                ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::add(InstancedMesh): no geometry");
            if (!physx_detail::inferShape(*g).valid) {
                throw std::runtime_error(
                        "PhysxWorld::add(InstancedMesh): cannot infer a collider from '" + g->type() +
                        "'. Shape inference covers Box/Sphere/Capsule geometry; instanced bodies "
                        "have no trimesh/convex fallback, so use one of those geometry types.");
            }
            if (!mat) mat = defaultMat_;
            std::vector<PxRigidActor*> actors;
            actors.reserve(mesh.count());
            Matrix4 m;
            Vector3 pos, scale;
            Quaternion rot;
            for (size_t i = 0; i < mesh.count(); ++i) {
                mesh.getMatrixAt(i, m);
                m.decompose(pos, rot, scale);
                // Per-instance shape: the mirror preserves per-instance scale
                // (see bind), so the collider must carry it too.
                const auto inferred = physx_detail::inferShape(*g, scale);
                PxRigidDynamic* body = physics_->createRigidDynamic(
                        PxTransform(toPxVec3(pos), toPxQuat(rot)));
                PxShape* shape = physics_->createShape(inferred.geom.any(), *mat, true);
                shape->setLocalPose(inferred.localPose);
                body->attachShape(*shape);
                shape->release();
                PxRigidBodyExt::updateMassAndInertia(*body, density);
                scene_->addActor(*body);
                actors.push_back(body);
            }
            bind(mesh, actors);
            return actors;
        }

        ::physx::PxPhysics& physics() { return *physics_; }
        ::physx::PxScene& scene() { return *scene_; }
        ::physx::PxFoundation& foundation() { return *foundation_; }
        ::physx::PxMaterial& defaultMaterial() { return *defaultMat_; }
        ::physx::PxCpuDispatcher& dispatcher() { return *dispatcher_; }

        // CUDA context — non-null only when Settings::enableGpuDynamics was set.
        // Required for soft bodies, particle systems, and GPU broadphase.
        ::physx::PxCudaContextManager* cudaContextManager() const { return cuda_; }

        const Settings& settings() const { return settings_; }
        bool directGpuEnabled() const { return settings_.enableDirectGpu; }

        // Raw fixed-rate substep with no binding sync — simulate() + fetchResults().
        // Use under direct-GPU (where the binding-sync step() is invalid); state is
        // read/written through PhysxGpuBatch rather than CPU getters.
        void simulateRaw(float dt) {
            scene_->simulate(dt);
            scene_->fetchResults(true);
        }

        void setGravity(const Vector3& g) {
            settings_.gravity = g;
            scene_->setGravity(toPxVec3(g));
        }

        // --- Soft body API (requires Settings::enableGpuDynamics). Implementations
        // live in PhysxSoftBody.hpp because they depend on the full SoftBody type.

        // Create a deformable-volume material. Owned by PxPhysics; no manual release.
        ::physx::PxDeformableVolumeMaterial* createSoftBodyMaterial(
                float youngsModulus = 1e6f, float poissonsRatio = 0.45f,
                float dynamicFriction = 0.5f);

        // Cook the supplied geometry into a deformable volume and add it to the
        // scene. The geometry's position attribute is mutated each frame to match
        // the deformed simulation. The geometry's positions are taken as-is (world
        // space). voxelResolution sets the simulation mesh detail (~10 default;
        // higher = finer simulation + more solver work).
        // mass (kg): when > 0 the body's total mass is set to this value; 0 leaves
        // the default unit-density mass derived from the tet volume.
        SoftBody* addSoftBody(
                const std::shared_ptr<BufferGeometry>& visualGeometry,
                ::physx::PxDeformableVolumeMaterial* material = nullptr,
                int voxelResolution = 10,
                unsigned solverIterations = 20,
                bool selfCollision = false,
                float mass = 0.f);

        // Convenience: bake mesh.matrixWorld into the geometry positions, reset
        // the mesh's local transform to identity, then add as a soft body. Useful
        // for typical scene-graph workflows (`mesh->position.set(...)` then add).
        // When cacheKey is non-empty, the expensive tet mesh cooking and per-vertex
        // binding computation are cached and reused for subsequent calls with the
        // same key (e.g. fish species ID). Only the first spawn of each key pays
        // the full cook cost.
        SoftBody* addSoftBody(
                Mesh& mesh,
                ::physx::PxDeformableVolumeMaterial* material = nullptr,
                int voxelResolution = 10,
                unsigned solverIterations = 20,
                bool selfCollision = false,
                const std::string& cacheKey = "",
                float mass = 0.f);

        // Resolve the SoftBody whose visual Mesh is `obj`, or whose Mesh is the
        // nearest ancestor of it. nullptr when this world simulates no soft
        // body for that object. The rigid-side counterpart is findActor().
        [[nodiscard]] SoftBody* findSoftBody(const Object3D* obj) const;

        // Destroy a soft body. Releases the PhysX actor + GPU/pinned resources and
        // — when the body was created via the Mesh& overload — also removes the
        // visual Mesh from its parent in the scene graph (single-call cleanup).
        // Soft bodies created from a bare BufferGeometry leave the scene graph alone.
        void removeSoftBody(SoftBody* softBody);

    private:
        // Bindings carry a `prevPose` snapshot taken right before each substep so
        // visual output can lerp(prev, current, alpha) where alpha is the leftover
        // accumulator fraction. `hasPrev` gates the first frame before any
        // snapshot has been taken (avoids interpolating against an identity pose).
        struct ObjBinding {
            Object3D* obj;
            ::physx::PxRigidActor* actor;
            ::physx::PxTransform prevPose{::physx::PxIdentity};
            bool hasPrev = false;
        };
        struct InstBinding {
            InstancedMesh* mesh;
            std::vector<::physx::PxRigidActor*> actors;
            std::vector<::physx::PxTransform> prevPoses;
            bool hasPrev = false;
        };

        void substep(float dt) {
            // Iterate by index, re-reading size each time: a callback is allowed
            // to register or remove a substep callback (a vehicle tearing itself
            // down mid-step), which would invalidate a range-for's iterators.
            for (size_t i = 0; i < preSubstep_.size(); ++i) preSubstep_[i].fn(dt);
            scene_->simulate(dt);
            scene_->fetchResults(true);
            for (size_t i = 0; i < postSubstep_.size(); ++i) postSubstep_[i].fn(dt);
            // Body states are fresh here (post fetchResults). Advance the sim clock
            // and drive any registered sensors — appended after existing hooks so
            // the dt / stepping path above is untouched.
            simTime_ += static_cast<double>(dt);
            ++substeps_;
            for (auto* s : sensors_) s->tick(static_cast<double>(dt), simTime_);
        }

        // Pull deformed positions GPU->CPU for every soft body, then write them into
        // the bound BufferGeometry's position attribute. Defined in PhysxSoftBody.hpp.
        void syncSoftBodies();

        ::physx::PxDeformableVolumeMaterial* defaultSoftBodyMaterial();

        // Snapshot every bound actor's current pose as the "before this substep"
        // state. Called once per substep iteration so prev = pose just before the
        // most recent simulate(); current = pose just after fetchResults().
        void snapshotPrevPoses() {
            for (auto& b : objBindings_) {
                b.prevPose = b.actor->getGlobalPose();
                b.hasPrev = true;
            }
            for (auto& b : instBindings_) {
                if (b.prevPoses.size() != b.actors.size()) {
                    b.prevPoses.resize(b.actors.size());
                }
                for (size_t i = 0; i < b.actors.size(); ++i) {
                    if (!b.actors[i]) continue;// removeActor() nulled this slot
                    b.prevPoses[i] = b.actors[i]->getGlobalPose();
                }
                b.hasPrev = true;
            }
        }

        // Write actor poses (interpolated between prev and current by alpha) into
        // bound Object3D / InstancedMesh transforms.
        void syncRigidBindings(float alpha) {
            for (auto& b : objBindings_) {
                const auto cur = b.actor->getGlobalPose();
                const auto t = b.hasPrev ? lerpPxTransform(b.prevPose, cur, alpha) : cur;
                const Vector3 worldPos = fromPxVec3(t.p);
                const Quaternion worldRot = fromPxQuat(t.q);
                if (auto* parent = b.obj->parent) {
                    // Convert world pose into parent-local pose so the visual lands
                    // correctly when the bound object is nested under transformed groups.
                    parent->updateWorldMatrix(true, false);
                    Matrix4 worldMat;
                    worldMat.compose(worldPos, worldRot, Vector3(1.f, 1.f, 1.f));
                    Matrix4 inverseParent;
                    inverseParent.copy(*parent->matrixWorld).invert();
                    Matrix4 localMat;
                    localMat.multiplyMatrices(inverseParent, worldMat);
                    Vector3 unusedScale;
                    localMat.decompose(b.obj->position, b.obj->quaternion, unusedScale);
                } else {
                    b.obj->position.copy(worldPos);
                    b.obj->quaternion.copy(worldRot);
                }
            }
            for (auto& b : instBindings_) {
                Matrix4 m;
                Vector3 pos, scale;
                Quaternion rot;
                for (size_t i = 0; i < b.actors.size(); ++i) {
                    if (!b.actors[i]) continue;// removeActor() nulled this slot
                    b.mesh->getMatrixAt(i, m);
                    m.decompose(pos, rot, scale);
                    const auto cur = b.actors[i]->getGlobalPose();
                    const auto t = b.hasPrev ? lerpPxTransform(b.prevPoses[i], cur, alpha) : cur;
                    m.compose(fromPxVec3(t.p), fromPxQuat(t.q), scale);
                    b.mesh->setMatrixAt(i, m);
                }
                b.mesh->instanceMatrix()->needsUpdate();
                b.mesh->computeBoundingSphere();
            }
        }

        Settings settings_;
        ::physx::PxDefaultAllocator allocator_;
        ::physx::PxDefaultErrorCallback errorCallback_;
        ::physx::PxFoundation* foundation_ = nullptr;
        ::physx::PxPhysics* physics_ = nullptr;
        // Whether PxInitExtensions succeeded — releaseCore() must not call
        // PxCloseExtensions on a physics whose extensions never came up.
        bool extensionsInitialized_ = false;
        ::physx::PxDefaultCpuDispatcher* dispatcher_ = nullptr;
        ::physx::PxScene* scene_ = nullptr;
        ::physx::PxMaterial* defaultMat_ = nullptr;
        ::physx::PxCudaContextManager* cuda_ = nullptr;
        CUstream cudaCopyStream_ = nullptr;
        ::physx::PxDeformableVolumeMaterial* defaultSoftBodyMat_ = nullptr;
        struct CookCacheEntry {
            ::physx::PxDeformableVolumeMesh* mesh = nullptr;
            std::vector<SoftBodyTetBind> bindings;
            bool hasBindings = false;
        };
        std::unordered_map<std::string, CookCacheEntry> cookCache_;
        std::vector<std::unique_ptr<SoftBody>> softBodies_;
        float accumulator_ = 0.f;
        float dtEma_ = 0.f;// smoothed timestep (see Settings::smoothTimestep); 0 = uninitialised
        double simTime_ = 0.0;// accumulated fixed-substep sim time; stamps sensor samples
        std::uint64_t substeps_ = 0;// substeps advanced, in step with simTime_

        std::vector<ObjBinding> objBindings_;
        std::vector<InstBinding> instBindings_;
        // Resolution-only entries for findActor — no pose write-back, no
        // interpolation state (which is all ObjBinding adds). See associate().
        struct Association {
            Object3D* obj;
            ::physx::PxRigidActor* actor;
        };
        std::vector<Association> associations_;
        struct SubstepEntry {
            SubstepHandle handle;
            std::function<void(float)> fn;
        };
        std::vector<SubstepEntry> preSubstep_;
        std::vector<SubstepEntry> postSubstep_;
        SubstepHandle nextSubstepHandle_ = 1;// 0 reserved as "no handle"
        std::vector<Sensor*> sensors_;// non-owning; sampled each substep
        // Installed as the scene's simulation event callback at construction. The
        // scene holds a bare pointer to it, but ~PhysxWorld releases the scene in
        // its body — before any member is destroyed — so this stays valid for as
        // long as the scene can call it.
        ContactDispatcher contacts_;
    };

}// namespace threepp

#include "threepp/extras/physx/PhysxSoftBody.hpp"

#endif//THREEPP_PHYSX_WORLD_HPP
