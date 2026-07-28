
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
#include <functional>
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
     * Routes PhysX contact notifications to per-actor callbacks.
     *
     * Owned by PhysxWorld and installed as the scene's simulation event
     * callback. Contact reporting is OPT-IN per actor (see
     * PhysxWorld::watchContacts): PhysX only generates these notifications for
     * pairs whose filter data asks for them, so a world with no contact
     * watchers pays nothing beyond the branch in the filter shader.
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

        // Drop every watcher of `actor` — called when the actor is released, so
        // a stale watch entry cannot match a recycled pointer later.
        void forget(const ::physx::PxRigidActor* actor) {
            entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                          [actor](const Entry& e) { return e.watch == actor; }),
                           entries_.end());
        }

        [[nodiscard]] bool empty() const { return entries_.empty(); }

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

        // Unused halves of the interface. PhysX requires all of them.
        void onConstraintBreak(::physx::PxConstraintInfo*, ::physx::PxU32) override {}
        void onWake(::physx::PxActor**, ::physx::PxU32) override {}
        void onSleep(::physx::PxActor**, ::physx::PxU32) override {}
        void onTrigger(::physx::PxTriggerPair*, ::physx::PxU32) override {}
        void onAdvance(const ::physx::PxRigidBody* const*, const ::physx::PxTransform*,
                       const ::physx::PxU32) override {}

    private:
        struct Entry {
            Handle handle;
            ::physx::PxRigidActor* watch;
            std::function<void(const ContactEvent&)> fn;
        };
        std::vector<Entry> entries_;
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

        if ((filterData0.word3 | filterData1.word3) & kContactReportFilterBit) {
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

            foundation_ = PxCreateFoundation(PX_PHYSICS_VERSION, allocator_, errorCallback_);
            if (!foundation_) throw std::runtime_error("PxCreateFoundation failed");

            // trackOutstandingAllocations=true matches the GPU samples; harmless when off.
            physics_ = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation_, PxTolerancesScale(),
                                       settings_.enableGpuDynamics);
            if (!physics_) throw std::runtime_error("PxCreatePhysics failed");

            if (!PxInitExtensions(*physics_, nullptr)) {
                throw std::runtime_error("PxInitExtensions failed");
            }

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

        ~PhysxWorld() {
            using namespace ::physx;
            // Soft bodies must be released BEFORE the scene/physics/cuda context;
            // their destructor releases the PxDeformableVolume actor and frees pinned
            // host memory through the CUDA context.
            softBodies_.clear();
            for (auto& [_, entry] : cookCache_) {
                if (entry.mesh) entry.mesh->release();
            }
            cookCache_.clear();
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
                PxCloseExtensions();
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

        [[nodiscard]] ContactDispatcher& contactDispatcher() { return contacts_; }

        // Accumulated simulation time (s) — sum of every fixed substep advanced so
        // far. This is the clock stamped onto sensor samples.
        [[nodiscard]] double simTime() const { return simTime_; }

        // Resolve the PxRigidActor that governs `obj`: walk up the scene graph from
        // obj to the root and return the actor bound (via bind()/add()) to the
        // nearest ancestor (or obj itself). nullptr if none is managed here. Used
        // by sensors to map their attachment node to a rigid body.
        [[nodiscard]] ::physx::PxRigidActor* findActor(const Object3D* obj) const {
            for (const Object3D* o = obj; o != nullptr; o = o->parent) {
                for (const auto& b : objBindings_) {
                    if (b.obj == o) return b.actor;
                }
            }
            return nullptr;
        }

        // After each step, copy actor's world pose into Object3D.position/quaternion.
        void bind(Object3D& obj, ::physx::PxRigidActor& actor) {
            objBindings_.push_back({&obj, &actor});
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
            auto inferred = inferShape(*g);
            if (!inferred.valid) throw std::runtime_error("PhysxWorld::add: unsupported geometry");
            if (!mat) mat = defaultMat_;
            mesh.updateMatrixWorld();
            Vector3 pos, scale;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scale);
            PxRigidDynamic* body = physics_->createRigidDynamic(
                    PxTransform(toPxVec3(pos), toPxQuat(rot)));
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
            auto inferred = inferShape(*g);
            if (!inferred.valid) throw std::runtime_error("PhysxWorld::addStatic: unsupported geometry");
            if (!mat) mat = defaultMat_;
            mesh.updateMatrixWorld();
            Vector3 pos, scale;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scale);
            PxRigidStatic* body = physics_->createRigidStatic(
                    PxTransform(toPxVec3(pos), toPxQuat(rot)));
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
            mesh.updateMatrixWorld();
            Vector3 pos, scale;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scale);
            return addStaticTrimesh(
                    *g,
                    ::physx::PxTransform(toPxVec3(pos), toPxQuat(rot)),
                    scale,
                    mat);
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
            root.updateMatrixWorld();
            root.traverseType<Mesh>([&](Mesh& m) {
                if (filter && !filter(m)) return;
                if (auto* body = addStaticTrimesh(m, mat)) out.push_back(body);
            });
            return out;
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
            const PxU32 vertCount = static_cast<PxU32>(posAttr->count());
            if (vertCount < 4) return nullptr;

            PxConvexMeshDesc desc;
            desc.points.count = vertCount;
            desc.points.stride = sizeof(float) * 3;
            desc.points.data = positions.data();
            desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;

            PxCookingParams params(physics_->getTolerancesScale());
            PxConvexMesh* convex = PxCreateConvexMesh(params, desc);
            if (!convex) return nullptr;

            if (!mat) mat = defaultMat_;
            mesh.updateMatrixWorld();
            Vector3 pos, scale;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scale);

            PxConvexMeshGeometry geom(convex, PxMeshScale(toPxVec3(scale)));
            PxRigidDynamic* body = physics_->createRigidDynamic(
                    PxTransform(toPxVec3(pos), toPxQuat(rot)));
            PxShape* shape = physics_->createShape(geom, *mat, true);
            body->attachShape(*shape);
            shape->release();
            convex->release();
            PxRigidBodyExt::updateMassAndInertia(*body, density);
            scene_->addActor(*body);
            bind(mesh, *body);
            return body;
        }

        // One PxRigidDynamic per instance. Initial pose taken from each instance matrix.
        std::vector<::physx::PxRigidActor*> add(InstancedMesh& mesh,
                                                float density,
                                                ::physx::PxMaterial* mat = nullptr) {
            using namespace ::physx;
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("PhysxWorld::add(InstancedMesh): no geometry");
            auto inferred = inferShape(*g);
            if (!inferred.valid) throw std::runtime_error("PhysxWorld::add(InstancedMesh): unsupported geometry");
            if (!mat) mat = defaultMat_;
            std::vector<PxRigidActor*> actors;
            actors.reserve(mesh.count());
            Matrix4 m;
            Vector3 pos, scale;
            Quaternion rot;
            for (size_t i = 0; i < mesh.count(); ++i) {
                mesh.getMatrixAt(i, m);
                m.decompose(pos, rot, scale);
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

        struct InferredShape {
            ::physx::PxGeometryHolder geom;
            ::physx::PxTransform localPose{::physx::PxIdentity};
            bool valid = true;
        };

        InferredShape inferShape(const BufferGeometry& geometry) const {
            using namespace ::physx;
            InferredShape out;
            if (auto box = dynamic_cast<const BoxGeometry*>(&geometry)) {
                out.geom = PxBoxGeometry(box->width * 0.5f, box->height * 0.5f, box->depth * 0.5f);
                return out;
            }
            if (auto sph = dynamic_cast<const SphereGeometry*>(&geometry)) {
                out.geom = PxSphereGeometry(sph->radius);
                return out;
            }
            if (auto cap = dynamic_cast<const CapsuleGeometry*>(&geometry)) {
                out.geom = PxCapsuleGeometry(cap->radius, cap->length * 0.5f);
                // PhysX capsule axis is X; threepp capsule axis is Y. Rotate -PI/2 about Z.
                out.localPose = PxTransform(PxQuat(-PxHalfPi, PxVec3(0, 0, 1)));
                return out;
            }
            out.valid = false;
            return out;
        }

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
                    parent->updateMatrixWorld();
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

        std::vector<ObjBinding> objBindings_;
        std::vector<InstBinding> instBindings_;
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
