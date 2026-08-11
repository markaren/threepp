// GPU PBD particles (PxPBDParticleSystem) on a borrowed PhysxWorld: the
// granular / fluid half of PhysX's GPU solver, which is what makes a conveyor
// carry SAND instead of crates.
//
// Header-only and PhysX-dependent, exactly like PhysxWorld and ConveyorPhysics
// — the threepp library proper never links PhysX, so this file is included only
// by builds that found the SDK. The headers ship with PhysX 5.5 whether or not
// the machine has CUDA, so this COMPILES everywhere; only construction needs a
// GPU (see the throw below), which is why the demo checks
// PhysxWorld::cudaContextManager() first.
//
// Shape of the binding, and why:
//
//   • ONE PxPBDParticleSystem holds the solver settings that are physically
//     global — the particle radius (rest offsets), the neighbourhood grid, the
//     iteration counts. Everything in one system collides with everything else
//     in it, which is the point: two materials pouring onto one belt must
//     interact.
//   • MANY groups, one PxParticleBuffer + one phase (PBD material + behaviour
//     flags) each. A buffer is the unit of both authoring and readback, so a
//     group is exactly "one pile of one material with its own mesh and colour":
//     no per-particle material bookkeeping on our side, no interleaved ranges
//     to un-mix when driving an InstancedMesh.
//   • Positions come back through pinned host mirrors, one async DtoH per group
//     on a private stream with a single synchronize — the same batched drain
//     PhysxWorld::syncSoftBodies() uses. That is pull(), and it is what a host
//     consumer (an InstancedMesh, the demo's own telemetry) needs.
//   • A RENDERER consumer does not need the host at all: Group::
//     registerVulkanMemory + copyPositionsToVulkan (F6) import the Vulkan
//     renderer's exported ParticleField positions allocation and copy into it
//     device to device, so nothing crosses the bus. pull() then exists only for
//     whoever actually wants the numbers on the CPU.
//
// The world is BORROWED, with the same contract as ConveyorPhysics: destroy
// this while the world is still alive (the destructor releases the particle
// buffers, the actor and the pinned memory THROUGH the CUDA context), or call
// abandon() first if the world is already gone.

#ifndef THREEPP_PHYSX_PARTICLES_HPP
#define THREEPP_PHYSX_PARTICLES_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>
#include <extensions/PxCudaHelpersExt.h>

#ifdef THREEPP_PHYSX_CUDA_VK_INTEROP
// CUDA driver API for external-memory import — the zero-copy bridge into the
// Vulkan renderer's EXPORTED ParticleField position buffer
// (VulkanRenderer::enableParticleFieldInterop). Needs the CUDA toolkit headers
// + driver library (CUDA::cuda_driver); see the Physics example CMake wiring.
#include <cuda.h>
#endif

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace threepp {

    class PbdParticles {

    public:
        // Solver-wide settings. `spacing` is the one knob that matters: it is
        // the distance at which two particles come to rest, so it doubles as
        // the diameter to draw them at. Everything else derives from it.
        struct Settings {
            // Resting distance between two touching particles (metres).
            float spacing = 0.06f;
            // Position iterations. Granular piles need more than fluid: too few
            // and a deep pile keeps sinking into itself under its own weight.
            unsigned solverIterations = 8;
            // Max neighbours the solver reserves per particle. Directly sizes a
            // GPU allocation (maxParticles * this * 4 bytes), so it is the first
            // thing to trim if the GPU heap runs out. 96 is PhysX's default and
            // is generous for a solid phase (dense packing sees ~30).
            unsigned maxNeighborhood = 96;
            bool selfCollision = true;
            // Speculative CCD against rigids. ON by default because the
            // alternative is not "slightly worse contact" but grains LEAVING the
            // world: a particle falling two metres arrives at several m/s and
            // crosses more than its own contact offset per substep, and discrete
            // collision then misses the floor entirely. Measured on this demo's
            // 1.9 m discharge — slick grains sank through a 0.4 m floor slab
            // until this was raised. \see PxParticleFlag::eENABLE_SPECULATIVE_CCD
            bool speculativeCcd = true;
            // Velocity clamp; 0 = derive from spacing (100 radii per second,
            // matching PhysX's own snippets). A clamp is not optional: an
            // emitter that overlaps existing particles depenetrates explosively
            // without one.
            float maxVelocity = 0.f;
        };

        // A PBD material. The PhysX factory takes eleven positional reals; this
        // struct exists so a caller can NAME the one or two that matter to it
        // instead of counting commas. For a granular material that is friction
        // (the repose angle of a heap is its internal friction) and damping;
        // there is deliberately no restitution knob, because PxPBDMaterial has
        // none.
        struct MaterialSpec {
            float friction = 0.4f;
            float damping = 0.f;
            float adhesion = 0.f;// sticks to RIGIDS (belt, walls)
            float viscosity = 0.f;
            float vorticityConfinement = 0.f;
            float surfaceTension = 0.f;
            float cohesion = 0.f;// sticks to OTHER PARTICLES — the pile-angle knob
            float cflCoefficient = 1.f;
            float gravityScale = 1.f;
            // Multiplies the rest offset to give the distance at which adhesion
            // stops pulling. 0 with a non-zero adhesion is a no-op.
            float adhesionRadiusScale = 0.f;
        };

        // One buffer + one phase: a pile of a single material, read back and
        // drawn on its own. Created by PbdParticles::addGroup, owned by it.
        class Group {

        public:
            Group(::physx::PxCudaContextManager& cuda, ::physx::PxParticleBuffer& buffer,
                  ::physx::PxPBDMaterial& material, ::physx::PxU32 phase, unsigned maxParticles)
                : cuda_(&cuda), buffer_(&buffer), material_(&material), phase_(phase),
                  maxParticles_(maxParticles) {

                // The readback mirror doubles as the upload staging for
                // positions: emit() writes the new slice here and copies just
                // that slice up, and pull() copies the whole active range back
                // over it. Both touch the same particles with the same values,
                // so there is nothing to reconcile.
                positions_ = PX_EXT_PINNED_MEMORY_ALLOC(::physx::PxVec4, *cuda_, maxParticles_);
                velocityChunk_ = PX_EXT_PINNED_MEMORY_ALLOC(::physx::PxVec4, *cuda_, kEmitChunk);
                phaseChunk_ = PX_EXT_PINNED_MEMORY_ALLOC(::physx::PxU32, *cuda_, kEmitChunk);
                if (!positions_ || !velocityChunk_ || !phaseChunk_) {
                    throw std::runtime_error("PbdParticles: pinned host allocation failed");
                }
                // A group's phase never changes, so its upload chunk is filled
                // once and re-sent verbatim.
                for (unsigned i = 0; i < kEmitChunk; ++i) phaseChunk_[i] = phase_;
            }

            ~Group() {
                if (!cuda_) return;
#ifdef THREEPP_PHYSX_CUDA_VK_INTEROP
                if (vkExtMem_) {
                    // CUDA side first, and the mapped buffer before the memory
                    // that backs it — the same order PhysxSoftBody's destructor
                    // documents. The renderer's own allocation outlives this:
                    // it goes when the field is swept or the device torn down.
                    ::physx::PxScopedCudaLock _lock(*cuda_);
                    if (vkPosPtr_) {
                        cuMemFree(vkPosPtr_);
                        vkPosPtr_ = 0;
                    }
                    cuDestroyExternalMemory(vkExtMem_);
                    vkExtMem_ = nullptr;
                }
#endif
                PX_EXT_PINNED_MEMORY_FREE(*cuda_, positions_);
                PX_EXT_PINNED_MEMORY_FREE(*cuda_, velocityChunk_);
                PX_EXT_PINNED_MEMORY_FREE(*cuda_, phaseChunk_);
            }

            Group(const Group&) = delete;
            Group& operator=(const Group&) = delete;

            // Append `count` particles at the given positions, all sharing one
            // launch velocity and mass. Returns how many were actually added:
            // an emitter that outruns the buffer just stops pouring rather than
            // corrupting the tail, so a demo can run forever on a fixed budget.
            //
            // Must be called between simulate() calls (i.e. not from inside a
            // substep hook) — it writes the device buffer the solver owns.
            unsigned emit(const Vector3* points, unsigned count, const Vector3& velocity, float mass) {

                using namespace ::physx;

                count = std::min(count, maxParticles_ - active_);
                if (count == 0 || !points) return 0;

                // invMass 0 would pin the particle in place; treat a
                // non-positive mass as "use 1 kg" rather than silently
                // producing immovable grains.
                const float invMass = 1.f / (mass > 0.f ? mass : 1.f);

                PxScopedCudaLock _lock(*cuda_);
                auto* ctx = cuda_->getCudaContext();
                auto* dstPos = buffer_->getPositionInvMasses();
                auto* dstVel = buffer_->getVelocities();
                auto* dstPhase = buffer_->getPhases();

                for (unsigned done = 0; done < count;) {
                    const unsigned n = std::min(kEmitChunk, count - done);
                    const unsigned base = active_ + done;
                    for (unsigned i = 0; i < n; ++i) {
                        const Vector3& p = points[done + i];
                        positions_[base + i] = PxVec4(p.x, p.y, p.z, invMass);
                        velocityChunk_[i] = PxVec4(velocity.x, velocity.y, velocity.z, 0.f);
                    }
                    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(dstPos + base),
                                    positions_ + base, n * sizeof(PxVec4));
                    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(dstVel + base),
                                    velocityChunk_, n * sizeof(PxVec4));
                    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(dstPhase + base),
                                    phaseChunk_, n * sizeof(PxU32));
                    done += n;
                }

                active_ += count;
                buffer_->setNbActiveParticles(active_);
                // Tell the solver to re-read what we just wrote. The flags cover
                // the WHOLE buffer, not our slice — that is fine (the rest of it
                // is the state the solver itself wrote last step) but it does
                // mean emitting every substep costs a full re-upload, so pour in
                // bursts instead.
                buffer_->raiseFlags(PxParticleBufferFlag::eUPDATE_POSITION);
                buffer_->raiseFlags(PxParticleBufferFlag::eUPDATE_VELOCITY);
                buffer_->raiseFlags(PxParticleBufferFlag::eUPDATE_PHASE);
                return count;
            }

            // Host mirror of the positions, valid for [0, active()) after the
            // owning PbdParticles::pull(). w = inverse mass, not a position.
            [[nodiscard]] const ::physx::PxVec4* positions() const { return positions_; }

            [[nodiscard]] unsigned active() const { return active_; }
            [[nodiscard]] unsigned capacity() const { return maxParticles_; }

            [[nodiscard]] ::physx::PxPBDMaterial& material() { return *material_; }
            [[nodiscard]] ::physx::PxParticleBuffer& buffer() { return *buffer_; }

#ifdef THREEPP_PHYSX_CUDA_VK_INTEROP
            // ── CUDA → Vulkan zero-copy (plans/particle-atmosphere.md F6) ────
            // The particle twin of PhysxSoftBody::registerVulkanMemory /
            // copyTetToVulkan, and deliberately the same shape: the renderer
            // backs an Ownership::Interop ParticleField's positions with an
            // EXPORTED dedicated allocation, this imports it once, and
            // copyPositionsToVulkan() — handed to the renderer as its per-frame
            // deviceCopy callback — moves the solver's positions device to
            // device. No DtoH pull, no pinned mirror, no host memcpy.
            //
            // Glue (after the first render, polled in the render loop):
            //   if (g->needsVkInteropRegister()) {
            //       auto h = vk->enableParticleFieldInterop(*field,
            //                        [g] { g->copyPositionsToVulkan(); });
            //       if (!h.osHandle || !g->registerVulkanMemory(h.osHandle, h.sizeBytes))
            //           ...keep the pull path...
            //   }
            [[nodiscard]] bool needsVkInteropRegister() const {
                return !vkInteropRegistered_ && !vkInteropTried_;
            }
            [[nodiscard]] bool vkInteropRegistered() const { return vkInteropRegistered_; }

            bool registerVulkanMemory(void* osHandle, std::size_t sizeBytes) {

                if (vkInteropRegistered_ || vkInteropTried_ || !osHandle || sizeBytes == 0)
                    return false;
                vkInteropTried_ = true;// attempt once; on failure the caller keeps pull()
                ::physx::PxScopedCudaLock _lock(*cuda_);
                CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd{};
#ifdef _WIN32
                hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
                hd.handle.win32.handle = osHandle;// NT handle stays the renderer's (CUDA dups it)
#else
                hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;// fd ownership transfers
                hd.handle.fd = static_cast<int>(reinterpret_cast<std::intptr_t>(osHandle));
#endif
                hd.size  = sizeBytes;
                hd.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;// the export is a dedicated VkDeviceMemory
                if (cuImportExternalMemory(&vkExtMem_, &hd) != CUDA_SUCCESS) {
                    vkExtMem_ = nullptr;
                    return false;
                }
                CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
                bd.offset = 0;
                bd.size   = sizeBytes;
                if (cuExternalMemoryGetMappedBuffer(&vkPosPtr_, vkExtMem_, &bd) != CUDA_SUCCESS) {
                    cuDestroyExternalMemory(vkExtMem_);
                    vkExtMem_ = nullptr;
                    vkPosPtr_ = 0;
                    return false;
                }
                vkInteropRegistered_ = true;
                return true;
            }

            // The renderer's per-frame deviceCopy callback. The solver's
            // positionInvMass block IS PxVec4 = {pos.xyz, invMass}, which is
            // byte-identical to threepp::ParticlePos, so this is a straight
            // device-to-device memcpy of the live prefix with no repack.
            //
            // SYNCHRONIZED before returning, and that is a contract rather than
            // caution: the renderer records the snapshot that reads this memory
            // into a command buffer it submits after the callback returns, so
            // "the copy has landed by then" is the entire ordering guarantee in
            // the absence of a shared Vulkan/CUDA semaphore. Same host-ordering
            // contract as the pull() + submit() it replaces.
            void copyPositionsToVulkan() const {
                if (!vkInteropRegistered_ || active_ == 0) return;
                ::physx::PxScopedCudaLock _lock(*cuda_);
                cuMemcpyDtoDAsync(vkPosPtr_,
                                  reinterpret_cast<CUdeviceptr>(buffer_->getPositionInvMasses()),
                                  std::size_t(active_) * sizeof(::physx::PxVec4), nullptr);
                cuStreamSynchronize(nullptr);
            }
#endif

        private:
            friend class PbdParticles;

            // Upload granularity. Bounds the pinned scratch to a few hundred kB
            // per group instead of one array per capacity; an emitter burst is
            // a few thousand particles, so this is one copy in practice.
            static constexpr unsigned kEmitChunk = 4096;

            ::physx::PxCudaContextManager* cuda_ = nullptr;
            ::physx::PxParticleBuffer* buffer_ = nullptr;
            ::physx::PxPBDMaterial* material_ = nullptr;
            ::physx::PxU32 phase_ = 0;
            unsigned maxParticles_ = 0;
            unsigned active_ = 0;

            ::physx::PxVec4* positions_ = nullptr;     // pinned, capacity
            ::physx::PxVec4* velocityChunk_ = nullptr; // pinned, kEmitChunk
            ::physx::PxU32* phaseChunk_ = nullptr;     // pinned, kEmitChunk

#ifdef THREEPP_PHYSX_CUDA_VK_INTEROP
            CUexternalMemory vkExtMem_ = nullptr;// imported Vulkan positions allocation
            CUdeviceptr      vkPosPtr_ = 0;      // its device pointer (the copy's dst)
            bool vkInteropRegistered_ = false;
            bool vkInteropTried_      = false;
#endif
        };

        // Throws when the world has no CUDA context — PBD has no CPU fallback
        // in PhysX, so a caller that wants to degrade gracefully must check
        // PhysxWorld::cudaContextManager() (or catch) and say so.
        PbdParticles(PhysxWorld& world, const Settings& settings)
            : world_(&world), settings_(settings) {

            using namespace ::physx;

            cuda_ = world.cudaContextManager();
            if (!cuda_) {
                throw std::runtime_error(
                        "PbdParticles: PhysxWorld::Settings::enableGpuDynamics is false — PBD "
                        "particle systems are a CUDA-only PhysX feature with no CPU path.");
            }

            system_ = world.physics().createPBDParticleSystem(*cuda_, settings_.maxNeighborhood);
            if (!system_) throw std::runtime_error("createPBDParticleSystem failed");

            // Offsets, all derived from spacing. Two solid particles come to
            // rest 2*solidRestOffset apart, so solidRestOffset is the particle
            // RADIUS and spacing is its diameter — which is why a demo can draw
            // spheres of exactly this radius and have contact look right.
            //
            // PhysX validates the ordering (restOffset < contactOffset;
            // particleContactOffset > both rest offsets) and silently misbehaves
            // if it is violated, so keep the multipliers, not the numbers.
            const float rest = solidRestOffset();
            system_->setSolidRestOffset(rest);
            system_->setFluidRestOffset(rest * 0.6f);
            system_->setRestOffset(rest);              // particle vs rigid
            system_->setContactOffset(rest * 1.5f);    // > restOffset
            system_->setParticleContactOffset(rest * 1.2f);// > both rest offsets
            system_->setSolverIterationCounts(settings_.solverIterations, 1);
            system_->setMaxVelocity(settings_.maxVelocity > 0.f ? settings_.maxVelocity
                                                                : rest * 100.f);
            // Depenetration is the other half of the same guard: an emitter that
            // drops a grain inside the pile must be pushed out gently, or the
            // pile erupts. 10 radii/s is firm enough to resolve in a frame.
            system_->setMaxDepenetrationVelocity(rest * 10.f);
            system_->setParticleFlag(PxParticleFlag::eDISABLE_SELF_COLLISION,
                                     !settings_.selfCollision);
            system_->setParticleFlag(PxParticleFlag::eENABLE_SPECULATIVE_CCD,
                                     settings_.speculativeCcd);
            world.scene().addActor(*system_);

            PxScopedCudaLock _lock(*cuda_);
            cuda_->getCudaContext()->streamCreate(&stream_, 0);
        }

        PbdParticles(const PbdParticles&) = delete;
        PbdParticles& operator=(const PbdParticles&) = delete;

        ~PbdParticles() {

            using namespace ::physx;

            if (!world_) {
                // abandon()ed: the world took the scene, the actor, the buffers
                // and the CUDA context with it. Dropping the pinned mirrors now
                // would be a free-through-a-dead-context, so leak them with the
                // process instead.
                for (auto& g : groups_) g->cuda_ = nullptr;
                return;
            }
            for (auto& g : groups_) {
                system_->removeParticleBuffer(&g->buffer());
                g->buffer().release();
            }
            groups_.clear();// frees the pinned mirrors while the context lives
            world_->scene().removeActor(*system_);
            system_->release();
            system_ = nullptr;
            if (stream_) {
                PxScopedCudaLock _lock(*cuda_);
                cuda_->getCudaContext()->streamDestroy(stream_);
                stream_ = nullptr;
            }
        }

        // Give up every reference to the world WITHOUT touching it — for the
        // teardown path where the world is already destroyed.
        void abandon() { world_ = nullptr; }

        // A new pile of one material, with its own capacity. Groups collide with
        // each other (one solver, one neighbourhood grid); they are separate only
        // for authoring and readback.
        Group& addGroup(unsigned maxParticles, const MaterialSpec& m) {

            using namespace ::physx;

            auto* mat = world_->physics().createPBDMaterial(
                    m.friction, m.damping, m.adhesion, m.viscosity, m.vorticityConfinement,
                    m.surfaceTension, m.cohesion, 0.f /*lift, deprecated*/,
                    0.f /*drag, deprecated*/, m.cflCoefficient, m.gravityScale);
            if (!mat) throw std::runtime_error("createPBDMaterial failed");
            mat->setAdhesionRadiusScale(m.adhesionRadiusScale);

            // A SOLID self-colliding phase: no eParticlePhaseFluid, so the
            // solver runs contact constraints instead of density constraints.
            // Fluid on granular material is the classic mistake — it makes sand
            // level itself like water and never hold a pile angle.
            const PxU32 phase = system_->createPhase(
                    mat, PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollide));

            auto* buffer = world_->physics().createParticleBuffer(maxParticles, 0, cuda_);
            if (!buffer) throw std::runtime_error("createParticleBuffer failed");
            buffer->setNbActiveParticles(0);

            // Build the Group (which allocates the pinned mirrors, and can
            // throw) BEFORE the system takes ownership of the buffer, and undo
            // the buffer if it does — otherwise a failed allocation leaves a
            // buffer registered with a particle system that has no group to
            // drive or release it.
            std::unique_ptr<Group> group;
            try {
                group = std::make_unique<Group>(*cuda_, *buffer, *mat, phase, maxParticles);
            } catch (...) {
                buffer->release();
                throw;
            }
            system_->addParticleBuffer(buffer);
            groups_.push_back(std::move(group));
            return *groups_.back();
        }

        // Drain every group's device positions into its host mirror. Call ONCE
        // per frame after world.step() — one CUDA lock, one async copy per group,
        // one synchronize, mirroring PhysxWorld::syncSoftBodies().
        void pull() {

            using namespace ::physx;

            if (!world_ || groups_.empty()) return;
            PxScopedCudaLock _lock(*cuda_);
            auto* ctx = cuda_->getCudaContext();
            bool any = false;
            for (auto& g : groups_) {
                if (g->active_ == 0) continue;
                ctx->memcpyDtoHAsync(g->positions_,
                                     reinterpret_cast<CUdeviceptr>(g->buffer_->getPositionInvMasses()),
                                     std::size_t(g->active_) * sizeof(PxVec4), stream_);
                any = true;
            }
            if (any) ctx->streamSynchronize(stream_);
        }

        [[nodiscard]] std::size_t groupCount() const { return groups_.size(); }
        [[nodiscard]] Group& group(std::size_t i) { return *groups_[i]; }

        // Total live particles across all groups.
        [[nodiscard]] unsigned active() const {
            unsigned n = 0;
            for (const auto& g : groups_) n += g->active();
            return n;
        }

        // The particle radius: half the resting distance between neighbours, and
        // the radius to draw a grain at.
        [[nodiscard]] float solidRestOffset() const { return settings_.spacing * 0.5f; }

        [[nodiscard]] const Settings& settings() const { return settings_; }
        [[nodiscard]] ::physx::PxPBDParticleSystem& system() { return *system_; }

    private:
        PhysxWorld* world_ = nullptr;
        Settings settings_;
        ::physx::PxCudaContextManager* cuda_ = nullptr;
        ::physx::PxPBDParticleSystem* system_ = nullptr;
        CUstream stream_ = nullptr;
        std::vector<std::unique_ptr<Group>> groups_;
    };

}// namespace threepp

#endif// THREEPP_PHYSX_PARTICLES_HPP
