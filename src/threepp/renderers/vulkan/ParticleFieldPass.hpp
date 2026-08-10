// ParticleFieldPass — the device-side half of threepp::ParticleField.
//
// Phase 0 of plans/particle-field.md, and deliberately only phase 0: this owns
// the per-field POSITION buffer (the one required buffer, §1.1), the per-field
// live-count block (§1.3) and the per-frame FieldDesc SSBO (§1.5). No shader
// reads any of it yet. Its reason to exist now is that everything phases 1-5
// add — mesh expansion, billboards, the density scatter, the AABB BLAS — reads
// exactly these three things, so getting their lifetime and their write window
// right once is what makes the later phases small.
//
// Frame flow (Impl::prepareParticleFields, called from renderFrame at the same
// site as prepareInstanceExpansion):
//
//   1. Allocate the ring for any field seen for the first time.
//   2. Copy this field's host staging into THIS FRAME's ring slot, but only
//      when the field's data serial moved since that slot last saw it — a
//      parked or static field costs nothing.
//   3. Rewrite the whole FieldDesc array. O(fields), never O(particles); that
//      is the entity's thesis expressed as an upload size.
//   4. Sweep fields that left the scene, through the frame-serial retire queue.
//
// Everything above runs AFTER this frame-in-flight's fence and BEFORE
// recording — the VUID-03047 zone that InstanceExpand.hpp:10-16 names — which
// is what makes writing this slot's buffers safe.
//
// ── RING DEPTH ──────────────────────────────────────────────────────────────
// kFramesInFlight + 1 host-visible position buffers per field, the tet ring
// pattern verbatim (TetSkinningPipeline::kPosSlots, commit 5584d2ab). The host
// memcpy for frame N must not land in a buffer frames N-1 or N-2 are still
// reading; with kFramesInFlight prior frames possibly live, slot N % 3 was last
// touched by frame N-3, which has provably completed. A single buffer instead
// gives torn positions mid-read and consecutive frames rendering the same
// physics state (the duplicate-then-skip judder 5584d2ab fixed for tets).
//
// Ownership::Interop (§1.4a) will NOT use this ring — one ExternalBuffer,
// single-instance across all slots, because that overlap is GPU-to-GPU and
// benign. The ring exists only because a HOST write races an in-flight read.

#ifndef THREEPP_VULKAN_PARTICLE_FIELD_PASS_HPP
#define THREEPP_VULKAN_PARTICLE_FIELD_PASS_HPP

#include "VulkanImplCommon.hpp"
#include "VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace threepp {
    class Object3D;
    class ParticleField;
}// namespace threepp

namespace threepp::vulkan {

    class VulkanContext;

    // One SSBO of these for all live fields, rewritten whole every frame.
    // Scalar layout; matches plans/particle-field.md §1.5 field for field.
    struct FieldDescGpu {
        float          world[16];  // the field's matrixWorld
        VkDeviceAddress posAddr;   // buffer_reference address of positions
        VkDeviceAddress prevPosAddr;// 0 until phase 1 allocates it
        VkDeviceAddress oriAddr;   // 0 = no orientation buffer
        VkDeviceAddress attrAddr;  // 0 = no attribute buffer
        VkDeviceAddress countAddr;
        std::uint32_t  capacity;
        // == instanceCustomIndex, i.e. this field's MeshEntry index. Load-
        // bearing: it preserves the renderer's oldest invariant, that
        // instanceCustomIndex IS the entry index.
        std::uint32_t  entryIndex;
        float          uniformRadius;
        std::uint32_t  wSemantic;
        std::uint32_t  reprMask;// bit0 mesh, bit1 billboard, bit2 density, bit3 blas
        std::uint32_t  classId; // semantic class (setObjectClassId), for label AOVs
    };
    static_assert(sizeof(FieldDescGpu) == 128, "FieldDescGpu layout drift (plan §1.5)");

    // Mirrors ParticleFieldCounts (§1.3). Device-visible, one per ring slot.
    struct FieldCountsGpu {
        std::uint32_t liveCount;
        std::uint32_t _pad[3];
    };
    static_assert(sizeof(FieldCountsGpu) == 16, "FieldCountsGpu layout drift");

    class ParticleFieldPass {

    public:
        // Ring depth. Tied to kFramesInFlight by the same static_assert the tet
        // ring carries — a deeper pipeline that forgets this ring is a race.
        static constexpr std::uint32_t kSlots = impl::kFramesInFlight + 1u;
        static_assert(kSlots == impl::kFramesInFlight + 1u,
                      "ParticleField ring depth must track kFramesInFlight");

        // One visible field and the entry index the scene expansion gave it.
        struct Rec {
            ParticleField* field;
            std::uint32_t  entryIndex;
            std::uint32_t  classId;
            // Vertices per proxy instance — the index count of MeshRepr's
            // geometry, resolved by the renderer from the same BlasRecord the
            // DrawInfo addresses come from. 0 = the field draws nothing this
            // frame (no MeshRepr, or its geometry has not uploaded yet).
            std::uint32_t  proxyVertexCount = 0;
        };

        // What the raster pass needs to issue one field's draw. Rebuilt every
        // prepareFrame, in the order the Rec list arrived, so the renderer can
        // zip it against its own per-field DrawInfo indices.
        struct DrawState {
            const ParticleField* field    = nullptr;
            VkBuffer        indirect      = VK_NULL_HANDLE;// ONE VkDrawIndirectCommand
            VkBuffer        counts        = VK_NULL_HANDLE;// src of the instanceCount copy
            VkDeviceAddress posAddr       = 0;
            VkDeviceAddress prevPosAddr   = 0;
            VkDeviceAddress oriAddr       = 0;// 0 → identity orientations
            std::uint32_t   vertexCount   = 0;// 0 → skip the draw
        };

        // Same contract and same reason as InstanceExpand::RetireBufferFn: a
        // field that leaves the scene may still be named by an in-flight frame,
        // so its buffers go to the renderer's frame-serial retire queue rather
        // than being destroyed inline.
        using RetireBufferFn = std::function<void(Buffer&&)>;

        ParticleFieldPass(VulkanContext& ctx, RetireBufferFn retireFn);
        ~ParticleFieldPass();
        ParticleFieldPass(const ParticleFieldPass&) = delete;
        ParticleFieldPass& operator=(const ParticleFieldPass&) = delete;

        // The whole per-frame job (steps 1-4 above). `serial` is the monotonic
        // frame serial being recorded; `frame` is the frame-in-flight index.
        void prepareFrame(std::uint64_t serial, std::uint32_t frame,
                          const std::vector<Rec>& fields);

        // Publish liveCount into each field's VkDrawIndirectCommand, on the
        // DEVICE: a 4-byte copy into byte offset 4 of the command, which is
        // where instanceCount lives (plan §1.3, route 1). The CPU writes the
        // record's other three words in prepareFrame and never learns the
        // count — under Ownership::Interop the counts block is written by the
        // sim's CUDA kernel and there is nothing to learn. Recorded at the head
        // of the frame's command buffer, before any consumer.
        void recordCounts(VkCommandBuffer cb);

        [[nodiscard]] const std::vector<DrawState>& drawStates() const { return draws_; }

        // The FieldDesc SSBO for a frame-in-flight, and how many entries of it
        // prepareFrame filled. Nothing reads these before phase 1; they are the
        // handles the phase-1 descriptor writes will name.
        [[nodiscard]] VkBuffer descBuffer(std::uint32_t frame) const {
            return descBufs_[frame].handle;
        }
        [[nodiscard]] std::uint32_t descCount() const { return descCount_; }

        // Fields with resident device state. Read by the renderer's own
        // early-out: a scene that never had a field must not pay for one, and a
        // scene that just lost its last field still has a sweep to run.
        [[nodiscard]] std::size_t liveFieldCount() const { return states_.size(); }

    private:
        struct State {
            // Non-owning; liveness is tracked through `owner` when the field was
            // created the documented way (ParticleField::create → make_shared).
            std::weak_ptr<Object3D> owner;
            bool          ownerTracked = false;
            std::uint32_t capacity     = 0;
            Buffer        positions[kSlots]{};
            Buffer        counts[kSlots]{};
            // One VkDrawIndirectCommand per slot. Per-slot rather than shared
            // because its instanceCount is written by a device copy inside the
            // frame that reads it, and two frames in flight must not share the
            // word one of them is still consuming.
            Buffer        indirect[kSlots]{};
            // Orientations, snorm16x4. SINGLE instance, not ringed: write-once
            // by contract (ParticleField::setOrientations documents why).
            Buffer        orientations{};
            std::uint64_t oriSerial = 0;// ParticleField::orientationSerial() uploaded
            // ParticleField::dataSerial() this slot was last filled from. 0 =
            // freshly allocated, i.e. holds garbage and must be re-sent.
            std::uint64_t slotSerial[kSlots]{};
            std::uint64_t lastSeenSerial = 0;
        };

        VulkanContext& ctx_;
        RetireBufferFn retireFn_;

        std::unordered_map<const ParticleField*, std::unique_ptr<State>> states_;
        Buffer        descBufs_[impl::kFramesInFlight]{};
        std::uint32_t descCapacity_ = 0;// in FieldDescGpu elements
        std::uint32_t descCount_    = 0;
        std::vector<FieldDescGpu> descScratch_;
        std::vector<DrawState>    draws_;

        State& ensureState(const ParticleField& field);
        void   ensureDescCapacity(std::uint32_t frame, std::uint32_t count);
        void   destroyState(State& st);
        void   retireOrDestroy(Buffer& b);
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_PARTICLE_FIELD_PASS_HPP
