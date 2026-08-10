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

    // ── PHASE 2: the world-space density volume (plan §3.3) ─────────────────
    // Volumes bound to the froxel/shade descriptor set at once. KEEP IN SYNC
    // with kMaxDensityFields in shaders/particle_density.glsl. Fields past this
    // many keep every other representation and simply contribute no density.
    inline constexpr std::uint32_t kMaxDensityFields = 4;

    // Fixed-point scale for the r32ui accumulator: 12 fractional bits.
    // KEEP IN SYNC with kParticleDensityScale in shaders/particle_density.glsl,
    // where the choice (quantum 1/4096 /m, saturation 2^32/4096 = 1.05e6 /m)
    // is argued against plan R4.
    inline constexpr float kDensityFixedScale = 4096.f;

    // What the renderer needs to bind one field's volume into the deferred set
    // and tell the shader where it lives. Mirrors ParticleDensityUbo in
    // shaders/particle_density.glsl one for one.
    struct DensityVolumeDesc {
        VkImageView view = VK_NULL_HANDLE;
        // The r16f mirror (particle_density_convert.comp) — binding 69, the
        // volume the deferred shade's per-pixel dust march samples with
        // hardware trilinear. Same lifetime as `view`.
        VkImageView linView = VK_NULL_HANDLE;
        float       boxMin[3]{};    // world min corner
        float       resolution = 0.f;// voxels/axis, as the UBO's boxMin.w
        float       boxInvSize[3]{};// 1 / (2 * halfExtent)
        // ── PER-FIELD medium params (plans/particle-atmosphere.md F-A) ──────
        // These used to be one shared value taken from whichever field was
        // enumerated first. They are per volume now because a fire field and a
        // smoke field are the same scene, and one albedo cannot be both.
        float albedo[3]{1.f, 1.f, 1.f};// DensityRepr::albedo
        float anisotropy = 0.f;        // DensityRepr::anisotropy (HG g)
        // DensityRepr's emission, packed as the shader reads it:
        // x = intensity (0 = pure dust), y = bottom K, z = top K, w = exponent.
        float emission[4]{0.f, 1900.f, 800.f, 1.6f};
    };

    // Binding 68 of the deferred set. MUST match ParticleDensityUbo in
    // shaders/particle_density.glsl, which is std140 (NOT scalar) because it is
    // pulled into shaders that do not all enable GL_EXT_scalar_block_layout —
    // everything here is a vec4/uvec4, so the two layouts coincide anyway.
    // (160 -> 272 B when albedoAniso went per field and emission was added;
    // still one UBO, still one upload, and every member is still a vec4, which
    // is the invariant that keeps std140 and this C mirror the same bytes.)
    struct ParticleDensityUboGpu {
        float         boxMin[kMaxDensityFields][4];    // xyz = world min, w = resolution
        float         boxInvSize[kMaxDensityFields][4];// xyz = 1 / (2 * halfExtent)
        float         albedoAniso[kMaxDensityFields][4];// rgb = albedo, a = HG g
        float         emission[kMaxDensityFields][4];  // x = intensity, yzw = ramp
        // x = active volumes, y = 1 when any of them is emissive
        std::uint32_t counts[4];
    };
    static_assert(sizeof(ParticleDensityUboGpu) == 272, "ParticleDensityUbo layout drift");

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
        // Same contract for the density volume, which is an IMAGE and is named
        // by descriptor SETS as well as by command buffers — so it goes through
        // the frame-serial queue too, and the sets that name it are refreshed
        // through the renderer's per-frame dirty flags (see densityGeneration).
        using RetireImageFn = std::function<void(Image2D&&)>;
        // 3D-image factory. The renderer already owns one (Impl::createImage3D,
        // the froxel grids' own constructor); handing it in keeps a second copy
        // of the VkImageCreateInfo/VkImageViewCreateInfo pair out of the tree.
        using CreateImage3DFn = std::function<Image2D(std::uint32_t, std::uint32_t,
                                                     std::uint32_t, VkFormat,
                                                     VkImageUsageFlags, const char*)>;

        ParticleFieldPass(VulkanContext& ctx, RetireBufferFn retireFn,
                          RetireImageFn retireImageFn, CreateImage3DFn createImage3DFn);
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

        // ── PHASE 2 ─────────────────────────────────────────────────────────
        // Zero the density volumes and splat this frame's live particles into
        // them. ONE dispatch per field for ALL views (plan R9): the volume is
        // world-anchored precisely so K cameras share this work, so it is
        // recorded at the head of the frame's command buffer, alongside
        // recordCounts, and never inside a per-view block. No-op when no field
        // has a density representation — not one command is written.
        void recordDensityScatter(VkCommandBuffer cb);

        // This frame's bound volumes, in descriptor-array order. Never longer
        // than kMaxDensityFields.
        [[nodiscard]] const std::vector<DensityVolumeDesc>& densityVolumes() const {
            return densityVols_;
        }
        // Bumped whenever the bound volume LIST changes (a volume allocated,
        // retired, or reordered). The renderer compares it against what its
        // descriptor sets were last written with; equal means the sets are
        // already correct and no descriptor write is needed, which is what
        // keeps a steady-state dust scene out of the VUID-03047 zone entirely.
        [[nodiscard]] std::uint64_t densityGeneration() const { return densityGen_; }

        // Any field contributed density this frame. Drives heteroActive, the
        // froxel-pass gate and the shade's flags bit 11 — the "real, small,
        // easy-to-miss" wiring plan §3.3 calls out.
        [[nodiscard]] bool densityActive() const { return !densityVols_.empty(); }

        // Fields whose DensityRepr is on but which did not fit in
        // kMaxDensityFields this frame. Reported, not silently dropped.
        [[nodiscard]] std::uint32_t densityOverflowCount() const { return densityOverflow_; }

        [[nodiscard]] const std::vector<DrawState>& drawStates() const { return draws_; }

        // The FieldDesc SSBO for a frame-in-flight, and how many entries of it
        // prepareFrame filled. Nothing reads these before phase 1; they are the
        // handles the phase-1 descriptor writes will name.
        [[nodiscard]] VkBuffer descBuffer(std::uint32_t frame) const {
            return descBufs_[frame].handle;
        }
        [[nodiscard]] std::uint32_t descCount() const { return descCount_; }

        // TEST/DEBUG: one field's density volume, for readback. False when the
        // field has none (representation off, or never rendered). See
        // VulkanRenderer::readParticleDensityVolume for why this exists.
        [[nodiscard]] bool densityVolumeFor(const ParticleField& field,
                                            VkImage& image, std::uint32_t& res) const;

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
            // ── Density volume (phase 2) ────────────────────────────────────
            // Allocated at the FIRST frame DensityRepr::enabled is seen and
            // never resized — the same fixed-size contract as the position
            // ring, and for the same reason: it is named by descriptor sets,
            // so a resize is a structural change, not a reallocation.
            // DensityRepr::center/halfExtent may still move every frame; they
            // are a transform, not a size.
            Image2D         density{};
            std::uint32_t   densityRes = 0;// latched with the image
            VkDescriptorSet densitySet = VK_NULL_HANDLE;
            // The r16f mirror + the convert dispatch's set (src uint volume,
            // dst float volume). Created and written alongside densitySet,
            // once, outside any frame's record.
            Image2D         densityLin{};
            VkDescriptorSet convertSet = VK_NULL_HANDLE;
        };

        // One field's scatter dispatch, resolved in prepareFrame so recording
        // touches no ParticleField and no map.
        struct DensityDispatch {
            VkDescriptorSet set   = VK_NULL_HANDLE;
            VkImage         image = VK_NULL_HANDLE;
            VkImage         linImage   = VK_NULL_HANDLE;
            VkDescriptorSet convertSet = VK_NULL_HANDLE;
            std::uint32_t   res   = 0;
            std::uint32_t   groups = 0;// ceil(capacity / 64)
            float           world[16]{};
            VkDeviceAddress posAddr   = 0;
            VkDeviceAddress countAddr = 0;
            float           boxMin[3]{};
            float           boxInvSize[3]{};
            std::uint32_t   capacity   = 0;
            float           sigmaFixed = 0.f;
        };

        VulkanContext&  ctx_;
        RetireBufferFn  retireFn_;
        RetireImageFn   retireImageFn_;
        CreateImage3DFn createImage3DFn_;

        std::unordered_map<const ParticleField*, std::unique_ptr<State>> states_;
        Buffer        descBufs_[impl::kFramesInFlight]{};
        std::uint32_t descCapacity_ = 0;// in FieldDescGpu elements
        std::uint32_t descCount_    = 0;
        std::vector<FieldDescGpu> descScratch_;
        std::vector<DrawState>    draws_;

        // Density scatter pipeline — created lazily, on the first field that
        // asks for a volume, so a scene without dust allocates nothing.
        VkDescriptorSetLayout densityDsLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      densityPool_     = VK_NULL_HANDLE;
        VkPipelineLayout      densityPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            densityPipe_     = VK_NULL_HANDLE;
        // r32ui → r16f convert (particle_density_convert.comp), one dispatch
        // per field after its scatter. Shares densityPool_.
        VkDescriptorSetLayout convertDsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      convertPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            convertPipe_       = VK_NULL_HANDLE;

        // A destroyed field's descriptor set, held until no in-flight frame can
        // still name it. Same rule as VulkanRetireQueue (serial +
        // kFramesInFlight <= current), kept local because the renderer's queue
        // takes resources, not sets. Freeing one inline would be a
        // VUID-vkFreeDescriptorSets-pDescriptorSets-00309 the moment a field is
        // dropped while a frame that drew it is still executing.
        struct RetiredSet {
            VkDescriptorSet set;
            std::uint64_t   serial;
        };
        std::vector<RetiredSet> densitySetRetire_;

        std::vector<DensityVolumeDesc> densityVols_;
        std::vector<DensityDispatch>   densityDispatch_;
        std::uint64_t densityGen_      = 0;
        std::uint32_t densityOverflow_ = 0;

        State& ensureState(const ParticleField& field);
        void   ensureDescCapacity(std::uint32_t frame, std::uint32_t count);
        void   destroyState(State& st);
        void   retireOrDestroy(Buffer& b);
        void   retireOrDestroy(Image2D& img);
        void   ensureDensityPipeline();
        // Allocates the field's volume on first use; false when the field's
        // DensityRepr is off or the volume could not be created.
        bool   ensureDensityVolume(State& st, const ParticleField& field);
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_PARTICLE_FIELD_PASS_HPP
