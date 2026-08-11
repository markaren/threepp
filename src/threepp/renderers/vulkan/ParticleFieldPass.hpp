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
//
// Ownership::Renderer does not use it either, for exactly the same reason: its
// writer is particle_emit.comp, recorded into the same command buffer as every
// consumer and ordered against them by a barrier. See ensureState.

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

    // ── F3: per-field billboard appearance (plans/particle-atmosphere.md F-D) ─
    // One record per visible billboard field, rewritten whole every frame into
    // a per-frame-in-flight host-visible block and reached by DEVICE ADDRESS.
    //
    // A buffer reference rather than a descriptor, and that is the point: this
    // pass allocates no descriptor set, writes no descriptor set, and therefore
    // cannot update one a frame in flight still names (VUID-03047) — the same
    // argument particle_emit.comp's zero-set push block makes, applied to data
    // that is too big for the 128 B push range once the camera matrices are in
    // it. O(fields) bytes per frame, never O(particles).
    //
    // MUST mirror the BbParams buffer_reference block in
    // shaders/particlefield_billboard.vert member for member. The 64-bit
    // addresses sit first so the block's 8-byte alignment is satisfied at
    // offset 0 and every float after is naturally 4-aligned, making MSVC's
    // layout and GLSL's scalar layout the same bytes by construction.
    struct BillboardParamsGpu {
        VkDeviceAddress posAddr;      //   0
        VkDeviceAddress prevPosAddr;  //   8
        float colorHot[3];            //  16
        float sizeScale;              //  28
        float colorCool[3];           //  32
        float uniformRadius;          //  44
        float stretchOverDt;          //  48
        float stretchMax;             //  52
        float intensity;              //  56
        float softness;               //  60
        float fadePower;              //  64
        float brightJitter;           //  68
        float sizeTaper;              //  72
        float lifetime;               //  76  0 = no age is knowable
        float lifetimeJitter;         //  80
        float duty;                   //  84
        float time;                   //  88
        std::uint32_t seed;           //  92
        std::uint32_t flags;          //  96  bit0: w IS the radius
        // ── F4 ──────────────────────────────────────────────────────────────
        float glow;                   // 100  > 0: this field feeds the glow pyramid
        float stretchMaxScreen;       // 104  streak cap as a fraction of frame height
        float nearFade;               // 108
        float lodNear;                // 112  collapse the quad CLOSER than this
        float lodFade;                // 116
        std::uint32_t _pad[2];        // 120
    };
    static_assert(sizeof(BillboardParamsGpu) == 128,
                  "BillboardParamsGpu drifted from particlefield_billboard.vert");

    // ── F4: the per-VIEW billboard record ───────────────────────────────────
    // Everything a field billboard needs that belongs to the CAMERA and the
    // FRAME rather than to the field: the display transform, and the fog medium
    // the quads are now attenuated by.
    //
    // A second buffer_reference block rather than more push constants, for one
    // arithmetic reason — the push block was already exactly 128 B, the range
    // every implementation guarantees, and the fog closed form needs a dozen
    // floats. Not a descriptor, for the reason nothing in this pass is one: a
    // set written per view per frame is exactly the VUID-03047 exposure the
    // design avoids. `exposure` and `toneMapMode` MOVED here out of the push
    // block, which is what freed the 8 bytes the address rides in.
    //
    // One record per (view, output mode) per frame, written by the renderer
    // during recording into a per-frame-in-flight host-visible block. Safe in
    // that window for the same reason prepareFrame's writes are: this slot's
    // fence was waited on before recording began, so no in-flight frame can be
    // reading it.
    //
    // MUST mirror the BbView block in shaders/particlefield_billboard.vert.
    struct BillboardViewGpu {
        float         exposure;      //  0
        std::uint32_t toneMapMode;   //  4
        std::uint32_t flags;         //  8  bit0 fog active, bit1 LINEAR HDR out
        float         hfDensity;     // 12
        float         hfBaseY;       // 16
        float         hfFalloff;     // 20
        float         murkDensity;   // 24
        float         waterSurfaceY; // 28
        float         camWorldY;     // 32
        float         viewToWorldY[3];// 36
    };                               // 48
    static_assert(sizeof(BillboardViewGpu) == 48,
                  "BillboardViewGpu drifted from particlefield_billboard.vert");
    // Flag bits, mirrored in the shader.
    inline constexpr std::uint32_t kBbViewFogActive = 1u;
    inline constexpr std::uint32_t kBbViewLinearOut = 2u;

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
            // ── F3 billboards ───────────────────────────────────────────────
            // A SECOND indirect record, because the two representations are
            // independent: a field can draw a mesh proxy, a quad, both or
            // neither, and the mesh record's vertexCount is the proxy's index
            // count while the billboard's is always 4. Both get the same 4-byte
            // device copy of liveCount into instanceCount.
            VkBuffer        bbIndirect    = VK_NULL_HANDLE;// VkDrawIndirectCommand{4, live}
            // Device address of this field's BillboardParamsGpu record. The
            // billboard vertex stage reaches it by buffer_reference, so the
            // whole pass has no descriptor of its own beyond the (optional)
            // sprite texture.
            VkDeviceAddress bbParamsAddr  = 0;
            bool            billboard     = false;// draw the quad this frame
            // F4: BillboardRepr::glow > 0 — this field is also drawn into the
            // offscreen glow target and feeds the billboard bloom pyramid. The
            // gate is per FIELD so weather (rain, snow) pays literally nothing:
            // with no glow field in the scene the target is never allocated and
            // the whole chain is never recorded.
            bool            glow          = false;
            float           glowThreshold = 0.f;// bright-pass knee, linear HDR
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

        // ── F2: the device emitter (Ownership::Renderer) ────────────────────
        // ONE dispatch per Renderer field, for ALL views — the positions are
        // field-local and view-independent, the same world-anchored argument the
        // density scatter makes (plan R9). Recorded at the HEAD of the frame's
        // command buffer, before recordCounts / recordDensityScatter and before
        // any view's raster pass, and closed with a barrier that covers every
        // consumer: the density scatter (compute), the G-buffer draw (vertex)
        // and, when a BLAS ever exists, the acceleration-structure build.
        //
        // No-op when no field is Renderer-owned — not one command is written.
        void recordEmit(VkCommandBuffer cb);

        // Any Renderer-owned field will dispatch this frame. Lets the renderer
        // skip the timestamp bracket (and therefore keep the timing honest) on
        // the overwhelmingly common frame that has no emitter.
        [[nodiscard]] bool emitActive() const { return !emitDispatch_.empty(); }

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

        // Any visible field asked for the glow chain this frame. Lets the
        // renderer skip allocating the offscreen target — and recording the
        // pass, and the composite draw — for every scene that does not have a
        // spark in it, which is nearly all of them.
        [[nodiscard]] bool glowActive() const { return glowActive_; }
        // The largest bright-pass knee any glow field asked for. One pyramid
        // serves every glow field in the scene (they all composite additively
        // into the same target, so separating them would buy nothing but N
        // chains), and the threshold is therefore a scene-level number.
        [[nodiscard]] float glowThreshold() const { return glowThreshold_; }

        // ── F4: publish one per-VIEW billboard record and get its address ────
        // Called during recording, once per (view, output mode). Returns 0 when
        // the block could not be allocated, which the caller must treat as "do
        // not draw" — a null buffer_reference dereference is undefined, not a
        // no-op. Index resets to 0 in prepareFrame.
        VkDeviceAddress pushViewRecord(std::uint32_t frame, const BillboardViewGpu& rec);

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
            bool          rendererOwned = false;
            Buffer        positions[kSlots]{};
            // ── Ownership::Renderer: ONE device-local buffer, plus its
            // prevPositions sibling. NOT the kSlots ring.
            //
            // The ring above exists for exactly one reason (ParticleFieldPass
            // header, "RING DEPTH"): a HOST memcpy for frame N must not land in
            // a buffer frames N-1 / N-2 are still reading. A Renderer field has
            // no host write at all. Its writer is particle_emit.comp, recorded
            // into the SAME command buffer as every consumer and separated from
            // them by a barrier, so the write and the reads are ordered by the
            // GPU's own dependency graph rather than by three frames of
            // latency. The overlap that remains is GPU-to-GPU and benign — the
            // identical argument commit 5584d2ab records for the CUDA interop
            // buffer ("the interop buffer stays single-instance across all
            // slots"). Three copies would cost 3x the VRAM of a 1M-particle
            // field (48 MB) to solve a race that does not exist.
            //
            // prevPositions is non-negotiable for a raster-drawn field (plan
            // §1.2) and is written by the SAME dispatch and the SAME thread as
            // positions, from f(t - dt) — so there is no copy to order and no
            // way for the two to describe different frames.
            Buffer        devPositions{};
            Buffer        devPrevPositions{};
            Buffer        counts[kSlots]{};
            // One VkDrawIndirectCommand per slot. Per-slot rather than shared
            // because its instanceCount is written by a device copy inside the
            // frame that reads it, and two frames in flight must not share the
            // word one of them is still consuming.
            Buffer        indirect[kSlots]{};
            // The billboard representation's own draw record — see
            // DrawState::bbIndirect for why it is a second buffer rather than a
            // second use of the one above. Allocated LAZILY, on the first frame
            // the field's BillboardRepr is on, so a field that never draws
            // quads costs nothing for the ability to.
            Buffer        bbIndirect[kSlots]{};
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

        // One field's emit dispatch, resolved in prepareFrame so recording
        // touches no ParticleField and no map. `pc` is the shader's push block
        // verbatim — see EmitPc in the .cpp, which mirrors particle_emit.comp.
        struct EmitDispatch {
            std::uint32_t groups = 0;// ceil(capacity / 64)
            unsigned char pc[128]{}; // EmitPc, prebuilt
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

        // F3 billboards: one host-visible, device-addressable block per
        // frame-in-flight holding this frame's BillboardParamsGpu records.
        // Written in prepareFrame — the post-fence, pre-record window — so the
        // slot being filled is provably not the one an in-flight frame reads.
        // Grown, never shrunk; a scene's field count does not oscillate.
        Buffer        bbParamBufs_[impl::kFramesInFlight]{};
        std::uint32_t bbParamCapacity_ = 0;// in BillboardParamsGpu elements
        std::vector<BillboardParamsGpu> bbParamScratch_;

        // F4: the per-view records. Unlike the params block this one is filled
        // DURING recording (the camera of a secondary view is not known any
        // earlier), so it is written through a persistently mapped pointer and
        // grown only between frames — a growth mid-recording would invalidate
        // addresses already pushed into the command buffer, so the block is
        // sized generously and a request past the end returns 0 rather than
        // reallocating. kBbViewSlots covers the primary's display + glow draws
        // plus a dozen secondary views, which is more than any sensor rig in
        // the tree.
        static constexpr std::uint32_t kBbViewSlots = 32;
        Buffer        bbViewBufs_[impl::kFramesInFlight]{};
        std::uint32_t bbViewNext_ = 0;
        bool          glowActive_    = false;
        float         glowThreshold_ = 0.f;

        // Device emitter (F-C). Created lazily, on the first Renderer-owned
        // field, so a scene of HostRing fields allocates nothing. NO descriptor
        // set layout and NO pool: the two buffer addresses and the whole
        // parameter block ride in a 128 B push constant, which is what keeps
        // this pass entirely out of the VUID-03047 zone.
        VkPipelineLayout emitPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline       emitPipe_       = VK_NULL_HANDLE;
        std::vector<EmitDispatch> emitDispatch_;

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
        void   ensureBbParamCapacity(std::uint32_t frame, std::uint32_t count);
        void   destroyState(State& st);
        void   retireOrDestroy(Buffer& b);
        void   retireOrDestroy(Image2D& img);
        void   ensureEmitPipeline();
        void   ensureDensityPipeline();
        // Allocates the field's volume on first use; false when the field's
        // DensityRepr is off or the volume could not be created.
        bool   ensureDensityVolume(State& st, const ParticleField& field);
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_PARTICLE_FIELD_PASS_HPP
