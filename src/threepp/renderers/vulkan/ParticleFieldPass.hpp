// ParticleFieldPass — the device-side half of threepp::ParticleField.
//
// Phase 0 of plans/particle-field.md: this owns the per-field position buffer,
// the per-field live-count block and the per-frame FieldDesc SSBO. Everything
// later phases add — mesh expansion, billboards, the density scatter, the
// AABB BLAS — reads exactly these three things, so their lifetime and write
// window are established here once.
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
// Ownership::Interop keeps the plan's invariant where the plan states it — the
// EXPORTED buffer the foreign API writes is ONE buffer, single-instance across
// all slots, because that overlap is GPU-to-GPU and benign (5584d2ab) — but it
// does still use this ring, as the destination of a device-to-device SNAPSHOT
// taken at the head of every frame (see recordInteropSnapshot). The ring is
// device-local there, never host-mapped, and it buys two things a direct read
// of the exported buffer cannot:
//
//   • prevPositions. Motion vectors need the previous state, and the foreign
//     copy lands on the host timeline (it is synchronous, in prepareFrame),
//     so anything the renderer copies afterwards inside the same frame's
//     command buffer would capture the state that just arrived — every motion
//     vector exactly zero. Two consecutive snapshots taken by the renderer's
//     own queue are a genuinely consecutive pair.
//   • A stable frame. Consumers (G-buffer draw, density scatter, and whatever
//     a later BLAS phase adds) then read a buffer no foreign API is writing,
//     so the field cannot change under them mid-frame. The unsynchronised
//     overlap is confined to the one copy at the head of the frame.
//
// It costs one device-local copy of capacity*16 B per frame — bandwidth on the
// board, never over the bus, which is the entire point of the phase.
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

    // ── F4/F5: the emitter's per-field AUX RECORD ───────────────────────────
    // particle_emit.comp's push block is 128 B EXACTLY — the range every Vulkan
    // implementation guarantees — and F4 spent the last two floats of it on the
    // toroidal follow centre. F5 needs a height-map address plus a dozen
    // lifecycle scalars and there was not one byte left.
    //
    // So those two floats became ONE 64-bit address and everything behind it
    // moved here: the same move F3 made for BillboardParamsGpu and F4 for
    // BillboardViewGpu, with the same guarantee. A buffer_reference is not a
    // descriptor, so the emit pipeline STILL allocates no set and writes no set,
    // and still cannot update one a frame in flight names (VUID-03047).
    //
    // Written in prepareFrame — the post-fence, pre-record window — into a
    // per-frame-in-flight host-visible block, exactly like the billboard params.
    // A field that neither follows nor rests gets NO record and pushes a null
    // address, which is bit-identical to the two zero floats it replaced.
    //
    // MUST mirror the EmitAux block in shaders/particle_emit.comp.
    struct EmitAuxGpu {
        VkDeviceAddress heightAddr;   //  0  the bake; 0 = no surface interaction
        float           followX;      //  8  field-LOCAL centre of the wrap box
        float           followZ;      // 12
        float           bakeOriginX;  // 16  field-LOCAL min corner of the map
        float           bakeOriginZ;  // 20
        float           bakeInvCell;  // 24  texels per metre
        std::uint32_t   bakeRes;      // 28  texels per axis
        float           bakeMiss;     // 32  height off the map / where nothing hit
        float           landBias;     // 36
        float           restSeconds;  // 40
        float           restJitter;   // 44
        float           fadeSeconds;  // 48
        float           splashSeconds;// 52  > 0: land as a ring, not as a rest
        // The ring's radius in ABSOLUTE METRES at impact and at the end of the
        // splash. Absolute rather than a multiple of the particle's own radius,
        // and that is what makes the billboard's decode exact: with a relative
        // scale the value in w would depend on the slot's size hash, and the
        // vertex stage — which does not know it — could only recover the phase
        // to within the size jitter. splashR0 is set just above the largest
        // radius the emitter's jitter can produce, so w >= splashR0 can only
        // mean a splash. 0 = this field has none.
        float           splashR0;     // 56
        float           splashR1;     // 60
    };
    static_assert(sizeof(EmitAuxGpu) == 64,
                  "EmitAuxGpu drifted from the EmitAux block in particle_emit.comp");

    // ── F3: per-field billboard appearance (plans/particle-atmosphere.md F-D) ─
    // One record per visible billboard field, rewritten whole every frame into
    // a per-frame-in-flight host-visible block and reached by DEVICE ADDRESS.
    //
    // A buffer reference rather than a descriptor: this pass allocates no
    // descriptor set, writes no descriptor set, and therefore cannot update
    // one a frame in flight still names (VUID-03047) — the same argument as
    // particle_emit.comp's zero-set push block, applied to data that is too
    // big for the 128 B push range once the camera matrices are in it.
    // O(fields) bytes per frame, never O(particles).
    //
    // MUST mirror the BbParams buffer_reference block in
    // shaders/particlefield_billboard.vert member for member. The 64-bit
    // addresses sit first so the block's 8-byte alignment is satisfied at
    // offset 0 and every float after is naturally 4-aligned, keeping MSVC's
    // layout and GLSL's scalar layout byte-identical.
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
        // ── F5: the splash ring ─────────────────────────────────────────────
        // The emitter encodes a splash's phase in the RADIUS it writes to w —
        // it grows the particle, and nothing else in the lifecycle ever does —
        // so these two numbers are the decode. w > splashR0 means "this is a
        // ring", and (w - R0) / (R1 - R0) is how far through the splash it is.
        //
        // Absolute metres rather than a multiple of the particle's own radius:
        // the alternative is for the vertex stage to re-derive the per-slot
        // size hash, which would add a third shared contract to keep in sync
        // for no gain. R0 is computed host-side as just above the largest
        // radius the emitter's size jitter can produce.
        float splashR0;               // 120  0 = this field has no splash
        float splashR1;               // 124
        float splashRingWidth;        // 128  annulus width, fraction of radius
        // ── 4c: the sprite slice ────────────────────────────────────────────
        // Rides the last 12 bytes the record already reserved, so the sprite
        // mode costs the parameter block nothing.
        float opacity;                // 132  alphaOver coverage scale
        float litPhaseG;              // 136  HG asymmetry for the lit lobe
        float litAmbient;             // 140  ambient share of the lit radiance
        // ── VOLUMETRIC SPRITES (plans/particle-volumetric-sprites) ──────────
        // The address lands at 144, which is 8-aligned, so the block's scalar
        // layout and MSVC's stay byte-identical without a pad.
        VkDeviceAddress attrAddr;     // 144  Config::attributes; 0 = use the ramp
        // ROWS of the affine field->world matrix. The marches happen in WORLD
        // space (the density volume is world-anchored) while the position
        // buffer is field-local, and the vertex stage's only other basis is
        // view — so the one matrix it cannot reconstruct travels with the field
        // rather than being rebuilt per vertex.
        float model[12];              // 152
        float boxMin[3];              // 200  the density volume's world min
        float boxInvSize[3];          // 212  1 / (2 * halfExtent)
        float volumeExtinction;       // 224  exponent on T_cam; 0 = no march
        float volumeShadow;           // 228  mix toward the sun term; 0 = no march
        float volumeAmbient;          // 232
        float volumeSunGain;          // 236
        // R8: this field's slice of the frame's transmittance buffer, which
        // particlefield_transmit.comp fills once per particle per view. 240 is
        // 8-aligned, so this lands naturally and the block needs no pad.
        // 0 exactly when the field marches nothing, i.e. when bit 16 is clear.
        VkDeviceAddress transAddr;    // 240
        // Weight of the fragment falloff's fixed t^9 core term. 0.85 was the
        // hardcoded constant before it became a knob (BillboardRepr's default
        // keeps those bytes); big soft smoke parcels set it toward 0 so every
        // sprite stops planting a 1-2 px hard dot the skirt cannot soften.
        float coreWeight;             // 248
        float pad0;                   // 252  keeps sizeof at the 8-aligned 256
    };                                // 256
    static_assert(sizeof(BillboardParamsGpu) == 256,
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
        // ── 4c: the sun, for BillboardRepr::lit ─────────────────────────────
        // The lights UBO is a descriptor set this pass does not have (no set
        // means no VUID-03047 exposure), and the sun is
        // three floats. So the renderer snapshots the scene's brightest
        // DirectionalLight — the same one-sun the deferred path shades with —
        // into this record, per view per frame, and the vertex stage evaluates
        // one HG lobe against it. Zero radiance when the scene has no sun,
        // which makes a `lit` field fall back to pure ambient rather than go
        // black.
        float         sunDir[3];     // 48  world space, TOWARD the sun
        float         _pad0;         // 60
        float         sunRadiance[3];// 64  linear, colour x intensity
        float         _pad1;         // 76
        float         ambient[3];    // 80  the scene's summed AmbientLights
        float         _pad2;         // 92
        // ── The volumetric marches' two WORLD-space inputs ──────────────────
        // sunDir above is already in VIEW space (the lit lobe needs it there),
        // but the density volume is world-anchored, so the marches need the
        // world vector and the world eye point as well. Two vec3s rather than a
        // full inverse view: those are the only two quantities the marches ask
        // for, and the block stays a flat scalar struct.
        float         camWorld[3];   // 96  the eye, world space
        float         _pad3;         // 108
        float         sunDirWorld[3];// 112 unit, TOWARD the sun
        float         _pad4;         // 124
    };                               // 128
    static_assert(sizeof(BillboardViewGpu) == 128,
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
            // 4c: BillboardRepr::alphaOver — this field's quads composite
            // premultiplied SRC_ALPHA-over instead of ONE/ONE, which is a
            // PIPELINE property and therefore has to be visible to the
            // recorder, not just to the shader.
            bool            bbAlphaOver   = false;
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

        // What enableInterop hands back: the exported OS handle (a Win32 NT
        // handle owned by us, a POSIX fd owned by the importer) and the size to
        // import. This pass's own type rather than
        // VulkanRenderer::ParticleFieldInteropHandle — the pass does not
        // include the renderer's public header, and the Impl converts.
        struct InteropExport {
            void*       osHandle  = nullptr;
            std::size_t sizeBytes = 0;
            // Config::attributes: the SECOND exported allocation, same size and
            // same layout (one vec4 per slot), handed out by the same call. A
            // field without attributes reports a null handle here and nothing
            // else changes — but a field WITH them can never end up in a state
            // where the positions imported and the colours did not.
            void*       attrHandle  = nullptr;
            std::size_t attrSizeBytes = 0;
        };

        // ── F6: Ownership::Interop ──────────────────────────────────────────
        // Export this field's positions allocation and register the copy that
        // fills it. Once per field, from the application, OUTSIDE any frame:
        // it may allocate the field's device state, which is why it is not a
        // per-frame call.
        //
        // `deviceCopy` is invoked once per frame from prepareFrame — the
        // post-fence, pre-record window — and must be SYNCHRONOUS (the
        // soft-body path's copyTetToVulkan() is the template: cuMemcpyDtoDAsync
        // + cuStreamSynchronize). That contract is what orders the foreign
        // write against this frame's snapshot without an external semaphore:
        // the copy has completed by the time the callback returns, and the
        // command buffer that reads it has not been submitted yet.
        //
        // Returns {} when the device cannot export memory. The field has then
        // been put into ParticleField::hostFallback() and told the user why, so
        // the caller's pull-to-host leg can feed it unchanged.
        InteropExport enableInterop(ParticleField& field, std::function<void()> deviceCopy);

        // The whole per-frame job (steps 1-4 above). `serial` is the monotonic
        // frame serial being recorded; `frame` is the frame-in-flight index.
        void prepareFrame(std::uint64_t serial, std::uint32_t frame,
                          const std::vector<Rec>& fields);

        // ── F6: the head-of-frame device-to-device snapshot ─────────────────
        // One vkCmdCopyBuffer per Interop field, exported buffer → this frame's
        // ring slot, closed with a barrier that covers every consumer (vertex
        // pull, density scatter, transfer). MUST be recorded before all of
        // them, and before recordCounts for tidiness rather than correctness.
        // No-op — not one command — when no field is Interop-owned.
        void recordInteropSnapshot(VkCommandBuffer cb);

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

        // ── F5: the surface height bake ─────────────────────────────────────
        // Re-bake, for any field whose EmitterParams::Surface is on and whose
        // baked footprint no longer matches what the emitter is about to ask
        // for. Recorded IMMEDIATELY BEFORE recordEmit — the map is the emitter's
        // input — and closed with a barrier in both directions: this pass's
        // TLAS read must complete before recordDeformAndTlas's refit writes the
        // acceleration structure later in the same command buffer, and its
        // buffer write must complete before the emit dispatch reads it.
        //
        // No-op on the overwhelming majority of frames: a bake happens only when
        // the scene's structure changed, the follow centre snapped, the field
        // moved, or the footprint was reconfigured. Steady state records nothing.
        void recordSurfaceBake(VkCommandBuffer cb);

        // A bake will be recorded this frame. Same purpose as emitActive(): let
        // the renderer skip the timestamp bracket on the frames that do nothing.
        [[nodiscard]] bool surfaceBakeActive() const { return !bakeDispatch_.empty(); }

        // The scene TLAS every bake traces. Called every frame from
        // prepareParticleFields, in the post-fence window; the descriptor is
        // rewritten ONLY when the handle actually changes, which happens only on
        // a structural rebuild, which is itself vkDeviceWaitIdle-guarded — so
        // this pass never writes a set an in-flight frame could name.
        void setTlas(VkAccelerationStructureKHR tlas);

        // This frame's slot of the renderer's GeometryDesc ring, by device
        // address. Handed over beside the TLAS and for the same pass: the sun
        // occlusion needs one field of it (foamAddress) to tell a water hit,
        // which must be walked past, from a hull hit, which is the shadow.
        // Per FRAME, not per handle change — the ring rotates every frame.
        void setSceneGeomAddress(VkDeviceAddress addr) { sceneGeomAddr_ = addr; }

        // Throw every field's bake away. The renderer calls this when the entry
        // list is rebuilt, i.e. when the set of things a flake could land on may
        // have changed. Cheap: it bumps a counter, and the re-bake happens on
        // the next frame that needs one.
        void invalidateSurfaceBakes() { ++bakeStructGen_; }

        // Any Renderer-owned field will dispatch this frame. Lets the renderer
        // skip the timestamp bracket (so the timing measures only real work)
        // on the common frame that has no emitter.
        [[nodiscard]] bool emitActive() const { return !emitDispatch_.empty(); }

        // ── R8/R9: the per-view transmittance prepass ───────────────────────
        // One dispatch per marching field, writing (T_cam, T_sun) per SLOT into
        // this frame's buffer, closed with a compute-write → vertex-read
        // barrier. Recorded OUTSIDE any render-pass instance, immediately
        // before the billboard draws of the view whose eye `camWorld` is —
        // and re-recorded for the next view over the SAME buffer behind the
        // next barrier (R9). T_sun is view-independent and is recomputed with
        // it; a second buffer to avoid that would cost more memory than the
        // eight taps it saves.
        //
        // Views are recorded sequentially into one command buffer, so one
        // buffer per frame-in-flight is all the depth this needs. The glow leg
        // is the same camera as its display leg and deliberately does NOT
        // re-dispatch: it reads what the display leg's dispatch wrote.
        //
        // R10: no-op — not one command, and not one allocated byte — when no
        // visible field has a volumetric knob on.
        void recordTransmittance(VkCommandBuffer cb, const float camWorld[3],
                                 const float sunDirWorld[3]);

        // Any field will run the prepass this frame. Lets the renderer skip the
        // whole call (and its camera-inverse) on every scene that has no
        // volumetric field, which is nearly all of them.
        [[nodiscard]] bool transmittanceActive() const { return !transDispatch_.empty(); }

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

        // ── The per-volume MAJORANT (parent plan phase 3) ───────────────────
        // kMaxDensityFields uints in Q20.12 — max sigma_t over each bound
        // volume, in densityVolumes() order — reduced on the GPU by the
        // convert dispatch and zeroed at the head of every scatter block. The
        // LIDAR's delta tracking needs a bound it can trust per frame, and the
        // volume is the only thing that knows one; nothing here is authored.
        //
        // VK_NULL_HANDLE until the first field asks for a density volume, so
        // callers must handle its absence (there is nothing to bound then).
        [[nodiscard]] VkBuffer     densityMajorants() const { return densityMajorants_.handle; }
        [[nodiscard]] VkDeviceSize densityMajorantsSize() const { return densityMajorants_.size; }

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
        // ── F5: what a baked height map DESCRIBES ───────────────────────────
        // The whole re-bake trigger, expressed as a value. Every input the bake
        // depends on is in here, so "is this slot's map still the right map" is
        // one comparison and there is no list of events to keep in step with the
        // things that can invalidate one. The four that matter: the follow
        // centre SNAPPED (cx/cz), the field MOVED (worldX/Y/Z), the footprint
        // was reconfigured (extent/res/topY/depth), and the scene's structure
        // changed (structGen, bumped by invalidateSurfaceBakes).
        struct BakeKey {
            float         cx = 0.f, cz = 0.f;// field-local bake centre
            float         extent = 0.f;      // half-size, field-local metres
            std::uint32_t res    = 0;
            float         topY = 0.f, depth = 0.f;
            float         worldX = 0.f, worldY = 0.f, worldZ = 0.f;
            std::uint64_t structGen = 0;
            // Exact float comparison on purpose: every one of these is copied
            // from the same source each frame, so "unchanged" means bit-equal
            // and a tolerance would only let a real change through.
            bool operator==(const BakeKey& o) const {
                return cx == o.cx && cz == o.cz && extent == o.extent &&
                       res == o.res && topY == o.topY && depth == o.depth &&
                       worldX == o.worldX && worldY == o.worldY &&
                       worldZ == o.worldZ && structGen == o.structGen;
            }
            bool operator!=(const BakeKey& o) const { return !(*this == o); }
            [[nodiscard]] bool valid() const { return res != 0u; }
        };

        struct State {
            // Non-owning; liveness is tracked through `owner` when the field was
            // created the documented way (ParticleField::create → make_shared).
            std::weak_ptr<Object3D> owner;
            bool          ownerTracked = false;
            std::uint32_t capacity     = 0;
            bool          rendererOwned = false;
            // ── F6: Ownership::Interop ──────────────────────────────────────
            // The field asked for interop AND the device can export, so the
            // ring above is DEVICE-LOCAL and is filled by a copy from posExt
            // rather than by a host memcpy. False on an interop field that fell
            // back (no external memory), which is then a HostRing field in
            // every respect except the exception submit() no longer throws.
            bool                 interopOwned = false;
            // THE single instance the plan's §1.4(a) invariant is about: one
            // exported allocation for all frames in flight, written by the
            // foreign API, never ringed. Its usage carries no device address —
            // createExternalBuffer allocates without
            // VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, so a SHADER_DEVICE_ADDRESS
            // buffer on that memory would be invalid — which is a second reason
            // the shaders read the snapshot and not this.
            vulkan::ExternalBuffer posExt{};
            // Invoked once per frame in prepareFrame. Null until enableInterop.
            std::function<void()>  deviceCopy;
            // An interop field with no copy registered renders an empty scene
            // and there is nothing in the pixels to say why, so it says so
            // itself — ONCE, not once per frame.
            bool                 interopUnarmedLogged = false;
            // Has this slot ever been snapshotted? The serial gate below cannot
            // answer that for an interop field: dataSerial only moves when the
            // HOST changes the live count, while the device positions change
            // every frame with nobody on the host any the wiser.
            bool                 snapped[kSlots]{};
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
            // ── Config::attributes: the positions' path, verbatim ────────────
            // Under HostRing / Renderer this is ONE host-visible buffer written
            // write-once (setAttributes' contract, the orientations' rule).
            // Under Ownership::Interop it is the same kSlots RING the positions
            // use, device-local, filled by the head-of-frame snapshot from
            // attrExt — because attributes that arrive per frame have exactly
            // the write-during-read hazard positions do, and solving it twice
            // in two different ways is how the two get out of step.
            Buffer        attributes[kSlots]{};
            vulkan::ExternalBuffer attrExt{};
            std::uint64_t attrSerial = 0;// ParticleField::attributeSerial() uploaded
            // ParticleField::dataSerial() this slot was last filled from. 0 =
            // freshly allocated, i.e. holds garbage and must be re-sent.
            std::uint64_t slotSerial[kSlots]{};
            std::uint64_t lastSeenSerial = 0;
            // Which host UPLOAD filled each slot, counted in uploads and not in
            // frames. Config::hostStableSlots turns the previous ring slot into
            // a prevPositions buffer, and that is only true when the two slots
            // hold CONSECUTIVE submits: a frame the host skipped leaves its slot
            // holding a submit three frames old, and reading it as "one step
            // ago" would streak every particle over three steps and, once the
            // ring wraps past it, backwards. fillSeq == 0 means never filled.
            std::uint64_t slotFill[kSlots]{};
            std::uint64_t fillSeq = 0;
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
            // R8: the transmittance prepass's view of the SAME r16f mirror, as
            // a COMBINED_IMAGE_SAMPLER so the march gets the hardware trilinear
            // filtering the vertex stage's textureLod had. Allocated and
            // written ONCE, on the first frame this field actually marches —
            // the image handle never changes for the life of the field, so this
            // set is never rewritten and can never be a VUID-03047 in-flight
            // update. VK_NULL_HANDLE on a field that never marches, which is
            // half of R10's "not one allocated byte".
            VkDescriptorSet marchSet = VK_NULL_HANDLE;

            // ── F5: the baked height map ────────────────────────────────────
            // RINGED over the frames in flight, unlike the density volume,
            // because a bake is a WRITE and the previous frames' emit dispatches
            // are still READING. One buffer would be an unsynchronised
            // write-after-read across command buffers every time the follow
            // centre snapped; three are 768 KB at the default resolution and
            // remove the hazard rather than reasoning about it. Allocated on the
            // first frame the field asks for a surface, and re-allocated only if
            // the resolution changes.
            Buffer        heights[impl::kFramesInFlight]{};
            std::uint32_t heightRes = 0;// texels/axis the buffers were sized for
            // What each slot's contents describe. A slot whose key differs from
            // what this frame wants is re-baked before the emit reads it.
            BakeKey       bakedKey[impl::kFramesInFlight]{};
        };

        // One field's emit dispatch, resolved in prepareFrame so recording
        // touches no ParticleField and no map. `pc` is the shader's push block
        // verbatim — see EmitPc in the .cpp, which mirrors particle_emit.comp.
        struct EmitDispatch {
            std::uint32_t groups = 0;// ceil(capacity / 64)
            unsigned char pc[128]{}; // EmitPc, prebuilt
            // Index into auxScratch_, patched into pc's trailing address once
            // the aux block has been (re)allocated — growing it mid-loop would
            // invalidate every address already handed out. 0xffffffff = this
            // field needs no aux record and pushes a null address.
            std::uint32_t auxIndex = 0xffffffffu;
        };

        // F5: one field's height bake, resolved in prepareFrame so recording
        // touches no ParticleField and no map — the same shape as the two above.
        struct BakeDispatch {
            std::uint32_t groups = 0;// ceil(res / 8) per axis
            unsigned char pc[64]{};  // BakePc, prebuilt
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
            VkDeviceAddress attrAddr = 0;// Config::attributes; .a = medium weight, 0 = none
            float           boxMin[3]{};
            float           boxInvSize[3]{};
            std::uint32_t   capacity   = 0;
            float           sigmaFixed = 0.f;
            // This volume's slot in ParticleDensityUbo / densityVolumes(), and
            // therefore its slot in the majorant buffer. Pushed to the convert
            // dispatch, which reduces the volume's maximum into it.
            std::uint32_t   volIndex   = 0;
        };

        // R8: one field's transmittance dispatch, resolved in prepareFrame so
        // recording touches no ParticleField and no map — the same shape as
        // EmitDispatch / BakeDispatch / DensityDispatch above. Everything here
        // is view-INDEPENDENT; the two per-view vectors arrive as arguments to
        // recordTransmittance and are patched into the push block there, which
        // is what makes re-dispatching for the next view free of any host work.
        struct TransDispatch {
            VkDescriptorSet set     = VK_NULL_HANDLE;// the r16f mirror, sampled
            std::uint32_t   groups  = 0;             // ceil(capacity / 64)
            unsigned char   pc[120]{};               // TransmitPc, prebuilt
            // The sun-occlusion follow-up, when the field asked for it and the
            // device can trace. It rides in the SAME record because it covers
            // exactly the same slots with the same groups and the same field
            // matrix — a second parallel vector would only be a way for the two
            // to disagree about which field they mean.
            bool            occlude = false;
            unsigned char   opc[88]{};               // OccludePc, prebuilt
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

        // F4/F5: the emitter's aux records, one host-visible device-addressable
        // block per frame-in-flight. Same lifetime, same window and same growth
        // rule as bbParamBufs_ above — filled in prepareFrame, never during
        // recording, so the slot being written is provably not one an in-flight
        // frame reads.
        Buffer        auxBufs_[impl::kFramesInFlight]{};
        std::uint32_t auxCapacity_ = 0;// in EmitAuxGpu elements
        std::vector<EmitAuxGpu> auxScratch_;

        // ── F5: the height bake ─────────────────────────────────────────────
        // The ONE pipeline in this pass with a descriptor set, because an
        // acceleration structure cannot ride a device address. A single set, not
        // one per frame in flight: its only binding is the TLAS, whose handle
        // changes only on a structural rebuild, which is vkDeviceWaitIdle
        // guarded — so it is never rewritten under an in-flight frame.
        VkDescriptorSetLayout tlasDsLayout_   = VK_NULL_HANDLE;
        VkDescriptorPool      tlasPool_       = VK_NULL_HANDLE;
        VkDescriptorSet       tlasSet_        = VK_NULL_HANDLE;
        VkPipelineLayout      bakePipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            bakePipe_       = VK_NULL_HANDLE;
        VkAccelerationStructureKHR tlasBound_  = VK_NULL_HANDLE;// what the set holds
        VkAccelerationStructureKHR wantTlas_  = VK_NULL_HANDLE;// what the scene has
        std::uint64_t         bakeStructGen_  = 0;
        std::vector<BakeDispatch> bakeDispatch_;
        // Reported once rather than every frame: a device with no ray query
        // cannot bake, so surface interaction quietly does nothing there and
        // says so exactly one time.
        bool                  bakeUnsupportedLogged_ = false;

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
        // kMaxDensityFields Q20.12 majorants, device-local, written by the
        // convert dispatch and read by the LIDAR pass. Allocated alongside the
        // convert pipeline (a dust-free scene allocates neither) and never
        // resized, so the per-field convert sets that name it are written once.
        Buffer                densityMajorants_{};

        // ── R8: the transmittance prepass ───────────────────────────────────
        // Created lazily, on the first field that actually marches, so a scene
        // with dust but no volumetric sprites compiles nothing and allocates
        // nothing. Its own sampler because the pass owns no other one, and a
        // linear-clamp 3D sampler is what makes the compute march read the same
        // filtered values the vertex stage's textureLod did.
        VkSampler             transSampler_    = VK_NULL_HANDLE;
        VkDescriptorSetLayout transDsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      transPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            transPipe_       = VK_NULL_HANDLE;
        // (T_cam, T_sun) as one packHalf2x16 uint per SLOT, for every marching
        // field this frame laid end to end. One buffer per frame-in-flight
        // (R9): views re-dispatch over the same one behind their own barriers,
        // but the PREVIOUS frame's vertex reads must not see this frame's
        // writes — the same per-FIF discipline every other per-frame resource
        // here follows. Device-local: nothing on the host ever reads it.
        Buffer        transBufs_[impl::kFramesInFlight]{};
        std::uint32_t transCapacity_ = 0;// in uint elements, i.e. in slots
        std::vector<TransDispatch> transDispatch_;

        // The sun-occlusion pass: one ray query per particle against the scene
        // TLAS, folded into the T_sun half of the word the prepass wrote. NO
        // descriptor of its own — it binds the shared tlasSet_, and its three
        // buffers (positions, transmittance, the renderer's GeometryDesc array)
        // are device addresses in an 88 B push block. Created only for a field
        // that set BillboardRepr::sunGeometryShadow, which defaults off.
        VkPipelineLayout occludePipeLayout_ = VK_NULL_HANDLE;
        VkPipeline       occludePipe_       = VK_NULL_HANDLE;
        // The renderer's per-entry GeometryDesc array for THIS frame's slot,
        // handed over beside the TLAS. Only foamAddress is read, and only to
        // tell water (which is opaque-masked in this TLAS, so it would shadow
        // every submerged parcel) from geometry that really does occlude.
        VkDeviceAddress  sceneGeomAddr_     = 0;

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

        // F6: the same rule for a swept field's EXPORTED allocation, which the
        // renderer's retire queue cannot take (it holds Buffers, and this one is
        // outside VMA). Destroying it inline would free memory an in-flight
        // frame's snapshot copy still reads — and, on Windows, close a handle
        // under the importer.
        struct RetiredExternal {
            vulkan::ExternalBuffer buf;
            std::uint64_t          serial;
        };
        std::vector<RetiredExternal> extRetire_;

        // F6: this frame's snapshot copies, resolved in prepareFrame so
        // recording touches no ParticleField and no map — the same shape as
        // EmitDispatch / BakeDispatch / DensityDispatch above.
        struct InteropCopy {
            VkBuffer     src   = VK_NULL_HANDLE;// the exported allocation
            VkBuffer     dst   = VK_NULL_HANDLE;// this frame's ring slot
            VkDeviceSize bytes = 0;
        };
        std::vector<InteropCopy> interopCopies_;

        std::vector<DensityVolumeDesc> densityVols_;
        std::vector<DensityDispatch>   densityDispatch_;
        std::uint64_t densityGen_      = 0;
        std::uint32_t densityOverflow_ = 0;

        State& ensureState(const ParticleField& field);
        void   ensureDescCapacity(std::uint32_t frame, std::uint32_t count);
        void   ensureBbParamCapacity(std::uint32_t frame, std::uint32_t count);
        void   ensureAuxCapacity(std::uint32_t frame, std::uint32_t count);
        void   ensureBakePipeline();
        // The scene TLAS set both tracing pipelines bind. Creates the objects on
        // first use and refreshes the descriptor when the handle moved; false
        // when this device has no ray query or the scene has no TLAS yet, which
        // is what every caller checks before recording a traversal.
        bool   ensureTlasSet();
        void   destroyState(State& st);
        void   retireOrDestroy(Buffer& b);
        void   retireOrDestroy(Image2D& img);
        void   ensureEmitPipeline();
        void   ensureDensityPipeline();
        // R8. False when the pipeline could not be created, which leaves the
        // field's kBbVolume bit clear and its sprites flat rather than wrong.
        bool   ensureTransmittancePipeline();
        // The geometry half of T_sun. False when this device has no ray query or
        // the pipeline could not be created, which leaves the field's sun term
        // exactly the volume-only one it was before the flag existed.
        bool   ensureSunOccludePipeline();
        void   ensureTransCapacity(std::uint32_t count);
        // Allocates the field's volume on first use; false when the field's
        // DensityRepr is off or the volume could not be created.
        bool   ensureDensityVolume(State& st, const ParticleField& field);
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_PARTICLE_FIELD_PASS_HPP
