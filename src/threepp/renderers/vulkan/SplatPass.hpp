// SplatPass — 3D Gaussian Splats on the Vulkan deferred renderer.
//
// A COMPUTE tile rasterizer, not a raster pass. sceneHdr carries no
// COLOR_ATTACHMENT usage (BloomPass owns it as a STORAGE|SAMPLED image), and
// the reference 3DGS design is a software tile rasterizer anyway — so the
// splats are composited by a compute dispatch that read-modify-writes sceneHdr.
//
// WHERE IT SITS. Between the deferred shade and the depth of field
// (VulkanCoreRecord::recordSplats, called from recordCommandBuffer between
// recordSceneDispatch and recordDepthOfField). That is LINEAR HDR, PRE-post:
// splats get DoF, bloom, tone mapping and TAA for free, and they get them
// through the same code path the rest of the scene does. The post-TAA overlay
// path — where the world Sprites and particles composite — is display-referred
// LDR; putting splats there would mean re-deriving every one of those effects.
//
// THE FIVE THINGS THAT MAKE IT LOOK RIGHT RATHER THAN NEARLY RIGHT
//
//  1. PRE-EXPOSURE. The shade/resolve multiplies its sceneHdr stores by the
//     frame's pre-exposure factor (VulkanCoreImpl.hpp, physical-camera mode);
//     anything composited afterwards has to apply the same factor or it is
//     wrong by the whole auto-exposure gain. It is 1.0 in legacy mode, which
//     is exactly why this is easy to leave out and never notice.
//  2. COLOUR DOMAIN. 3DGS spherical harmonics are fitted against sRGB-encoded
//     training images, so the basis evaluates to a DISPLAY-referred colour —
//     that is the number the GL path writes straight to its framebuffer.
//     sceneHdr is linear, so the splat is sRGB-decoded on the way in (see
//     splatSrgbToLinear in splat_common.glsl).
//  3. FOG. deferred_shade_60_fog_volumetrics.glsl bakes fog into sceneHdr
//     during the shade, so a later compositor gets none of it. Applied here
//     per splat from the accumulated expected depth (V2), the same way
//     particle_light.comp does it for billboards.
//  4. DEPTH. The G-buffer depth is REVERSED-Z; the tile loop linearizes it
//     through the camera's projection inverse and stops accumulating behind
//     opaque geometry — depth-test on, depth-write off, in software. Depth
//     WRITE has one consumer that needs it back: the post-TAA overlay pass
//     draws after this one and tests against a depth buffer nothing here
//     touched, so a wireframe behind a cloud drew over it. The depth AOV is
//     stamped into that buffer by VulkanCoreRecord::recordSplatOverlayDepthStamp
//     (shaders/splat_overlay_depth.frag) to close it.
//  5. DETERMINISM. Sensor goldens depend on it. The danger is the tile
//     expansion: an
//     atomic-append expansion produces a different order every run. This one
//     computes a prefix sum over the per-splat tile counts first, so every
//     (splat, tile) pair lands at an index that is a pure function of the
//     input. The radix sort below it is a stable LSD counting sort, which is
//     likewise a pure function of its input.
//
// PIPELINE (all compute, one descriptor set layout shared by every stage):
//
//   splat_project   per splat: view transform, near/frustum cull, 3D->2D
//                   covariance (EWA), conic, tile rect, SH colour, and an
//                   atomic min/max over the visible view distances
//   splat_scan  x N exclusive prefix sum over the per-splat tile counts
//                   (recursive: block scan -> scan the block sums -> add back)
//   splat_expand    writes (key, splatIndex) for every covered tile at
//                   offset[splat] + k — deterministic by construction
//   splat_radix x 2 per 4-bit digit: per-block histogram, then a scatter that
//                   uses the scanned histogram plus a stable in-block rank
//   splat_range     tile -> [begin, end) from the sorted key array
//   splat_raster    one 16x16 workgroup per tile, one thread per pixel,
//                   front-to-back with transmittance early-out
//
// SCOPE. The primary view always; a secondary view (addView) only if it ASKS,
// one by one (VulkanRenderer::setViewSplats), because "secondary" does not mean
// "wants splats" — an RGB camera sensor pointed at a scan does, an editor
// viewport pane costs the same sort and may not. Default OFF everywhere, so a
// scene that does not ask renders byte-identically to before this existed. What
// an opted-in view buys is a SECOND run of the whole pipeline below at its own
// extent: the radix sort scales with SPLAT COUNT, not view size, so a 640x480
// sensor on a 5M-splat town costs the same ~8-13 ms the primary does. That is
// the caller's choice to make, which is why it is a per-view switch and not a
// renderer mode. The depth AOV stays PRIMARY-ONLY (a secondary's AOV image is
// 1x1 by construction), and so does the debug checksum readback.
//
// Splats still cast no shadows, contribute to no probe and are invisible to the
// RT sensors —
// all deliberate, all documented as out of scope in
// plans/gaussian-splats-vulkan.md. Reflections crossed that wall on purpose
// (plans/splat-volume-reflections.md): every upload also BAKES the cloud into
// a small rgba16f volume (splat_bake_*.comp) that the deferred shade's traced
// reflection legs march (splat_volume.glsl's svLeg, water/glass and glossy) —
// primary view only, behind flags bit 12, so the sensor wall stands.
//
// The one crack in that wall is the DEPTH AOV
// (VulkanRenderer::setSplatDepthAov). It does not put splats in an
// acceleration structure — nothing here does — but it does let a consumer
// outside the renderer find out that a cloud was in front of a given pixel and
// how far away, which is what picking needs and what a coarse occupancy build
// can start from. Off by default; see splat_raster.comp for the coverage gate,
// the nearest-wins rule and the two statistics it can carry (expected depth for
// picking, median depth for surface fusion — plans/splat-surface-bake.md), and
// the note on setSplatDepthAov for what each value is and is not.
//
// And the second crack, the one plans/splat-surface-bake.md's P2 opened: a
// scan can now return LIDAR RANGES AND SENSOR DEPTH, through a proxy.
// threepp::splats::bakeSurface fuses that median depth into a triangle mesh,
// splats::makeSensorMesh marks it VulkanRenderer::kSensorOnlyLayer, and
// setSensorOnlySurfaces(true) lets the scene's lidar beams perceive it — plus
// the secondary views that ASK, one by one (setViewSensorSurfaces), because an
// RGB camera preview and an editor viewport pane are secondary views too and
// the shell must not stand in front of the splats there. The walls that still
// stand: no sensor sees the SPLATS —
// what it sees is a mesh baked from them, at voxel resolution, with no colour
// (an RGB view that wants the real thing takes setViewSplats above, which is a
// raster, not a trace); the splat pass is still absent from every
// acceleration structure; and the opt-in defaults OFF, with the mesh's TLAS
// instance carrying mask 0 until it is taken, so a scene that does not ask
// renders and senses exactly as it did before.
//
// ENVIRONMENT KNOBS, all off by default and all A/B switches rather than
// settings — each one turns off a term so its contribution can be MEASURED
// rather than asserted (which is how the motion-vector +2.4 dB and the
// blend-domain +9.6 dB numbers in the commit log were arrived at):
//
//   THREEPP_VK_SPLAT_CHECKSUM=1   hash the sorted key/payload arrays, print the
//                                 entry count, and assert two invariants (the
//                                 scan really is the exclusive scan; the sorted
//                                 keys really are non-decreasing)
//   THREEPP_VK_SPLAT_NOMOTION=1   skip the gbufMotion write
//   THREEPP_VK_SPLAT_NOFOG=1      skip the per-splat fog
//   THREEPP_VK_SPLATVOL_OFF=1     never bake the reflection volume: no image,
//                                 no dispatch, no VRAM, volumeEntries() empty.
//                                 Read ONCE at construction, so it is an A/B
//                                 lever for a whole run rather than a setting.
//   THREEPP_VK_SPLATVOL_RES=N     longest-axis voxel budget for the bake,
//                                 default 128, clamped to [16, 256]. 256 is
//                                 8x the VRAM (128 MB/cloud resident, 268 MB
//                                 transient bake scratch) and buys metre ->
//                                 half-metre voxels on a building-scale scan.
//                                 Read once, like the others: two clouds baked
//                                 under different budgets in one run would
//                                 hash differently for reasons no test could
//                                 name.
//
// THE VOLUME BAKE. Separate from everything above and off the frame path:
// uploadCloud voxelizes each cloud once, in cloud-local space, into an rgba16f
// 3D image (rgb = linear radiance, a = sigma_t per local metre) that rays the
// tile rasterizer cannot serve — reflection legs first — march as a
// participating medium. The primary camera leg must NEVER march it: SplatPass
// already composites the real thing there and a second contribution would
// double-count every visible cloud. See plans/splat-volume-reflections.md.

#ifndef THREEPP_VULKAN_SPLAT_PASS_HPP
#define THREEPP_VULKAN_SPLAT_PASS_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {
    class SplatCloud;
    struct SplatData;
}

namespace threepp::vulkan {

    class VulkanContext;
    class GpuTimings;

    // ── The reflection-volume table (plans/splat-volume-reflections.md) ──────
    // Volumes bound to the deferred set at once. KEEP IN SYNC with
    // kMaxSplatVolumes in shaders/splat_volume.glsl, with kMaxSplatVolumeSlots
    // in DeferredShade.cpp, and with SplatPass::kMaxClouds — the cap exists so
    // no resident cloud can fail to get a slot.
    inline constexpr std::uint32_t kMaxSplatVolumes = 8;

    // Binding 71 of the deferred set. MUST match SplatVolumeUbo in
    // shaders/splat_volume.glsl, which is std140 (NOT scalar) because the header
    // is pulled into shaders that do not all enable GL_EXT_scalar_block_layout —
    // everything here is a mat4/vec4/uvec4, so the two layouts coincide anyway.
    // That every member is 16-byte-aligned is the invariant the assert guards:
    // 8*64 + 3*8*16 + 16 = 912.
    struct SplatVolumeUboGpu {
        float         worldToUvw[kMaxSplatVolumes][16]; // world point -> [0,1]^3, column-major
        float         worldBoxMin[kMaxSplatVolumes][4]; // conservative world AABB of the OBB…
        float         worldBoxMax[kMaxSplatVolumes][4]; // …used only for the interval clip
        float         params[kMaxSplatVolumes][4];      // x = sigmaScale / cbrt(|det model3x3|)
        std::uint32_t counts[4];                        // x = active volumes
    };
    static_assert(sizeof(SplatVolumeUboGpu) == 912, "SplatVolumeUbo layout drift");

    class SplatPass {

    public:
        SplatPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight);
        ~SplatPass();
        SplatPass(const SplatPass&) = delete;
        SplatPass& operator=(const SplatPass&) = delete;

        // One entry per SplatCloud the sidecar collector found this frame.
        // `cloud` is borrowed for the duration of the call only — the GPU
        // residency cache keys off the pointer, and a cloud that stops being
        // collected is retired after its last referencing frame drains.
        struct CloudEntry {
            const SplatCloud* cloud = nullptr;
            float model[16]{};// cloud local -> world, column-major
            // p1 / p99 of this frame's view distances, estimated on the CPU
            // from a fixed-stride sample the same way SplatCloud::sortByDepth
            // does. The sort key's content interval — see splatDepthBucket.
            float pLo = 0.f, pHi = 0.f;
            bool  debugNonFinite = false;
            // SplatCloud::pointMix / pointSigmaPixels, copied per frame: the
            // shaders read them from the per-cloud UBO slot.
            float pointMix = 0.f;
            float pointSigma = 0.f;
            // Partial submission for per-chunk LOD: (source offset, count) into
            // the cloud's own splats, submitted in the order given. Empty =
            // submit the whole cloud, which is the identity path.
            //
            // This is the shape per-chunk LOD needs because the two obvious
            // alternatives measured badly: one SplatCloud per chunk costs ~1.3 ms
            // EACH (record() runs the whole pipeline per cloud, tile walk
            // included), and re-packing a merged buffer per selection change is a
            // re-upload of up to 1.2 GB. Ranges keep every chunk at every level
            // resident and uploaded once, and make the per-frame selection a
            // handful of integers. A chunk that fails a frustum test is simply
            // not in the list, so chunk culling is this and nothing more.
            std::vector<std::pair<uint32_t, uint32_t>> ranges;
        };

        // The CURRENT prefiltered environment, pushed every frame before
        // syncClouds. resize() also carries it, but resize() runs on swapchain
        // lifecycle — an environment REBUILT between resizes (scene switch, sky
        // change) destroys the view this pass cached, and the next cloud upload
        // then writes a dangling handle into its descriptor sets
        // (VUID-VkWriteDescriptorSet-descriptorType-02996, and a driver-level
        // access violation when sampled). A change marks the sets dirty;
        // syncClouds treats that as structural and rewrites them post-idle.
        void setEnvironment(VkImageView view, VkSampler sampler, uint32_t mips);

        // The immediate flavour, for the one call site that destroys the old
        // environment MID-FRAME (beginDeferredFrame's env swap, device already
        // drained): setEnvironment alone leaves the resident sets naming the
        // freed view until the NEXT syncClouds, and the splat dispatch recorded
        // later this same frame would bind them (VUID-vkCmdDispatch-None-08114).
        // Rewrites every resident set now and clears the dirty flag.
        void rewriteEnvironment(VkImageView view, VkSampler sampler, uint32_t mips);

        // Upload anything new, evict anything gone, size the shared scratch to
        // what this frame actually SUBMITS — growing for a bigger cloud and
        // shrinking (with hysteresis) when the demand halves, because after
        // indirect dispatch the oversize costs VRAM only and the doctrine is
        // absolute caps. Runs OUTSIDE command recording (the staging copy is a
        // one-shot submit), so it may allocate freely.
        //
        // `parked`: clouds that are IN the scene but not effectively visible.
        // They draw nothing and the scratch is not sized for them, but their
        // geometry and SH buffers stay resident, so toggling visibility on a
        // 5M-splat scan costs nothing instead of a seconds-long re-upload each
        // way. Hidden is not deleted; only a cloud absent from BOTH lists ages
        // out. The corollary is stated rather than hidden: a hidden scan holds
        // its VRAM on purpose, because it is still in the scene.
        void syncClouds(const std::vector<CloudEntry>& clouds,
                        const std::vector<const SplatCloud*>& parked = {});

        [[nodiscard]] bool hasClouds() const { return !frameClouds_.empty(); }

        // Per-frame-in-flight images the pass reads/writes. Rewrites every
        // descriptor set; call from the same place DofPass::resize is called.
        struct ResizeInputs {
            const VkImageView* sceneHdrPerFrame = nullptr;// rgba16f GENERAL
            const VkImageView* depthPerFrame    = nullptr;// D32 reversed-Z, read-only layout
            const VkImageView* motionPerFrame   = nullptr;// rgba16f, STORAGE-capable
            const VkImage*     motionImages     = nullptr;// [framesInFlight], for the layout flip
            const VkImageView* idsPerFrame      = nullptr;// rgba16ui, sampled read-only
            // Expected-depth AOV target, r32f in GENERAL. Full-res when the
            // AOV is enabled, 1x1 when it is not — the binding needs a real
            // image either way. Null leaves the pass unable to write its sets
            // at all, which is why the renderer always supplies one.
            const VkImageView* splatDepthPerFrame = nullptr;
            const VkImage*     splatDepthImages   = nullptr;// [framesInFlight], for the clear
            // The same fog / cloud / lights UBOs the deferred shade reads, and
            // the prefiltered env — the splat pass has to re-derive the fog the
            // shade already baked into sceneHdr for everything else.
            const VkBuffer*    fogUbos    = nullptr;// [framesInFlight]
            const VkBuffer*    cloudUbos  = nullptr;// [framesInFlight]
            const VkBuffer*    lightsUbos = nullptr;// [framesInFlight]
            VkImageView        envView    = VK_NULL_HANDLE;
            VkSampler          envSampler = VK_NULL_HANDLE;
            uint32_t           envMips    = 1;
        };
        // `target` picks which composite destination these images describe:
        // 0 is the primary and always exists; a secondary view that opted into
        // splats owns one of the remaining slots (acquireTarget). Every target
        // carries its own extent, tile grid and descriptor sets; they SHARE the
        // sort scratch, which is safe because the pass already reuses it
        // sequentially across clouds behind its own barriers.
        void resize(uint32_t width, uint32_t height, const ResizeInputs& in,
                    uint32_t target = 0);

        // Claim a non-primary target slot, kNoTarget when the table is full.
        // Slots are claimed for a view's lifetime and returned by releaseTarget;
        // the descriptor sets they name are per (cloud, target, frame) and are
        // written by resize/uploadCloud, never mid-frame.
        static constexpr uint32_t kMaxTargets = 4;
        static constexpr uint32_t kNoTarget   = 0xFFFFFFFFu;
        [[nodiscard]] uint32_t acquireTarget();
        void releaseTarget(uint32_t target);
        // Does this slot have images to composite into? False until its first
        // resize, which is what keeps a view that opted in before its resources
        // existed from recording a dispatch against unwritten sets.
        [[nodiscard]] bool targetValid(uint32_t target) const;

        [[nodiscard]] bool valid() const {
            return targets_[0].width > 0 && sampler_ != VK_NULL_HANDLE;
        }

        // What the driver needs that only the frame knows. Matrices are
        // column-major, threepp/GL convention (the projection is the camera's
        // OWN GL-style matrix with the raster's jitter shear already applied —
        // the pass does its own NDC->pixel mapping and never sees reverse-Z).
        struct RecordParams {
            float view[16]{};      // world -> view (camera.matrixWorldInverse)
            float proj[16]{};      // view -> clip, GL convention, jittered
            float projInverse[16]{};// clip -> view, REVERSE-Z (matches gbufDepth)
            float camWorld[16]{};  // view -> world (camera.matrixWorld)
            // View space -> the PREVIOUS frame's unjittered clip, composed on
            // the host from TaaResolve's own sky-reprojection so the splat
            // motion vectors come out of the same matrices the raster's do.
            float prevVPfromView[16]{};
            float camPos[3]{};
            float camFwd[3]{};
            float jitterClip[2]{};// the shear already applied to `proj`
            float nearPlane   = 0.1f;
            float preExposure = 1.f;
            bool  orthographic = false;
            bool  depthTest    = true;
            bool  motionVectors = true;// write gbufMotion + the reactivity flag
            bool  fog           = false;// a medium is active this frame
            // Export a per-pixel view distance to the depth AOV
            // (VulkanRenderer::setSplatDepthAov). Off by default: the image is
            // 1x1 unless the renderer was asked for it.
            bool  depthAov      = false;
            // Which statistic that AOV carries: the MEDIAN view distance (the
            // transmittance-0.5 crossing) instead of the expected one. Only the
            // AOV changes — motion vectors and per-splat fog keep using the
            // expected value, which is the depth their reprojection is defined
            // against. See plans/splat-surface-bake.md.
            bool  depthMedian   = false;
            // The scene background is a flat colour, so PostComposite hands
            // those pixels back verbatim and the shade never pre-exposed them.
            bool  bgIsSolidColor = false;
            // Hash the sorted key/payload arrays and the composited pixels
            // (VulkanRenderer::setSplatDebugChecksum).
            bool  checksum = false;
            // Which composite target this dispatch writes (see resize). 0 is
            // the primary; a secondary view that opted into splats passes the
            // slot it acquired. Only the target's images, extent and UBO region
            // change — the sort scratch, and therefore the determinism
            // argument, are the same ones the primary uses.
            uint32_t target = 0;
            // Optional per-stage timestamps. The caller already brackets the
            // whole pass with TP_Splat; handing the pool in lets the pass split
            // that into project / sort / raster from the inside, where the stage
            // boundaries actually are. Null = no per-stage timing.
            GpuTimings* timings = nullptr;
        };
        void record(VkCommandBuffer cb, uint32_t frame, const RecordParams& p);

        // Zero the depth AOV for this frame slot. Separate from
        // record() and called BEFORE it, because record() is skipped outright
        // on a frame with no clouds and the AOV still has to describe THAT
        // frame — an empty one. No-op when no AOV image was supplied.
        void clearDepthAov(VkCommandBuffer cb, uint32_t frame);

        // Debug/test surface. [0] sorted-key hash, [1] sorted-payload hash,
        // [2] composited-colour hash, [3] expanded entry count. Stalls the
        // device — a test accessor, never on the render path.
        void readDebug(uint64_t out[4]) const;

        // Total expanded (splat, tile) pairs that did not fit the budget on
        // the last drained frame. Non-zero means the frame was TRUNCATED.
        [[nodiscard]] uint32_t lastOverflow() const;

        // How many clouds hold GPU buffers right now. A test surface, like
        // readDebug: the eviction contract ("a deleted cloud's buffers are
        // freed once its last referencing frame drains") is invisible from the
        // outside except as VRAM, and VRAM is not assertable — this is.
        [[nodiscard]] std::size_t residentCount() const { return resident_.size(); }

        // The shared scratch's current high-water in splats — the other half of
        // the same test surface: "the scratch was released/shrunk" is a VRAM
        // claim, and VRAM is not assertable; this is.
        [[nodiscard]] uint32_t scratchSplats() const { return maxSplats_; }

        // ── The reflection volume (plans/splat-volume-reflections.md) ────────

        // One baked cloud, as a consumer of the volume table needs it. The
        // struct is deliberately minimal: worldToUvw and the world AABB are
        // composed on the HOST from `model` and the local box, because the
        // shader side wants one mat4 per entry and the host is where the
        // inverse belongs.
        struct VolumeEntry {
            VkImageView view = VK_NULL_HANDLE;// rgba16f sampler3D source, GENERAL
            float    model[16]{};      // cloud local -> world, column-major
            float    localBoxMin[3]{}; // the bake box, cloud-local
            float    localBoxSize[3]{};// strictly positive
            // Smallest voxel edge, cloud-LOCAL metres (min over axes of
            // size/res). The march scales its step count to this so a
            // building-scale volume is sampled at its own resolution instead
            // of a fixed 16 (splat_volume.glsl, params[i].y — the consumer
            // multiplies by the model scale on the way in, like sigma).
            float    voxelLocal = 0.f;
            uint32_t count = 0;        // splats in the cloud
        };

        // The volumes to bind THIS frame, built from frameClouds_ — so the
        // collector's own test decides membership, and that test is a
        // VISIBILITY test, not a frustum test (VulkanCoreScene.cpp:2848
        // traverses everything and only parks what is hidden). That is
        // load-bearing rather than incidental: a cloud BEHIND the camera still
        // has to appear in the mirror, which is precisely the failure mode
        // screen-space reflections cannot fix and this whole design exists to
        // avoid. Hidden/parked clouds are absent by construction, so an
        // invisible cloud does not haunt the water either.
        //
        // Empty under THREEPP_VK_SPLATVOL_OFF, and empty until the first bake.
        [[nodiscard]] std::vector<VolumeEntry> volumeEntries() const;

        // Bumped whenever the SET of baked volumes changes — a bake completes,
        // a cloud is retired. Image bindings change only then; model matrices
        // and per-frame params ride the UBO without touching descriptors. The
        // same role ParticleFieldPass::densityGeneration() plays for the dust
        // table, and consumed the same way (compare, rewrite, dirty the rest).
        [[nodiscard]] std::uint64_t volumeGeneration() const { return volumeGen_; }

        // Resident volume VRAM. The third of the assertable-VRAM surfaces next
        // to residentCount() / scratchSplats(), and for the same reason: "the
        // volume was freed with its cloud" is a VRAM claim, VRAM is not
        // assertable, and this is. Exactly 0 under THREEPP_VK_SPLATVOL_OFF.
        [[nodiscard]] std::uint64_t volumeBytes() const;

        // Test surface. [0] = FNV-1a hash of every resident volume's texels, in
        // ascending slot order (resident_ is unordered, so the ORDER has to be
        // imposed or the hash would not even be stable within one run); [1] =
        // total texels hashed; [2] = texels whose sigma is non-zero, so a bake
        // that deposited NOTHING cannot pass a determinism assertion by being
        // reproducibly empty. Stalls the device and allocates a readback buffer
        // the size of the volume — a test accessor, in readDebug's style, never
        // on the render path.
        void readVolumeHash(std::uint64_t out[3]) const;

    private:
        // The expanded key list is the sum of tiles covered per splat, and that
        // is DATA, not a constant. So this is only the FIRST GUESS: the
        // expansion reports how many entries it could not fit and syncClouds
        // resizes to the number the frame actually wanted — one truncated
        // frame, then correct, instead of a doubling ladder that costs a wrong
        // frame per rung.
        //
        // The guess matters for SPEED, not just memory, because the radix sort
        // and the tile-range pass are dispatched over the BUDGET (the expanded
        // count lives on the GPU and the host never sees it in time to size a
        // dispatch). Measured per-splat need at 960x600: 2.52 on the 216k ATLAS
        // scan, 1.74 on the 5.0M Sanctuaire scan — and orbit frame time on
        // Sanctuaire against the guess:
        //
        //     guess 8   36.2 ms   (4.6x more budget than the frame used)
        //     guess 4   28.8 ms   (2.3x)   <- here: still no truncated frame
        //     exact     25.2 ms   (1.25x, reached by letting it grow from 1)
        //
        // 4 is the largest guess that fits both real scans without truncating
        // anything, and it collects most of the difference. Closing the last
        // 3.6 ms wants an indirect dispatch off the GPU-side count, which is on
        // the V3 list in doc/vulkan_splats.md, not here.
        static constexpr uint32_t kEntriesPerSplat = 4;
        // Hard ceiling on the four key/payload buffers (4 B per entry each).
        // Past it the frame is genuinely too big for this design and the host
        // says so out loud, once, with the numbers — a silently truncated
        // splat cloud looks like a rendering bug forever.
        static constexpr uint64_t kMaxEntryBytes = 768ull * 1024 * 1024;
        static constexpr uint32_t kMaxEntries    = uint32_t(kMaxEntryBytes / 16);

        // How many clouds can be resident at once. Each one costs a descriptor
        // set per frame-in-flight and a UBO slot; the cap exists so the pool
        // and the UBO can be sized up front, and 8 is well past what a scene
        // that also has to fit in VRAM will hold.
        //
        // KEEP IN SYNC with kMaxSplatVolumes above: the reflection-volume table
        // has one slot per resident cloud, so a cloud that could be resident and
        // could not be bound would be a scan that renders but never reflects.
        static constexpr uint32_t kMaxClouds = 8;

        struct Cloud {
            Buffer   geom;// [n] mean.xyz + opacity + cov[6], fp32, 40 B
            Buffer   sh;  // [n * coeffs] two uints per coefficient (half-packed)
            uint32_t count      = 0;
            uint32_t shCoeffs   = 1;
            uint32_t shDegree   = 0;
            uint32_t slot       = 0;// UBO slot, stable for the cloud's lifetime
            std::vector<VkDescriptorSet> sets;// [framesInFlight]
            uint64_t lastSeen = 0;
            // The residency cache is keyed by SplatCloud POINTER, and a pointer
            // is not an identity: delete a cloud, allocate another, and the new
            // one can land at the SAME address — the cache then serves the dead
            // cloud's buffers under the live cloud's key, which renders the old
            // scan with the new transform and no error anywhere. The overlay
            // line cache had exactly this bug. uuid is the identity check.
            std::string uuid;

            // ── The reflection volume, baked once by uploadCloud ─────────────
            // Image2D is this tree's 3D-image record too — ParticleFieldPass
            // holds its density volumes in one (ParticleFieldPass.hpp:678,
            // built by Impl::createImage3D) — so the depth rides alongside in
            // volRes rather than in a type that does not exist. Null image =
            // not baked (THREEPP_VK_SPLATVOL_OFF, or a zero-splat cloud).
            //
            // Lives in GENERAL for its whole life, matching the dust volumes
            // (ParticleFieldPass.cpp:501): the consumer table binds live slots
            // and dummy slots through one array, and one layout across all of
            // them is one less thing for that write to get wrong.
            Image2D  volume{};
            uint32_t volRes[3]{};
            float    localBoxMin[3]{};
            float    localBoxSize[3]{};
        };

        // Retire every resident entry whose cloud was not submitted for
        // `framesInFlight_ + 1` consecutive syncs — one sync per frame, one
        // frame per submit, so by then no in-flight command buffer references
        // its buffers or sets and they can be destroyed WITHOUT a device stall.
        // The slot and descriptor sets go to a freelist for the next upload
        // (sets are rewritten on upload anyway; the pool never has to free).
        //
        // This is the eviction the class comment always promised ("retired
        // after its last referencing frame drains") and nothing implemented:
        // lastSeen was written and never read, so a deleted 5M-splat scan kept
        // its ~1.2 GB of buffers forever and the NEXT import allocated on top —
        // device OOM at import, reported as "crash when I deleted a splat and
        // then loaded a new one". The visible cost of eviction: a cloud HIDDEN
        // for a few frames is also retired, and pays its upload again when
        // re-shown. Predictable stall beats unbounded leak.
        void retireStale();

        // One composite destination. Slot 0 is the primary and is claimed from
        // construction; the rest are handed to secondary views that asked for
        // splats. Everything here is per-VIEW state — the sort scratch below is
        // deliberately not, because it is written and consumed inside one
        // dispatch chain and the targets record sequentially.
        struct Target {
            bool     claimed = false;// slot 0 always; the others on request
            uint32_t width = 0, height = 0;
            uint32_t tilesX = 0, tilesY = 0;
            // Cached per-frame image views so a cloud added mid-run can have its
            // freshly-allocated sets written without another resize() round trip.
            std::vector<VkImageView> sceneHdrViews, depthViews, motionViews, idsViews;
            std::vector<VkImageView> splatDepthViews;
            std::vector<VkImage>     motionImages, splatDepthImages;
        };
        std::array<Target, kMaxTargets> targets_{};

        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;
        uint64_t       syncSerial_ = 0;

        std::unordered_map<const SplatCloud*, std::unique_ptr<Cloud>> resident_;
        // This frame's draw list, in collection order (deterministic: the
        // scene traversal is).
        struct FrameCloud {
            Cloud* cloud = nullptr;
            float  model[16]{};
            float  pLo = 0.f, pHi = 0.f;
            bool   debugNonFinite = false;
            float  pointMix = 0.f;
            float  pointSigma = 0.f;
            // Validated copy of CloudEntry::ranges, clamped to the cloud and to
            // kMaxRanges, plus the total they submit. submitCount == cloud->count
            // and an empty list is the whole-cloud path.
            std::vector<std::pair<uint32_t, uint32_t>> ranges;
            uint32_t submitCount = 0;
        };
        std::vector<FrameCloud> frameClouds_;
        uint32_t slotsUsed_ = 0;
        uint64_t shrinkSince_ = 0;// first sync the scratch looked 2x oversized; 0 = it doesn't
        // (slot, descriptor sets) returned by retired clouds, reused by the
        // next upload so slots and pool capacity cycle instead of running out.
        std::vector<std::pair<uint32_t, std::vector<VkDescriptorSet>>> freeSlots_;

        // Shared scratch, sized to the largest resident cloud. ONE set, not
        // one per frame-in-flight: every buffer here is written and consumed
        // inside a single command buffer, and the leading barrier's first
        // synchronization scope covers the previous submission on the queue —
        // the same argument DofPass makes for its half-res scratch, and worth
        // ~70 MB here.
        uint32_t maxSplats_   = 0;
        uint32_t entryBudget_ = 0;
        bool     budgetCapped_ = false;// warned about the ceiling already
        Buffer   projBuf_{}, countBuf_{}, offsetBuf_{};
        Buffer   keyA_{}, valA_{}, keyB_{}, valB_{};
        Buffer   rangeBuf_{}, globalBuf_{}, histBuf_{}, scanBuf_{};
        // Two VkDispatchIndirectCommands written by splat_indirect.comp from the
        // expanded entry count. Fixed size, written and consumed inside one
        // submission, so one buffer serves every cloud and every frame.
        Buffer   indirectBuf_{};
        std::vector<Buffer> uboBuf_;  // [framesInFlight], host-visible
        std::vector<Buffer> debugBuf_;// [framesInFlight], host-visible readback

        VkSampler             sampler_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsLayout_  = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_  = VK_NULL_HANDLE;

        VkPipeline projectPipe_  = VK_NULL_HANDLE;
        VkPipeline scanPipe_     = VK_NULL_HANDLE;
        VkPipeline scanAddPipe_  = VK_NULL_HANDLE;
        VkPipeline expandPipe_   = VK_NULL_HANDLE;
        VkPipeline indirectPipe_ = VK_NULL_HANDLE;
        VkPipeline histPipe_     = VK_NULL_HANDLE;
        VkPipeline scatterPipe_  = VK_NULL_HANDLE;
        VkPipeline rangePipe_    = VK_NULL_HANDLE;
        VkPipeline rasterPipe_   = VK_NULL_HANDLE;
        VkPipeline checksumPipe_ = VK_NULL_HANDLE;

        // ── Volume bake: its own layout, pool and single set ─────────────────
        // The bake binds NOTHING the frame binds (geom SSBO, sh SSBO, a
        // transient scratch SSBO, the out image), so it gets its own four-
        // binding layout rather than a tenth user of the raster's twenty-four.
        //
        // ONE descriptor set for every cloud ever uploaded: uploadCloud's
        // one-shot waits on the queue before it returns, so the set is provably
        // not in flight when the next upload rewrites it. All null under
        // THREEPP_VK_SPLATVOL_OFF — the pipelines are not even created.
        VkDescriptorSetLayout bakeDsLayout_  = VK_NULL_HANDLE;
        VkPipelineLayout      bakePipeLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      bakeDescPool_  = VK_NULL_HANDLE;
        VkDescriptorSet       bakeSet_       = VK_NULL_HANDLE;
        VkPipeline            bakeScatterPipe_ = VK_NULL_HANDLE;
        VkPipeline            bakeResolvePipe_ = VK_NULL_HANDLE;
        // THREEPP_VK_SPLATVOL_OFF / _RES, read once at construction.
        bool     volumeOff_ = false;
        uint32_t volMaxRes_ = 128;// == kVolMaxResDefault; env-clamped in ctor
        uint64_t volumeGen_ = 0;

        std::vector<VkBuffer>    fogUbos_, cloudUbos_, lightsUbos_;
        VkImageView envView_    = VK_NULL_HANDLE;
        bool        envDirty_   = false;// sets hold a dead env view; rewrite post-idle
        VkSampler   envSampler_ = VK_NULL_HANDLE;
        uint32_t    envMips_    = 1;

        VkDeviceSize uboStride_ = 0;// per-cloud slot stride, alignment-padded
        mutable uint32_t lastFrame_ = 0;// frame slot the last record() wrote

        void createPipelines();
        void createDescriptorPool();
        void allocateScratch(uint32_t maxSplats, uint32_t entryBudget);
        // Every claimed target's sets, or one target's. The one-target flavour
        // exists so a slot claimed while other targets' sets are in flight can
        // be written without touching them (VUID-…-03047).
        void writeSets(Cloud& c);
        void writeSets(Cloud& c, uint32_t target);
        // rangeBuf_ is sized from the tile grid, and the targets do not share
        // an extent — so it is sized to the largest CLAIMED one.
        void resizeTileRange();
        void uploadCloud(Cloud& c, const SplatCloud& src);
        // Everything a resident cloud owns. Three sites destroy clouds (the
        // destructor, retireStale, and syncClouds' recycled-address branch) and
        // the volume is easy to forget in any one of them, so they share this.
        // Bumps volumeGeneration when the cloud had a volume to free.
        void destroyCloudResources(Cloud& c);
        // Decides the cloud's bake box and per-axis resolution on the CPU (the
        // only part of the bake that reads the SplatData at all). Leaves
        // volRes zeroed — "do not bake this cloud" — for a cloud it cannot
        // bound.
        void planVolume(Cloud& c, const SplatData& data);
        // Records the scatter + resolve dispatches into the caller's one-shot,
        // after the staging copies. `scratch` is the caller's transient buffer,
        // destroyed once the one-shot's wait returns.
        void recordBake(VkCommandBuffer cb, Cloud& c, const Buffer& scratch);
        // const because it touches no member state of its own — the readback
        // accessors are const and record through it too.
        void oneShot(const std::function<void(VkCommandBuffer)>& body) const;
        void barrier(VkCommandBuffer cb) const;
        // The compute write -> vkCmdDispatchIndirect read hazard. barrier()
        // covers compute and transfer only; indirect command fetch is its own
        // stage and access, and omitting it is the kind of miss that works on
        // one driver and hangs on another.
        void barrierIndirect(VkCommandBuffer cb) const;
        // Records a recursive exclusive prefix sum. mode0 picks the level-0
        // array: 0 = counts[] -> offsets[], 2 = hist[] in place. Higher levels
        // always run in scanScratch.
        void recordScan(VkCommandBuffer cb, uint32_t n, uint32_t mode0);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_SPLAT_PASS_HPP
