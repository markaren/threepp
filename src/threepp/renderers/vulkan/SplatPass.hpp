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
//     opaque geometry — depth-test on, depth-write off, in software.
//  5. DETERMINISM. Sensor goldens depend on it and "RT is never bit-exact" is
//     already a scar in this tree. The danger is the tile EXPANSION: an
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
// SCOPE. Primary view only: secondary views (addView) skip the splat pass
// entirely rather than silently painting splats into a sensor AOV. Splats cast
// no shadows, appear in no reflection, contribute to no probe and are invisible
// to the RT sensors — all deliberate, all documented as out of scope in
// plans/gaussian-splats-vulkan.md.
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

#ifndef THREEPP_VULKAN_SPLAT_PASS_HPP
#define THREEPP_VULKAN_SPLAT_PASS_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {
    class SplatCloud;
}

namespace threepp::vulkan {

    class VulkanContext;
    class GpuTimings;

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
        void resize(uint32_t width, uint32_t height, const ResizeInputs& in);

        [[nodiscard]] bool valid() const { return width_ > 0 && sampler_ != VK_NULL_HANDLE; }

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
            // The scene background is a flat colour, so PostComposite hands
            // those pixels back verbatim and the shade never pre-exposed them.
            bool  bgIsSolidColor = false;
            // Hash the sorted key/payload arrays and the composited pixels
            // (VulkanRenderer::setSplatDebugChecksum).
            bool  checksum = false;
            // Optional per-stage timestamps. The caller already brackets the
            // whole pass with TP_Splat; handing the pool in lets the pass split
            // that into project / sort / raster from the inside, where the stage
            // boundaries actually are. Null = no per-stage timing.
            GpuTimings* timings = nullptr;
        };
        void record(VkCommandBuffer cb, uint32_t frame, const RecordParams& p);

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

        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;
        uint32_t       width_ = 0, height_ = 0;
        uint32_t       tilesX_ = 0, tilesY_ = 0;
        uint64_t       syncSerial_ = 0;

        std::unordered_map<const SplatCloud*, std::unique_ptr<Cloud>> resident_;
        // This frame's draw list, in collection order (deterministic: the
        // scene traversal is).
        struct FrameCloud {
            Cloud* cloud = nullptr;
            float  model[16]{};
            float  pLo = 0.f, pHi = 0.f;
            bool   debugNonFinite = false;
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

        // Cached per-frame image views so a cloud added mid-run can have its
        // freshly-allocated sets written without another resize() round trip.
        std::vector<VkImageView> sceneHdrViews_, depthViews_, motionViews_, idsViews_;
        std::vector<VkImage>     motionImages_;
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
        void writeSets(Cloud& c);
        void uploadCloud(Cloud& c, const SplatCloud& src);
        void oneShot(const std::function<void(VkCommandBuffer)>& body);
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
