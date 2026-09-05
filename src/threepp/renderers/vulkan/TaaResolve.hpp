// TaaResolve — temporal anti-aliasing resolve pass.
//
// Reads the denoise output (`inputView(frame)` — denoise targets this image
// when TAA is active), reprojects last frame's history via the raster
// G-buffer's motion vector, blends with neighborhood-AABB clamp, writes the
// result to the swapchain + a fresh history slot for next frame.
//
// Extracted from VulkanRenderer.cpp during the file split. Owns its
// pipeline, descriptor set layout + pool + sets, sampler, input/history
// image ping-pong. External deps (raster G-buffer views, swapchain views)
// are passed in at descriptor-write time.

#ifndef THREEPP_VULKAN_TAA_RESOLVE_HPP
#define THREEPP_VULKAN_TAA_RESOLVE_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class TaaResolve {

    public:
        // `cmdPool` is used internally for one-shot image layout transitions
        // (UNDEFINED → GENERAL) at image creation time. Pipeline + layout +
        // sampler + descriptor pool + descriptor sets are allocated here;
        // images are deferred to `createImages` so we don't need the surface
        // size yet.
        TaaResolve(VulkanContext& ctx,
                   VkCommandPool cmdPool,
                   uint32_t imageCount,
                   uint32_t framesInFlight);
        ~TaaResolve();
        TaaResolve(const TaaResolve&) = delete;
        TaaResolve& operator=(const TaaResolve&) = delete;

        // Allocate input images at the render extent and history images at
        // the output (swapchain) extent. When the two extents differ the
        // resolve runs as a temporal upsampler; when equal it is a plain
        // 1:1 TAA resolve. Idempotent — frees existing images first. Resets
        // history-valid to false (the freshly-allocated slots are undefined).
        void createImages(uint32_t inWidth, uint32_t inHeight,
                           uint32_t outWidth, uint32_t outHeight);
        void destroyImages();

        // Rewrite all descriptor sets. Caller supplies the external view
        // sources from the raster G-buffer pass + the swapchain. Must be
        // called after createImages + after the raster G-buffer has been
        // allocated (its views must be valid). Both per-frame arrays are
        // indexed by frame-in-flight slot; swapchain views by swapchain
        // image index.
        struct DescriptorWriteInputs {
            VkSampler          gbufSampler         = VK_NULL_HANDLE;
            const VkImageView* gbufMotionPerFrame  = nullptr;// [framesInFlight]
            const VkImageView* gbufIdsPerFrame     = nullptr;// [framesInFlight]
            const VkImageView* gbufDepthPerFrame   = nullptr;// [framesInFlight] (prev slot → depth disocclusion)
            const VkImageView* swapchainViews      = nullptr;// [imageCount]
        };
        void rewriteDescriptors(const DescriptorWriteInputs& inputs);

        // Per-frame dispatch. Records barrier on input/history images, binds
        // pipeline + descriptor set + push constants, dispatches over the
        // OUTPUT extent in 8×8 groups (each thread reconstructs one full-res
        // pixel; the input may be lower-res). Auto-flips history-valid to
        // true after the first dispatch.
        // When `sharpen` is true the temporal resolve skips its swapchain
        // write and a post-resolve RCAS pass (sharpenAmount ~0.2–0.6) reads
        // the resolved frame back from the history slot and writes the
        // sharpened result to the swapchain instead.
        // `mblurShutter` > 0 inserts a motion-blur stage between the resolve
        // and the swapchain (McGuire 2012 tile-max + reconstruction), reading
        // the resolved history slot + the G-buffer motion vectors. The value
        // is the shutter open fraction of the frame interval (0.5 = a 180°
        // shutter). Chain: resolve → motion blur → optional RCAS. Full-frame
        // only — silently skipped for split-screen panes (dstX/dstY or
        // phys != region). History stays UNBLURRED (the blur never feeds
        // back into next frame's temporal accumulation).
        // `dtFrames` = this frame's duration in reference frames (dt · 90 fps,
        // clamped [1, 6] by the caller; 1 at high fps). The shader scales its
        // per-frame temporal constants (deviation-streak ramp, soft-clip rate)
        // by it so ghost decay is constant in wall-clock time, not frames.
        // Split-screen: the pane content is rendered region-sized AT THE IMAGE
        // ORIGIN of the (full-size) input/history textures. inWidth/inHeight and
        // outWidth/outHeight are the PANE (region) sizes; physInW/H and
        // physOutW/H are the full texture sizes (for UV normalisation); dstX/dstY
        // offset the swapchain write to the pane's screen position. Defaults
        // (phys = 0, dst = 0) reproduce the full-frame 1:1 behaviour exactly.
        //
        // `jitterTexX/Y` = this frame's Halton sub-pixel jitter in RENDER
        // TEXELS (the raw jx/jy the raster projection was offset by — NOT
        // clip units), or (0, 0) when the raster is unjittered (MSAA mode,
        // event camera). The resolve uses it to re-anchor every current-
        // frame read at the output pixel's UNJITTERED center — without this
        // the composed output translates with the 8-phase jitter pattern
        // (the systemic "everything shakes", measured ±0.9 px global shift).
        // Carried in its own push-constant slots at offsets 104/108 (which
        // double as the padding that keeps depthLin 16-byte-aligned).
        void recordResolve(VkCommandBuffer cb,
                           uint32_t frame,
                           uint32_t imageIndex,
                           uint32_t inWidth,
                           uint32_t inHeight,
                           uint32_t outWidth,
                           uint32_t outHeight,
                           float blendAlpha,
                           float dtFrames,
                           bool sharpen,
                           float sharpenAmount,
                           const float* skyReproj,
                           uint32_t dstX = 0,
                           uint32_t dstY = 0,
                           uint32_t physInW = 0,
                           uint32_t physInH = 0,
                           uint32_t physOutW = 0,
                           uint32_t physOutH = 0,
                           const float* depthLin = nullptr,// 4 floats: reverse-Z viewZ linearization (A,B,C,D)
                           float mblurShutter = 0.f,
                           float jitterTexX = 0.f,
                           float jitterTexY = 0.f);

        // Denoise writes its output here when TAA is active (replaces the
        // direct-to-swapchain write of non-TAA mode).
        [[nodiscard]] VkImageView inputView(uint32_t frame) const {
            return inputImagesPP_[frame].view;
        }
        [[nodiscard]] VkImage inputImage(uint32_t frame) const {
            return inputImagesPP_[frame].image;
        }

        // History images — accessed for inter-frame barriers in the caller's
        // pre-RT block (TAA writes them, denoise / next-frame TAA reads).
        [[nodiscard]] VkImage historyImage(uint32_t slot) const {
            return historyImagesPP_[slot].image;
        }
        [[nodiscard]] VkImageView historyView(uint32_t slot) const {
            return historyImagesPP_[slot].view;
        }
        // Whole-struct access (image + extent + format) for the determinism
        // audit's host readback (VulkanRenderer::readTaaDebugImages).
        [[nodiscard]] const Image2D& inputImage2D(uint32_t frame) const {
            return inputImagesPP_[frame];
        }
        [[nodiscard]] const Image2D& historyImage2D(uint32_t slot) const {
            return historyImagesPP_[slot];
        }
        // The history slot THIS frame-in-flight's recordResolve call writes
        // (rewriteDescriptors' writeSlot formula) — a pure function of the
        // frame-in-flight index, so callers can precompute which slot to bind
        // as a downstream reader (PostComposite in HDR mode) without waiting
        // for recordResolve to run first.
        [[nodiscard]] static uint32_t writeSlotFor(uint32_t frame) { return frame & 1u; }

        // First-frame history is undefined. Caller sets this to false on
        // resetAccumulation; the next recordResolves use alpha=1 so we don't
        // bleed garbage into history. TWO of them, not one: the history
        // ping-pongs across two images by frame parity, so a single forced
        // resolve rewrites one slot and leaves the other, written before the
        // invalidation, to be read by the resolve after it. The sensor
        // determinism audit measured exactly that: exact on the frame after a
        // reset, then a residue from the third frame on.
        void invalidateHistory() {
            historyValid_ = false;
            historyInvalidResolves_ = 2;
        }
        [[nodiscard]] bool historyValid() const { return historyValid_; }

        // HDR-mode finalize: reads PostComposite's hdrOut_ (display-extent
        // bgra8 — PostComposite ran the tone map at display res but can't
        // write the swapchain itself, see PostComposite.hpp) and either
        // RCAS-sharpens it into the swapchain (sharpen) or plain-copies it
        // (!sharpen). Reuses this pass's existing swapchain-indexed RCAS
        // descriptor infrastructure — the same shape a downstream RCAS
        // already reads the (LDR-mode) history slot through. Call must be
        // preceded by rewritePostFinalizeDescriptors (after PostComposite's
        // resizeHdrOutput) so hdrOutView is valid.
        void recordPostFinalize(VkCommandBuffer cb, uint32_t frame, uint32_t imageIndex,
                                uint32_t width, uint32_t height,
                                bool sharpen, float sharpenAmount);

        // Rewrite the descriptor sets recordPostFinalize uses. `hdrOutPerFrame`
        // / `hdrOutImagePerFrame` are PostComposite::hdrOutView / hdrOutImage
        // per frame-in-flight (the image handles are needed for the plain-
        // copy path's vkCmdCopyImage, which doesn't go through descriptors).
        // Call after PostComposite::resizeHdrOutput (or its own resize) and
        // after rewriteDescriptors (shares this pass's sampler + swapchain
        // views/images).
        void rewritePostFinalizeDescriptors(const VkImageView* hdrOutPerFrame,
                                            const VkImage* hdrOutImagePerFrame,
                                            const VkImageView* swapchainViews,
                                            const VkImage* swapchainImages);

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       imageCount_;
        uint32_t       framesInFlight_;

        // Input image per frame-in-flight (denoise's target) — sized to the
        // deferred RENDER extent. BGRA8_UNORM to match denoise.comp's
        // rgba8 output and the swapchain channel order.
        std::vector<Image2D> inputImagesPP_;
        // History ping-pong — sized to the OUTPUT (swapchain) extent, so it
        // accumulates the temporal upsampler's reconstructed full-res image.
        // RGBA16F (higher precision than the rgba8 input) so the running
        // mix() doesn't re-quantize to uint8 each frame, which produced
        // visible iso-luminance "lines" on smooth specular surfaces.
        std::array<Image2D, 2> historyImagesPP_{};
        VkSampler sampler_ = VK_NULL_HANDLE;

        VkDescriptorSetLayout dsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_       = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> descSets_;

        // Post-resolve RCAS sharpen (folded into this pass). Reads the just-
        // written history slot (the resolved frame) and writes the swapchain.
        VkDescriptorSetLayout rcasDsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      rcasPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            rcasPipe_       = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> rcasSets_;

        // Post-resolve motion blur (folded into this pass, like RCAS).
        // TileMax finds each 32px tile's dominant velocity; reconstruction
        // gathers the resolved frame along it. When RCAS is also active the
        // blur writes mblurOut_ and RCAS reads that instead of the history
        // slot; otherwise the blur writes the swapchain directly.
        std::vector<Image2D> tileMax_; // [framesInFlight] tilesX×tilesY rg16f
        std::vector<Image2D> mblurOut_;// [framesInFlight] output extent bgra8
        uint32_t tilesX_ = 0, tilesY_ = 0;
        VkPipelineLayout tilemaxPipeLayout_ = VK_NULL_HANDLE;// reuses rcasDsLayout_ (sampler@0 + storage@1)
        VkPipeline       tilemaxPipe_       = VK_NULL_HANDLE;
        VkDescriptorSetLayout mblurDsLayout_   = VK_NULL_HANDLE;// samplers@0-2 + storage@3
        VkPipelineLayout      mblurPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            mblurPipe_       = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> tilemaxSets_;  // [frame]: motion → tileMax
        std::vector<VkDescriptorSet> mblurSwapSets_;// [frame×image]: history+motion+tileMax → swapchain
        std::vector<VkDescriptorSet> mblurOutSets_; // [frame]: history+motion+tileMax → mblurOut
        std::vector<VkDescriptorSet> rcasMbSets_;   // [frame×image]: mblurOut → swapchain

        // HDR-mode finalize (recordPostFinalize): RCAS reads PostComposite's
        // hdrOut_ and writes the swapchain — reuses rcasDsLayout_/rcasPipe_,
        // just a different descriptor-set family (a different sampled
        // source than rcasSets_/rcasMbSets_, which read the history slot).
        // The !sharpen path is a plain vkCmdCopyImage — no pipeline needed,
        // hence the raw image handles cached alongside the descriptor sets.
        std::vector<VkDescriptorSet> postFinalizeRcasSets_;// [frame×image]: hdrOut → swapchain
        std::vector<VkImage> postFinalizeSrcImage_;// [framesInFlight]
        std::vector<VkImage> postFinalizeDstImage_;// [imageCount]

        bool historyValid_ = false;
        uint32_t historyInvalidResolves_ = 2;// forced-alpha resolves still owed

        // Internal helpers.
        Image2D createStorageSampledImage(uint32_t w, uint32_t h, VkFormat format,
                                          const char* label);
        void    transitionFreshImage(VkImage img);
        void    createPipeline();
        void    createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_TAA_RESOLVE_HPP
