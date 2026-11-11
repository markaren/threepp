// BillboardHdrLayer — ParticleField billboards summed in linear HDR, then
// tone-mapped once together with (or over) the frame.
//
// ── THE PROBLEM ─────────────────────────────────────────────────────────────
// Field billboards are drawn AFTER the temporal resolve / upscaler, so they
// stay outside TAA/DLSS/FSR (see BillboardGlowPass.hpp and plans/particle-
// atmosphere.md, F3 note 1); that placement does not change here. They used to
// be drawn straight onto the 8-bit swapchain, each sprite through its own
// display transform, so the blend unit added (or alpha-blended) values that
// were already tone-mapped and sRGB-encoded. Nothing tone-mapped the SUM: a
// dense cloud of lit sprites scattering toward a warm sun added a few hundred
// encoded values, every channel clamped at 1.0, and the cloud read flat white.
//
// ── THE LAYER ───────────────────────────────────────────────────────────────
// Every visible field is drawn, in field order and slot order and against the
// depth the overlay pass tests against (read-only), into two rgba16f targets
// at the display extent, in LINEAR radiance (the BB_HDR_LAYER variant of
// particlefield_billboard.frag):
//   over  premultiplied radiance + coverage of the alpha-over sprites,
//         blended (ONE, ONE_MINUS_SRC_ALPHA);
//   add   the additive sprites' radiance, blended (ONE, ONE), and dimmed by
//         every alpha-over sprite drawn after it (ZERO, ONE_MINUS_SRC_ALPHA on
//         that target).
// An additive sprite leaves `over` untouched (empty write mask there).
// Per-attachment blend state needs independentBlend, which VulkanContext
// always enables.
//
// ── THE COMPOSITE ───────────────────────────────────────────────────────────
//   • Upscaler path (DLSS/FSR, post_composite.comp's HDR mode): the layer is
//     drawn before the post composite, which blends it into the upscaled
//     scene in linear HDR, L = S * (1 - a_over) + over.rgb + add.rgb, and
//     tone-maps the result once. Nothing sprite-related is drawn later.
//   • Built-in TAA path, and secondary views: the post stack has already
//     tone-mapped the frame, so recordComposite's two fullscreen draws fold
//     the layer in display space: `over` tone-mapped as its mean colour and
//     alpha-blended, then `add` tone-mapped once and screen-blended
//     (dst + src * (1 - dst)).
//
// ── DEPTH ───────────────────────────────────────────────────────────────────
// The sprites test against the overlay pass's own one-sample depth where they
// can. Two cases need the layer's own copy (recordDepthCopy): a multisampled
// overlay, whose depth a one-sample pass cannot attach (the samples are
// reduced to the farthest), and a splat depth-stamp frame, whose stamp the
// renderer draws into the copy so the clouds hide the sprites behind them
// without the overlay's own buffer changing before the overlay pass.
//
// ── DESCRIPTOR DISCIPLINE ───────────────────────────────────────────────────
// The targets are per frame in flight and so are the sets that name them; all
// are allocated once from this class's own pool and written when the images
// are (re)allocated (a field appears, a resize: in the prepare window), never
// per frame (VUID-03047), and never from the overlay's shared per-frame
// texture pool, whose reset trap the F4 notes record. The depth-copy set names
// an image this class does not own and is rewritten only when that view's
// serial changes, as BillboardGlowPass does for its reduce set.

#ifndef THREEPP_VULKAN_BILLBOARD_HDR_LAYER_HPP
#define THREEPP_VULKAN_BILLBOARD_HDR_LAYER_HPP

#include "threepp/renderers/vulkan/RenderGraph.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    // Everything the layers share and nothing per view: the two sprite
    // pipelines, the display-space composite in its two render-pass shapes
    // (the primary's overlay pass, a secondary view's colour target), the depth
    // copy, their layouts and the sampler. Built ONCE, with the renderer's
    // other overlay pipelines, so the first frame with a field compiles
    // nothing.
    class BillboardHdrLayerPipelines {

    public:
        static constexpr VkFormat kColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

        // `spriteLayout`: the field billboard pipeline layout (the sprite
        // texture set + the 128 B push block), so recordFieldBillboards binds
        // and pushes exactly as for the glow leg. `colorFormat`: the
        // swapchain's (a secondary view's colour target has the same).
        // `overlaySamples`: the overlay pass's sample count.
        BillboardHdrLayerPipelines(VulkanContext& ctx, VkPipelineLayout spriteLayout, VkFormat colorFormat,
                                   VkSampleCountFlagBits overlaySamples);
        ~BillboardHdrLayerPipelines();
        BillboardHdrLayerPipelines(const BillboardHdrLayerPipelines&)            = delete;
        BillboardHdrLayerPipelines& operator=(const BillboardHdrLayerPipelines&) = delete;

    private:
        friend class BillboardHdrLayer;
        VulkanContext& ctx_;
        VkSampler      sampler_ = VK_NULL_HANDLE;

        VkPipeline spriteAdd_  = VK_NULL_HANDLE;
        VkPipeline spriteOver_ = VK_NULL_HANDLE;

        VkDescriptorSetLayout compositeDsLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout      compositeLayout_   = VK_NULL_HANDLE;
        // [0] = inside the primary's overlay pass (its depth attachment, its
        // sample count); [1] = onto a secondary view's colour target (no depth,
        // one sample).
        VkPipeline compositeOver_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        VkPipeline compositeAdd_[2]  = {VK_NULL_HANDLE, VK_NULL_HANDLE};

        VkDescriptorSetLayout copyDsLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout      copyLayout_   = VK_NULL_HANDLE;
        VkPipeline            copyPipe_     = VK_NULL_HANDLE;// sampler2D
        VkPipeline            copyPipeMs_   = VK_NULL_HANDLE;// sampler2DMS

        void createSpritePipelines(VkPipelineLayout spriteLayout);
        void createCompositePipelines(VkFormat colorFormat, VkSampleCountFlagBits overlaySamples);
        void createCopyPipelines();
    };

    // One view's layer: its targets, per frame in flight, and the sets that
    // name them.
    class BillboardHdrLayer {

    public:
        static constexpr VkFormat kColorFormat = BillboardHdrLayerPipelines::kColorFormat;
        static constexpr VkFormat kDepthFormat = BillboardHdrLayerPipelines::kDepthFormat;

        // `secondaryView`: composites onto a secondary view's colour target
        // rather than inside the primary's overlay pass.
        BillboardHdrLayer(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight,
                          const BillboardHdrLayerPipelines& pipes, bool secondaryView);
        ~BillboardHdrLayer();
        BillboardHdrLayer(const BillboardHdrLayer&)            = delete;
        BillboardHdrLayer& operator=(const BillboardHdrLayer&) = delete;

        // Allocate (or re-allocate) the targets for an extent and rewrite the
        // sets that name them; `ownDepth` adds the layer's own one-sample depth
        // (only a multisampled overlay or a splat-stamp scene needs it).
        // Idempotent for an unchanged request. All new images are made ready
        // in ONE submit. False when the extent is degenerate.
        bool ensureImages(uint32_t width, uint32_t height, bool ownDepth);
        void destroyImages();

        [[nodiscard]] VkExtent2D extent() const { return {width_, height_}; }
        [[nodiscard]] bool       hasOwnDepth() const { return !depth_.empty() && depth_[0].image != VK_NULL_HANDLE; }

        // The sprite pipelines: additive fields and alpha-over fields.
        [[nodiscard]] VkPipeline additivePipeline() const { return pipes_.spriteAdd_; }
        [[nodiscard]] VkPipeline alphaOverPipeline() const { return pipes_.spriteOver_; }

        // ── The layer's own depth (hasOwnDepth) ─────────────────────────────
        // Begins a depth-only render-pass instance on this frame's own depth
        // and copies `srcView` (DEPTH_STENCIL_READ_ONLY_OPTIMAL, the overlay's
        // depth at this extent, multisampled when `srcMultisampled`) into it.
        // The caller may draw more depth (the splat stamp) and then ends the
        // instance; depthWritten() orders those writes before the sprites'
        // test.
        void recordDepthCopy(VkCommandBuffer cb, uint32_t frame, VkImageView srcView, bool srcMultisampled);
        static void depthWritten(VkCommandBuffer cb);
        [[nodiscard]] VkImageView ownDepthView(uint32_t frame) const { return depth_[frame].view; }

        // Begin the render-pass instance the sprites draw in: both targets
        // CLEARED, `depthView` attached read-only in `depthLayout`, the given
        // viewport and scissor (the overlay pass's pane). The caller draws and
        // ends it.
        void beginSprites(VkCommandBuffer cb, uint32_t frame, VkImageView depthView, VkImageLayout depthLayout,
                          const VkViewport& viewport, const VkRect2D& scissor) const;

        // The two display-space composite draws (TAA path, secondary views),
        // recorded INSIDE the caller's render-pass instance under its viewport
        // and scissor. `exposure` is the frame's exposure divided by its
        // pre-exposure (the layer is stored pre-exposed); `toneMapMode` and
        // `grade` are the display transform the frame behind went through.
        struct CompositePC {
            float    exposure;
            uint32_t toneMapMode;
            uint32_t part;
            // Layer pixels per target pixel: 1 (or 0) when the layer is at the
            // target's size, < 1 for a view that renders below its output size.
            float    layerScale;
            uint8_t  grade[96];// vulkan::OverlayGradeGpu, opaque here
        };
        static_assert(sizeof(CompositePC) == 112, "CompositePC drifted from particlefield_hdr_layer.frag");
        void recordComposite(VkCommandBuffer cb, uint32_t frame, CompositePC pc) const;

        [[nodiscard]] VkImageView overView(uint32_t frame) const { return over_[frame].view; }
        [[nodiscard]] VkImageView addView(uint32_t frame) const { return add_[frame].view; }

        // Render-graph declarations. declareSprites: the sprite pass writes
        // both targets as colour attachments, and the own depth when it copies
        // into it. declareRead: a later pass samples both targets at `stage`.
        void declareSprites(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame, bool ownDepth) const;
        void declareRead(rg::RenderGraph& g, rg::PassBuilder& pass, uint32_t frame,
                         VkPipelineStageFlags2 stage) const;

    private:
        VulkanContext&                    ctx_;
        VkCommandPool                     cmdPool_;
        uint32_t                          framesInFlight_;
        const BillboardHdrLayerPipelines& pipes_;
        uint32_t                          variant_;// index into the pipelines' composite pairs
        uint32_t                          width_ = 0, height_ = 0;

        std::vector<Image2D> over_; // [framesInFlight] alpha-over layer
        std::vector<Image2D> add_;  // [framesInFlight] additive layer
        std::vector<Image2D> depth_;// [framesInFlight] own depth, kDepthFormat, GENERAL; empty = none

        VkDescriptorPool             descPool_ = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> compositeSets_; // [framesInFlight]
        std::vector<VkDescriptorSet> copySets_;      // [framesInFlight]
        std::vector<uint64_t>        copySetSerials_;// imageViewSerial each copy set holds

        Image2D createImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
                            VkImageAspectFlags aspect, const char* label);
        void    transitionFresh(const std::vector<std::pair<VkImage, VkImageAspectFlags>>& images);
        void    createDescriptors();
        void    rewriteDescriptors();
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_BILLBOARD_HDR_LAYER_HPP
