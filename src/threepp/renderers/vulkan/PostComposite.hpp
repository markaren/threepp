// PostComposite — the camera/display end of the shared HDR post stack.
//
// The deferred shade / PT resolve write the linear-HDR scene into
// BloomPass::sceneHdr; BloomPass (now pyramid-only) optionally builds the
// blurred bloom buffer from it. This pass closes the HDR path in ONE
// dispatch (post_composite.comp):
//
//   sceneHdr (+ bloom) → exposure → white balance (Bradford CAT, linear)
//     → tone map (incl. AgX) → sRGB encode → grading LUT (encoded domain)
//     → TAA input image
//
// Split out of BloomPass because tone mapping / camera response is not a
// bloom concern (the old layering violation): BloomPass owns the pyramid,
// PostComposite owns everything camera/display. With white balance and the
// grade inactive (the defaults) the dispatch reproduces the old
// composite.comp output byte-exactly.
//
// White balance: setWhiteBalance(temperatureK, tint) — estimates the scene
// illuminant's chromaticity on the Planckian locus (tint offsets
// perpendicular-ish via y), then Bradford-adapts scene-white → D65 in
// linear sRGB. 6500 K / 0 tint = identity (flag off, zero shader cost).
//
// Colour grade: setColorGrade(lift/gamma/gain, saturation, contrast) —
// baked on the CPU into a 33³ rgba16f LUT in the sRGB-ENCODED
// display-referred domain (the conventional space for grade wheels;
// perceptually uniform LUT coordinates), uploaded via a one-shot submit on
// change (UI-rate, not per-frame). Defaults = identity (flag off, LUT
// never sampled).
//
// HDR-MODE INPUT (VulkanRendererCore::setTaaHdrInput): when the TAA resolve
// runs BEFORE this pass (consuming linear HDR directly, see TaaResolve.cpp),
// PostComposite's binding 0 instead reads the resolve's already-bloomed,
// already-exposure-rescaled HDR history slot, and this pass runs at DISPLAY
// resolution (not the render extent) doing ONLY exposure/WB/tonemap/grade/
// sRGB — no second bloom add (the caller passes effBloomIntensity <= 0 in
// this mode). The sky-mask sources (gbufImg / rasterIdsTex) still live at
// RENDER resolution, so `srcWidth`/`srcHeight` (recordDispatch) tell the
// shader how to scale its dispatch-space (display) texel index down to the
// mask's native texel grid.
//
// HDR-MODE OUTPUT: this pass can't write the LDR mode's per-frame-in-flight
// TAA-input image (that image is render-extent BGRA8; HDR mode's dispatch is
// display-extent) and writing the swapchain directly would need swapchain-
// image-indexed descriptor sets (this class is frame-in-flight-indexed only,
// matching every other consumer here). Instead HDR mode writes hdrOut_ — an
// owned per-frame-in-flight display-extent BGRA8 scratch — and the caller
// (VulkanCoreImpl.hpp) finalizes it into the swapchain via TaaResolve's
// existing swapchain-indexed RCAS/copy machinery (recordPostFinalize),
// exactly the pattern a downstream RCAS already uses on the LDR history
// slot. setTaaHdrOutput(bool) selects which output binding 3 targets;
// resizeHdrOutput must be called (once, at display-extent) before HDR mode
// is used.

#ifndef THREEPP_VULKAN_POST_COMPOSITE_HPP
#define THREEPP_VULKAN_POST_COMPOSITE_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class PostComposite {

    public:
        PostComposite(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight);
        ~PostComposite();
        PostComposite(const PostComposite&) = delete;
        PostComposite& operator=(const PostComposite&) = delete;

        // Rewrite per-frame descriptor sets. One view per frame-in-flight:
        // the linear-HDR scene, the bloom pyramid's level 0, the G-buffer
        // storage view (solid-bg sky bypass) and the TAA input (output).
        // hdrScenePerFrame (HDR mode only): TaaResolve's resolved HDR history
        // slot (TaaResolve::historyView(TaaResolve::writeSlotFor(f))) —
        // bound to a SEPARATE combined-image-sampler binding (6) from the
        // LDR-mode sceneHdrPerFrame, since the two are different resolutions
        // (render vs display extent) and the shader picks between them via
        // the hdrMode push-constant flag, not a descriptor rewrite.
        struct DescriptorWriteInputs {
            const VkImageView* sceneHdrPerFrame  = nullptr;// [framesInFlight]
            const VkImageView* bloomPerFrame     = nullptr;// [framesInFlight]
            const VkImageView* gbufPerFrame      = nullptr;// [framesInFlight] PT gbuf (.w = id+1)
            const VkImageView* rasterIdsPerFrame = nullptr;// [framesInFlight] raster ids (.x = id+1)
            const VkImageView* taaInputPerFrame  = nullptr;// [framesInFlight]
            const VkImageView* hdrScenePerFrame  = nullptr;// [framesInFlight] HDR-mode-only
        };
        void rewriteDescriptors(const DescriptorWriteInputs& in);

        // (Re)allocate the HDR-mode output scratch at the DISPLAY extent (the
        // swapchain size — independent of renderScale). Idempotent; a no-op
        // if the size didn't change. Must run before the first HDR-mode
        // recordDispatch; call from the same resize path that recreates the
        // swapchain-extent images (TaaResolve::createImages et al).
        void resizeHdrOutput(uint32_t width, uint32_t height);
        [[nodiscard]] VkImageView hdrOutView(uint32_t frame) const { return hdrOut_[frame].view; }
        [[nodiscard]] VkImage     hdrOutImage(uint32_t frame) const { return hdrOut_[frame].image; }

        // ── Camera/display knobs (rare, UI-rate writes) ──────────────────────
        // Scene illuminant estimate: temperatureK on the Planckian locus
        // (1667–25000 K, 6500 = neutral), tint shifts green(−)/magenta(+).
        void setWhiteBalance(float temperatureK, float tint);

        struct ColorGrade {
            float lift[3]  = {0.f, 0.f, 0.f};
            float gamma[3] = {1.f, 1.f, 1.f};
            float gain[3]  = {1.f, 1.f, 1.f};
            float saturation = 1.f;
            float contrast   = 1.f;
        };
        // Bakes + uploads the LUT when the parameters differ from identity
        // (one-shot submit + queue wait — call between frames, not per frame).
        void setColorGrade(const ColorGrade& grade);

        // Records the composite dispatch (leading sceneHdr write→read
        // barrier included). effBloomIntensity is the level-normalized
        // intensity (<= 0 skips the bloom add); exposureBits = the FULL
        // exposure, preExposureBits = the factor already baked into sceneHdr
        // by the shade/resolve (1.0 in legacy mode). skyFromRasterIds: the
        // deferred renderer's sky test reads the raster ids attachment (the
        // PT-style gbuf ping-pong is never written on that path).
        // `srcWidth`/`srcHeight` = the RENDER-extent size of the sky-mask
        // sources (gbufImg/rasterIdsTex) — needed only when width/height (the
        // dispatch extent) differ from them, i.e. HDR mode at renderScale<1.
        // Default 0 ⇒ assumed equal to width/height (today's LDR behaviour,
        // byte-identical). `hdrMode`: reads hdrScenePerFrame (no bloom re-add
        // — pass effBloomIntensity <= 0) and writes hdrOut_ instead of the
        // LDR-mode TAA-input target.
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t width, uint32_t height,
                            uint32_t toneMapping, uint32_t exposureBits,
                            uint32_t preExposureBits,
                            bool bgIsSolidColor, float effBloomIntensity,
                            bool skyFromRasterIds,
                            uint32_t srcWidth = 0, uint32_t srcHeight = 0,
                            bool hdrMode = false);

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;

        VkSampler sampler_        = VK_NULL_HANDLE;// LINEAR clamp — bloom upsample + LUT
        VkSampler nearestSampler_ = VK_NULL_HANDLE;// raster ids (uint texelFetch)
        Image2D   gradeLut_{};                     // 33³ rgba16f (3D image), GENERAL layout

        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        // HDR-mode output scratch (display extent, bgra8 — matches the
        // swapchain channel order, same "no format qualifier" convention as
        // taa_resolve.comp's swapchain store). See the header comment above.
        std::vector<Image2D> hdrOut_;// [framesInFlight]
        uint32_t hdrOutW_ = 0, hdrOutH_ = 0;

        // White balance state: row-major 3×3 in linear sRGB (identity when
        // inactive; pushed as 3 vec4 rows).
        bool  wbActive_ = false;
        float wbMat_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        bool  lutActive_ = false;

        static constexpr uint32_t kLutSize = 33;

        void createLutImage();
        void uploadLut(const ColorGrade& grade);
        void createPipeline();
        void createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_POST_COMPOSITE_HPP
