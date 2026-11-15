// CoreState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the canvas and its size, the clear colour, viewport, scissor and split-screen pane, and the Vulkan
// context with the frame-local transient memory pools everything else allocates from.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_CORESTATE_HPP
#define THREEPP_VULKAN_STATE_CORESTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct CoreState {
        Canvas& canvas;
        WindowSize size;
        // Canvas size render() last responded to. The resize predicate compares
        // against THIS, not `size`: `size` is pinned to the swapchain extent,
        // and the platform can grant a different extent than the canvas asked
        // for (Windows enforces a minimum window width on the hidden-window
        // headless fallback), so asked-for vs granted can disagree permanently.
        WindowSize lastCanvasSize;
        Color clearColor{0.f, 0.f, 0.f};
        float clearAlpha = 1.f;
        Vector4 viewport;
        Vector4 scissor;
        bool scissorTest = false;
        bool autoClear_ = true;// mirrored from Renderer::autoClear each render()

        // Split-screen (Increment 0): the primary deferred-render pane is
        // clipped to the scissor sub-rect. The deferred pipeline renders the
        // pane region-sized AT THE IMAGE ORIGIN (gbuf/shade/denoise/bloom);
        // only the final TAA write is offset to the scissor position. All
        // default to the full frame, so the single-scene path is
        // byte-identical when scissorTest is off.
        VkExtent2D regionRenderExt_{};// render-extent-space pane size
        VkExtent2D regionSwapExt_{};  // swapchain-space pane size (TAA output)
        int32_t    regionDstX_ = 0;   // swapchain write offset (image space)
        int32_t    regionDstY_ = 0;

        std::unique_ptr<VulkanContext> ctx;
        // Device memory shared by frame-local scratch images (TransientPool):
        // declared right after ctx so it outlives every pass that binds images
        // into it and dies before the allocator. transientPool_ holds the
        // frame-local images — uv, sceneHdr, the filter / post / tail scratch,
        // the shadow maps (ShadowMapPass) —
        // each with the span of TransientPhases it is live in (images whose
        // spans do not meet alias each other, and so do frame-in-flight slots
        // and views); gbufMsPool_ holds the
        // multisampled G-buffer, whose only aliasing is across slots (it is in
        // use from the G-buffer pass to the shade, alongside shade scratch).
        std::unique_ptr<vulkan::TransientPool> transientPool_;
        std::unique_ptr<vulkan::TransientPool> gbufMsPool_;

        explicit CoreState(Canvas& c) : canvas(c), size(c.size()), lastCanvasSize(c.size()) {}
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_CORESTATE_HPP
