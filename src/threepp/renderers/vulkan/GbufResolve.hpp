// GbufResolve — MSAA raster G-buffer dominant-sample resolve.
//
// VulkanRenderer::setGbufferMsaa(2|4) rasterizes the G-buffer at K samples
// per pixel into MS attachments (RasterGbufImages::{normal,motion,ids,uv,
// albedo,depth}MS in VulkanCoreImpl.hpp). This pass picks, per pixel, the
// sample whose instance id has the most votes among the K live samples
// (ties broken by nearest reversed-Z depth) and writes that sample's
// G-buffer data into the EXISTING single-sample images — the same images
// every consumer (DeferredShade, TaaResolve, the raygen hybrid descriptor
// set, the debug blit) already reads, unchanged. See gbuf_resolve.comp for
// the full per-pixel algorithm and the ids.w metadata packing (dominant
// index + same-cluster mask + complex/edge flag), consumed by a later
// per-sample shading phase.
//
// Depth needs a second, tiny fullscreen-triangle rasterization pass
// (gbuf_resolve_depth.vert/.frag) because compute shaders cannot target a
// depth-aspect image; it re-derives the same winning sample index from the
// ids.w metadata gbuf_resolve.comp already wrote, so the two resolves can
// never disagree.

#ifndef THREEPP_VULKAN_GBUF_RESOLVE_HPP
#define THREEPP_VULKAN_GBUF_RESOLVE_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class GbufResolve {

    public:
        GbufResolve(VulkanContext& ctx, uint32_t framesInFlight);
        ~GbufResolve();
        GbufResolve(const GbufResolve&) = delete;
        GbufResolve& operator=(const GbufResolve&) = delete;

        // Per-frame-in-flight image views. MS views are the raw multisampled
        // raster attachments; the Resolved views are the existing single-
        // sample images every other pass reads. depthResolved is bound as a
        // DEPTH attachment (not sampled) for the depth pass; idsResolved is
        // re-sampled by the depth pass to read back the dominant index the
        // compute pass just wrote.
        struct DescriptorWriteInputs {
            const VkImageView* normalMS  = nullptr;// [framesInFlight]
            const VkImageView* motionMS  = nullptr;
            const VkImageView* idsMS     = nullptr;
            const VkImageView* uvMS      = nullptr;
            const VkImageView* albedoMS  = nullptr;
            const VkImageView* depthMS   = nullptr;
            const VkImageView* normalResolved = nullptr;
            const VkImageView* motionResolved = nullptr;
            const VkImageView* idsResolved    = nullptr;
            const VkImageView* uvResolved     = nullptr;
            const VkImageView* albedoResolved = nullptr;
        };
        void rewriteDescriptors(const DescriptorWriteInputs& in);

        // Dispatch the dominant-sample compute resolve (normal/motion/ids/
        // uv/albedo). Caller must barrier the MS attachments' render-pass
        // write -> COMPUTE_SHADER read (the MSAA render pass's own subpass
        // dependency already does this — see createRasterGbufRenderPassMS)
        // before calling, and barrier the resolve targets' COMPUTE write ->
        // whatever reads them next (recordDepthResolve, then every existing
        // G-buffer consumer) after.
        void recordComputeResolve(VkCommandBuffer cb, uint32_t frame,
                                  uint32_t width, uint32_t height, uint32_t sampleCount);

        // Depth-only fullscreen pass writing the dominant sample's depth
        // into depthResolved (bound as a real D32_SFLOAT depth attachment).
        // Must run AFTER recordComputeResolve (reads its ids.w output) with
        // a compute-write -> fragment-read barrier on idsResolved in
        // between, and needs its own depthResolved UNDEFINED/whatever ->
        // DEPTH_ATTACHMENT_OPTIMAL transition (caller's responsibility,
        // mirroring the overlay depth prepass pattern).
        void recordDepthResolve(VkCommandBuffer cb, uint32_t frame,
                                uint32_t width, uint32_t height,
                                VkImageView depthMSView, VkImageView idsResolvedView,
                                VkImageView depthResolvedView);

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        VkSampler             sampler_       = VK_NULL_HANDLE;// nearest; texelFetch ignores it but MS samplers still need one bound
        VkDescriptorSetLayout dsLayout_      = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_    = VK_NULL_HANDLE;
        VkPipeline            pipe_          = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_      = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        // Depth-resolve fullscreen pass — separate tiny pipeline/layout
        // (own descriptor set: depthMS + idsResolved), dynamic rendering
        // (no VkRenderPass/VkFramebuffer needed, matches overlay_depth's
        // approach) targeting depthResolved directly.
        VkDescriptorSetLayout depthDsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      depthPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            depthPipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      depthDescPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> depthSets_;// [framesInFlight]
        VkSampler             depthSampler_    = VK_NULL_HANDLE;

        void createComputePipeline();
        void createDepthPipeline();
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_GBUF_RESOLVE_HPP
