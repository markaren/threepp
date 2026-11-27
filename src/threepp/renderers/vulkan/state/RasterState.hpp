// RasterState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the G-buffer raster: its render passes, pipelines and layouts (single- and multisampled), the MSAA
// resolve and dummies, the G-buffer sampler and the material-raster toggles.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_RASTERSTATE_HPP
#define THREEPP_VULKAN_STATE_RASTERSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct RasterState {
        // Normal-map vMF/Toksvig specular-AA toggle (deferred G-buffer raster
        // path only — see setNormalMapToksvig). Threaded to gbuffer.frag via
        // the raster CameraUbo's prevJitter.z, which is otherwise dead (only
        // prevJitter.xy is read, by nothing in gbuffer.frag/.vert — see
        // uploadRasterCameraUbo) so no descriptor/PC layout change is needed.
        // ON by default: it is a no-op on materials without a normal map and at
        // mip 0 (nLen ~= 1), and strictly reduces normal-map minification
        // shimmer otherwise.
        bool     normalMapToksvig_ = true;
        bool     stochasticTilingIntegerHash_ = false;// setStochasticTilingIntegerHash

        // MSAA G-buffer dominant-sample resolve — see vulkan/GbufResolve.
        // Lazily constructed the first time gbufMsaaSamples_ > 1 (mirrors
        // debugResolvePipeline_'s on-demand pattern); stays null at the
        // msaa=1 default so that path allocates nothing extra.
        std::unique_ptr<vulkan::GbufResolve> gbufResolve_;

        VkRenderPass rasterGbufRenderPass = VK_NULL_HANDLE;
        // MSAA render pass, keyed by sample count (2 or 4). Only the pass
        // matching gbufMsaaSamples_ is ever created; the other stays
        // VK_NULL_HANDLE. Kept separate from rasterGbufRenderPass (the 1×
        // path) so the default (msaa=1) code path is 100% untouched.
        VkRenderPass rasterGbufRenderPassMS = VK_NULL_HANDLE;
        VkPipeline   rasterGbufIndirectPipelineMS = VK_NULL_HANDLE;
        VkPipeline   rasterGbufDecalPipelineMS    = VK_NULL_HANDLE;
        VkPipeline   rasterGbufParticlePipelineMS = VK_NULL_HANDLE;
        // Sample count backing the MS pipelines/render pass/images above (0
        // until first created; tracks which count they were built for, so a
        // 2→4 change knows to tear down and rebuild rather than reuse).
        VkSampleCountFlagBits gbufMsaaBuiltSamples_ = VK_SAMPLE_COUNT_1_BIT;
        // 1x1 dummy MS images (5, mirroring normal/depth/ids/uv/albedo) bound
        // to deferred_shade.comp's dispatch-B sampler2DMS bindings when
        // gbufMsaaSamples_ == 1 — sampler2DMS is a distinct SPIR-V type from
        // sampler2D, so (unlike other "1x1 dummy" bindings elsewhere) this
        // can't reuse a single-sample dummy; a real multisample image is the
        // minimum valid stand-in (2x where the device has it, else 4x — see
        // ensureGbufDummyMS). Created once, lazily, on first use.
        std::array<Image2D, 5> gbufDummyMS_{};
        bool gbufDummyMSCreated_ = false;

        VkDescriptorSetLayout rasterDsLayout       = VK_NULL_HANDLE;
        VkPipelineLayout      rasterPipelineLayout = VK_NULL_HANDLE;
        // The G-buffer pipeline: gbuffer_indirect.vert with bindless vertex
        // pulling, zero vertex input bindings, the per-frame DrawInfo SSBO at
        // binding 4; the opaque buckets draw through it in 1-4
        // vkCmdDrawIndirect calls (recordRasterGbufPass), decals and
        // particle fields through the variants below. Non-null also means
        // the raster G-buffer path is ready.
        VkPipeline            rasterGbufIndirectPipeline = VK_NULL_HANDLE;
        // Decal variant of the indirect pipeline (bucket [3], drawn last):
        // depth-write OFF, attachments 0-3 write-masked to zero, attachment 4
        // (albedo+metalness) alpha-blended with an RGB-only write mask — the
        // decal lerps its albedo over the receiver's, everything else (normal,
        // ids, motion, depth, metalness) stays the receiving surface's.
        VkPipeline            rasterGbufDecalPipeline = VK_NULL_HANDLE;
        // ParticleField variant: particlefield_gbuf.vert over the SAME render
        // pass, layout, fragment stage and blend/depth state as the indirect
        // pipeline. It exists as a sibling rather than a flag because it breaks
        // the one-instance-per-draw contract gbuffer_indirect.vert documents —
        // a field is one draw with instanceCount = the device-side live count,
        // so gl_InstanceIndex is the particle index and the DrawInfo index
        // arrives by push constant.
        VkPipeline            rasterGbufParticlePipeline = VK_NULL_HANDLE;

        // Nearest-filter sampler used by the deferred shade to read gbuffer
        // attachments. Nearest avoids bilinear smearing of normal/motion/ids
        // at silhouettes — primary visibility from raster is already exact-pixel.
        VkSampler gbufSampler_ = VK_NULL_HANDLE;

        // Raster G-buffer MSAA sample count (1/2/4). 1 = today's single-sample
        // path, byte-identical output, default. See setGbufferMsaa. Reallocation
        // (render pass + pipelines + MS images) is render-extent-resource work,
        // so it reuses the exact same idle-gate as renderScale.
        uint32_t gbufMsaaSamples_        = 1;

        // MSAA dispatch B (per-sample shading at complex/edge pixels) master
        // switch. OFF by default: measured on rock_flicker (msaa=4), dispatch
        // B's fallback variant (single largest non-dominant cluster, cheap
        // env-only diffuse — no RT gather/reflections/ReSTIR, to keep cost
        // proportional to edge-pixel count) made the static-camera flicker
        // metric WORSE (mean ~4600-4800 px/frame) than Phase 1 (dominant-
        // sample resolve) alone (~3700-3900), which was itself already below
        // the msaa=1 baseline (~4000-4030). Root cause: dispatch B's cheap
        // shading model disagrees with dispatch A's full model enough that
        // the disagreement is itself a new, comparably-sized noise source at
        // every edge pixel — trading one flicker mechanism for another
        // instead of removing it. The code compiles, is wired end-to-end,
        // and is architecturally sound (see deferred_shade.comp's shadeMode
        // branches) for a future attempt with a closer-matching per-sample
        // shading model; it just isn't a net win yet, so it stays off.
        // ON by default: under the UNJITTERED msaa raster (see
        // uploadRasterCameraUbo) dispatch B no longer regresses temporal
        // stability (static 23 vs 11 px/frame on the rock harness), and with
        // the corrected coverage accounting (dispatch A blends sky-minority
        // coverage itself; B fills the geometry-minority weight it reserves)
        // it supplies the spatial edge AA the dominant-pick resolve alone
        // lacks. Only consulted when gbufMsaaSamples_ > 1.
        bool gbufShadeBEnabled_ = true;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_RASTERSTATE_HPP
