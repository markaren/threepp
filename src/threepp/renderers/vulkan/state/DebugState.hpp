// DebugState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns diagnostics and stats: GPU timings, draw/dynamic-geometry/TLAS stats, ray counts, the hybrid debug
// views and their resolve pipeline.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_DEBUGSTATE_HPP
#define THREEPP_VULKAN_STATE_DEBUGSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct DebugState {
        // THREEPP_VK_DRAW_STATS=N: every N primary frames, one stderr line of
        // what the frame prep walked (drawStatsTick). Diagnostics only. A
        // view that prepares beside the others counts into its own
        // (ViewContext::prepDrawStats), added here once it is done.
        using DrawStats = vulkan::impl::DrawStats;
        DrawStats drawStats_;

        bool     countRays_       = false;
        std::array<Buffer, kFramesInFlight> rayCountBufs_{};    // device-local
        std::array<Buffer, kFramesInFlight> rayCountReadback_{};// host-mapped
        std::array<bool, kFramesInFlight>   rayCountCopied_{};  // the frame recorded its clear and copy
        std::array<uint32_t, kRayCountSlots> lastRayCounts_{};

        VulkanRenderer::DynamicGeomStats dynGeomStats_{};
        VulkanRenderer::TlasStats tlasStats_{};

        // Debug-view resolve (setHybridDebugView): a compute pass that
        // visualizes one G-buffer channel to the swapchain. Replaces the old
        // raw blit, which could not correctly show the signed motion or
        // integer ids attachments. Created lazily on first debug-view frame.
        // Layout moved to VulkanGpuLayouts.hpp.
        using DebugResolvePC = vulkan::impl::DebugResolvePC;
        VkDescriptorSetLayout debugResolveDsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      debugResolvePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            debugResolvePipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      debugResolveDescPool_       = VK_NULL_HANDLE;
        // One descriptor set per frame-in-flight, rewritten every frame the
        // debug view is active (gbuf views are per-frame, the storage target
        // is the acquired swapchain image). A single shared set would be
        // updated while the other in-flight frame still had it bound in a
        // pending command buffer — VUID-vkUpdateDescriptorSets-None-03047,
        // firing per frame on the AOV readback path (render_aov routes
        // through this pass every frame).
        std::array<VkDescriptorSet, kFramesInFlight> debugResolveDescSets_{};

        // ── Per-frame timing instrumentation ─────────────────────────────
        // Managed by vulkan/GpuTimings.{hpp,cpp}. Owns one VkQueryPool per
        // frame-in-flight; exposes begin/end brackets per TimingPass plus CPU
        // record/frame timing. Constructed in the Impl ctor, after ctx is
        // available.
        std::unique_ptr<vulkan::GpuTimings>  gpuTimings_;

        // Set by VulkanRenderer::render(...) right after ensureSceneBuilt
        // and consumed by gpuTimings_->readBack() on the next frame's fence
        // wait so the public getter sees the same frame's CPU + GPU numbers.
        float    pendingCpuEnsureSceneMs_ = 0.f;

        // G-buffer debug-view mode: blit one G-buffer channel onto the
        // swapchain instead of running the deferred shade. Lets us see the
        // raster output before the shade's ray-query integration.
        enum class HybridDebugView : uint32_t {
            Off    = 0,
            Normal = 1,
            Motion = 2,
            Depth  = 3,
            Ids    = 4,
            Albedo = 5,   // raster-first material G-buffer: linear albedo in rgb
        };
        HybridDebugView hybridDebugView_ = HybridDebugView::Off;
        // VulkanRenderer::LightingView as an integer (0 = the lit frame);
        // the primary view's denoise recombine reads it.
        uint32_t lightingView_ = 0;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_DEBUGSTATE_HPP
