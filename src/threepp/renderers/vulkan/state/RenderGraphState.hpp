// RenderGraphState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns recording: the frame render graph, the per-view record states,
// CPU-parallel recording (record-apart buffers, the secondary views' prep) and the graph diagnostics.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_RENDERGRAPHSTATE_HPP
#define THREEPP_VULKAN_STATE_RENDERGRAPHSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct RenderGraphState {
        // THREEPP_VK_CPU_RECORD: secondary command buffers that a pass records
        // apart, on the pool, and the primary executes in its place. Per
        // frame-in-flight slot, one transient pool per buffer, each holding its
        // one buffer: a pool is externally synchronized, and each recording
        // thread then owns the pool it records from. Handed out on the frame
        // thread; the slot's pools are reset at its first hand-out of a frame,
        // its fence having been waited.
        struct RecordApartBuffer {
            VkCommandPool   pool = VK_NULL_HANDLE;
            VkCommandBuffer cb   = VK_NULL_HANDLE;
        };
        std::array<std::vector<RecordApartBuffer>, kFramesInFlight> recordApart_;
        std::array<size_t, kFramesInFlight>                         recordApartUsed_{};
        std::array<uint64_t, kFramesInFlight>                       recordApartSerial_{};

        // The frame's render graph: every view's passes and the record tail,
        // built in full and then executed once (beginDeferredFrame).
        vulkan::rg::RenderGraph frameGraph_;

        // ── Several views, one graph ────────────────────────────────────────
        // Every view's passes are BUILT before any is RECORDED, so the CPU
        // state a view's pass callbacks read at record time — the current
        // view, the pane region, the draw lists buildIndirectDrawData made for
        // its camera, the camera-derived scalars updateCameraUbo and
        // uploadRasterCameraUbo leave in Impl members — would otherwise be the
        // LAST view's. Each view's passes therefore open with a "view" pass
        // (no resource uses) that applies what that view's build left behind,
        // saved into viewRecordStates_[its index in views_] once its passes
        // are built. The record tail opens with the primary's again.
        struct ViewRecordState {
            ViewContext* view = nullptr;
            VkExtent2D   regionRenderExt{}, regionSwapExt{};
            int32_t      regionDstX = 0, regionDstY = 0;
            bool         occlActive = false;
            ViewPrepOut  out;
        };
        std::vector<ViewRecordState> viewRecordStates_;
        // The water's mirror view's, applied by its own "view" pass inside
        // the primary's run of passes (addWaterMirrorPasses).
        ViewRecordState waterMirrorRecordState_;

        std::map<std::string, std::string> rgDumped_;
        std::set<std::string> rgDiagnosticsReported_;
        std::vector<std::string> rgDiagnostics_;// distinct, in the order first seen
        std::uint32_t rgDiagnosticCount_ = 0;   // VulkanRenderer::renderGraphDiagnosticCount()

        vulkan::par::Group        secondaryPrep_;
        bool                      secondaryPrepStarted_ = false;
        std::vector<ViewContext*> secondaryPrepViews_;// in view order
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_RENDERGRAPHSTATE_HPP
