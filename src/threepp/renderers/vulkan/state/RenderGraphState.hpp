// RenderGraphState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns recording: the frame and compute render graphs, the async-compute pilot, the per-view record states,
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

        // ── The async-compute pilot (THREEPP_VK_ASYNC_PILOT=1) ─────────────
        // plans/vulkan-cpu-and-async-compute-2026-10.md, item 6. With the knob
        // and a compute-only queue family (ctx->asyncComputeQueue()), a full
        // frame goes out as four command buffers:
        //   graphics 1  frame graph: head, deformers (the ocean WITHOUT its
        //               BLAS rebuild), "async.join"; signals S1
        //   graphics 2  frame graph: the G-buffer passes (no wait)
        //   compute     computeGraph_: ocean BLAS rebuild, tlas, probeGI,
        //               clusterBuild, cloudShadow, froxels; waits S1, signals S2
        //   graphics 3  frame graph: the rest, every secondary view, the tail,
        //               the overlay; waits S2, carries the fence and present
        // so the compute work can run under the G-buffer raster. "async.join"
        // is a pass with no commands that declares everything computeGraph_
        // touches (RenderGraph::declareAsPass): its barriers bring those
        // resources into the compute graph's layouts in graphics 1, and order
        // the frame graph's later uses after it. Semaphore rules: a signal
        // covers every command earlier in its queue's submission order and a
        // wait every command later, so compute(N+1) waiting S1(N+1) is behind
        // graphics 3 of frame N (the previous frame's TLAS and probe readers),
        // and graphics 3 waiting S2 is behind compute(N). Off (the default) or
        // without such a family nothing here is created or used.
        vulkan::rg::RenderGraph computeGraph_;
        bool asyncPilot_ = false;// ctx->asyncComputeQueue(), fixed at createCommandResources
        // What moves to the compute queue (THREEPP_VK_ASYNC_PILOT: =1 both,
        // =3 the dispatches only, =4 nothing; see VulkanContext).
        bool asyncMoveAs_ = false;      // ocean BLAS rebuild + TLAS refit
        bool asyncMoveDispatch_ = false;// probeGI, clusterBuild, cloudShadow, froxels
        bool asyncFrame_ = false;// this frame took the four-command-buffer path
        uint32_t asyncSplits_[2] = {0, 0};// first frame-graph pass of graphics 2 and 3
        VkCommandPool computeCmdPool_ = VK_NULL_HANDLE;
        std::array<VkCommandBuffer, kFramesInFlight> computeCmdBuffers_{};
        std::array<VkCommandBuffer, kFramesInFlight> asyncGbufCmdBuffers_{};
        std::array<VkCommandBuffer, kFramesInFlight> asyncTailCmdBuffers_{};
        std::array<VkSemaphore, kFramesInFlight> asyncHeadDone_{};   // S1
        std::array<VkSemaphore, kFramesInFlight> asyncComputeDone_{};// S2

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
