// FrameLoopState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the frame loop: swapchain images, command pool and buffers, semaphores and fences, the frame state
// machine and clocks, the retire queue, the overlay callback, the deferred-apply requests, and the frame
// outputs (scene capture, frame interop).
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct FrameLoopState {
        // Wall-clock time of the last beginDeferredFrame() call, used to compute
        // dt for the onBeginDeferredFrame() hook (auto-exposure, etc.).
        double lastFrameTime_ = 0.0;

        // A windowed canvas stays hidden until its first frame has been
        // presented (see the constructor of VulkanRenderer); once.
        bool   windowRevealed_ = false;

        // ── Deterministic frame clock ───────────────────────────────────
        // The one time source every frame-path wall-clock read goes through.
        // Negative (the default) = wall clock, behaviour unchanged. When the
        // app drives it (VulkanRenderer::setSimTime, once per frame before
        // render()), TAA/DLSS/FSR frame-dt, the shade's timeSec, foam decay,
        // deform timestamps and the cloud clock all advance on simulation
        // time, which is what makes two same-seed runs produce the same
        // pixels (wall time in the TAA blend alpha alone is enough to make
        // every run diverge from the first history frame onward).
        double simTimeSec_ = -1.0;
        // A headless canvas (ctx->presentSuppressed()) with no setSimTime has
        // no wall clock worth reading: a stepped loop renders as fast as the
        // GPU allows, so the TAA blend and the FSR/DLSS dt would think it runs
        // at hundreds of fps and ghost. advanceHeadlessClock steps this clock
        // 1/60 s per scene-building render() instead (0 on the first), and
        // says so once. Negative = not in use. setSimTime always wins.
        double headlessClockSec_ = -1.0;

        // Swapchain image count, sampled once in createSwapchainDependents and
        // never refreshed: pinned for the renderer's lifetime, not merely the
        // current swapchain's. Its one consumer is the primary view's
        // TaaResolve, which sizes its descriptor pool and set vectors from it
        // at construction and indexes them `frame * imageCount + imageIndex`
        // forever after — refreshing this scalar on a swapchain recreate would
        // move the index while the arrays stayed their allocated size.
        //
        // A recreate producing a different count is legal in principle but
        // cannot happen here: every input to the negotiation (vsync_,
        // presentSuppressed_, the surface itself) is fixed at context
        // construction, leaving only caps.min/maxImageCount.
        // recreateSwapchainAndDescriptors verifies that rather than trusting
        // it. If it ever fires, the fix is to rebuild the primary view's
        // TaaResolve there (pool, set vectors and all), not to assign a new
        // value here.
        uint32_t imageCount_ = 0;

        // Per-frame command resources.
        VkCommandPool                                cmdPool = VK_NULL_HANDLE;
        std::array<VkCommandBuffer, kFramesInFlight> cmdBuffers{};
        std::array<VkSemaphore,     kFramesInFlight> imageAvailable{};
        // The stages the submit waits imageAvailable at (endFrame) AND the
        // srcStageMask of every barrier that first touches the acquired image.
        // They must be the same set: a barrier's layout transition is ordered
        // after the semaphore wait only when its first scope includes a waited
        // stage. The UNDEFINED transitions used TOP_OF_PIPE, which is not one,
        // so the presentation engine could still be reading an image this
        // frame was already transitioning (syncval WAR vs vkAcquireNextImageKHR).
        static constexpr VkPipelineStageFlags2 kAcquireWaitStages =
                VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        // Present-wait semaphores are PER SWAPCHAIN IMAGE (indexed by the
        // acquired image index), not per frame-in-flight: a binary semaphore
        // handed to vkQueuePresentKHR stays "in use" until the presentation
        // engine releases that image (signalled by a later acquire of the
        // SAME index), so a frame-indexed semaphore can be re-signalled
        // while the swapchain still holds it (VUID-vkQueueSubmit2-semaphore-
        // 03868). Sized to the swapchain image count; recreated with the
        // swapchain (createRenderFinishedSemaphores).
        std::vector<VkSemaphore> renderFinished;
        std::array<VkFence,         kFramesInFlight> inFlight{};

        // Suppressed-present swapchain state (ctx->presentSuppressed(), i.e. a
        // headless canvas). Nothing is ever presented, so nothing is ever
        // released back to the presentation engine either: each frame-in-flight
        // slot acquires ONE image the first time it runs and keeps it for the
        // swapchain's lifetime. UINT32_MAX = this slot has not acquired yet.
        // Reset by recreateSwapchainAndDescriptors, which is the only thing that
        // invalidates the images.
        std::array<uint32_t, kFramesInFlight> pinnedSwapImage_{};
        // Whether imageAvailable[slot] still carries an unconsumed signal from
        // the acquire. True for exactly the one frame that follows an acquire;
        // false afterwards, so the submit does not wait on an already-consumed
        // binary semaphore (which would deadlock).
        std::array<bool, kFramesInFlight> acquireSemPending_{};

        uint32_t currentFrame = 0;
        bool needsResize = false;

        // ── Frame-serial deferred-deletion (retire) queue ────────────────────
        // Replaces staleness-triggered vkDeviceWaitIdle stalls (runtime texture
        // / material / particle / sprite swaps, AS rebuilds). See
        // VulkanRetireQueue.hpp for the full fence-invariant derivation.
        //
        // frameSerial_ = serial of the frame CURRENTLY being recorded. Advanced
        // once per SUBMITTED frame in endFrame() (in lockstep with currentFrame,
        // so a failed acquire that skips endFrame never desyncs serial↔slot).
        // Any retire()/retireAS() during a frame stamps frameSerial_; the drain
        // at frame start reclaims resources stamped kFramesInFlight+ frames ago.
        uint64_t frameSerial_ = 0;
        vulkan::RetireQueue retireQueue_;

        // Records pruneFreedAssets took out of blasCache, each with the serial
        // of the frame that retired it, destroyed by the retire queue's rule.
        std::vector<std::pair<uint64_t, std::unique_ptr<BlasRecord>>> retiredRecords_;

        // ImGui (or any post-render overlay) hook. When set, the swapchain
        // image is transitioned GENERAL → COLOR_ATTACHMENT_OPTIMAL after the
        // deferred-shaded frame, a dynamic render pass is opened with
        // LOAD_OP_LOAD, the callback draws into it, then we transition to
        // PRESENT_SRC.
        std::function<void(void*)> overlayCallback;

        // Per-frame lifecycle state, following the multi-pass / HUD pattern:
        // the first render() of a user animate iteration runs
        // beginFrame() (acquire + per-frame UBO uploads + cmd buffer record),
        // subsequent render() calls in the same iteration append to that
        // same cmd buffer, and the Canvas frame-end callback fires endFrame()
        // (ImGui overlay, swapchain → PRESENT_SRC, submit, vkQueuePresentKHR).
        //
        // Idle              : no frame in flight; next render() runs beginFrame.
        // RecordingPostShade: deferred shade pass recorded; further
        //                     render(ortho) calls append HUD overlay draws
        //                     to the same cb.
        // RecordingOrthoOnly: render(ortho) called with no prior deferred
        //                     shade — frame was opened with an empty result
        //                     (cleared swapchain). Rare; mostly defensive.
        enum class FrameState { Idle, RecordingPostShade, RecordingOrthoOnly };
        FrameState frameState_ = FrameState::Idle;

        // A swapchain readback (readRGBPixels) finished this canvas iteration's
        // frame early, to read the pixels render() had only recorded. A pass
        // that can only EXTEND an open frame (the HUD's ortho render, a
        // split-screen pane) has nothing to draw onto after that, and from
        // Idle it would open a frame of its own. render() drops it while this
        // is set. Set only inside the canvas animate body, cleared by the
        // frame-end callback: a caller driving render() directly has no
        // iteration to speak of, and its next render() is a new frame anyway.
        bool frameEndedByReadback_ = false;

        // setOrthographicSceneRendering: an ortho camera names a 3D VIEW, not a
        // 2D overlay, so a standalone render() with one takes the deferred path.
        // Off by default — every existing 2D/HUD user of this backend renders
        // through the ortho-only overlay path and must keep doing so.
        bool orthoSceneRendering_ = false;

        // Internal ortho camera used for the screen-space sprite overlay
        // auto-call from beginDeferredFrame. Lazy-created on first use, then
        // its bounds + projection are rebuilt each frame to track the
        // current swapchain extent. Lives here (vs. stack-allocated each
        // frame) so its matrixWorldInverse / projectionMatrix don't get
        // rebuilt from scratch when the size hasn't changed.
        std::shared_ptr<OrthographicCamera> screenSpaceCam_;
        uint32_t   frameImageIndex_ = 0;

        // Scene-only swapchain capture (for event-camera and similar
        // sensors that need a pre-overlay readback). When enabled, the
        // renderer copies the post-TAA swapchain image into this host-
        // visible buffer BEFORE the sprite + ImGui overlays composite,
        // so sensor pipelines don't see their own visualisation.
        bool             sceneCaptureEnabled_ = false;
        vulkan::Buffer   sceneCaptureBuf_{};
        uint32_t         sceneCaptureBufW_ = 0;
        uint32_t         sceneCaptureBufH_ = 0;

        // ── Frame zero-copy interop (VulkanRenderer::enableFrameInterop) ──
        // One exported TRANSFER_DST buffer per armed (view, channel), filled by
        // a vkCmdCopyImageToBuffer recorded at the frame's record tail. Kept
        // here rather than on ViewContext because the invalidation rule is
        // "kill the whole view's set at once" and the record walk wants the
        // armed views, not all of them — a scene with no interop iterates an
        // empty vector.
        struct FrameInteropChannel {
            VulkanRenderer::FrameChannel channel{};
            vulkan::ExternalBuffer       buf{};
            uint32_t width = 0, height = 0, bpp = 0;
            bool     bgra = false;
        };
        struct FrameInteropView {
            uint32_t viewHandle = 0;// 0 = primary, else a secondary's handle
            std::vector<FrameInteropChannel> channels;
        };
        std::vector<FrameInteropView> frameInterops_;

        struct FrameInteropCopy {
            VkImage image;
            VkImageAspectFlags aspect;
            VkImageLayout restLayout;
            VkBuffer dst;
            uint32_t width, height;
        };
        std::vector<FrameInteropCopy> frameInteropCopies_;// this frame's, built by addFrameInteropPass

        // Deferred-apply queue for setters that issue vkDeviceWaitIdle +
        // realloc descriptor/image resources. These collide with an open
        // cmd buffer (frameState_ != Idle), so when called mid-frame we
        // stash the request and apply it at the next beginDeferredFrame —
        // after the prior frame's fence wait, before any record. Setters
        // called when Idle apply immediately as before.
        bool pendingRenderScaleRealloc_ = false;
        bool pendingAccumulationReset_  = false;
        int  pendingProbeCascades_      = -1;// setProbeCascades asked mid-frame; -1 = nothing pending
    };

}// namespace threepp::vulkan::impl
