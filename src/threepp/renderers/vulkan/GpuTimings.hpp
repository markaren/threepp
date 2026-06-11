// GpuTimings — per-frame GPU timestamp readback + CPU record/frame timing.
//
// Owns one VkQueryPool per frame-in-flight. Each pool holds begin/end pairs
// for every TimingPass. After the per-frame fence signals, readBack() reads
// the previous use of that slot and populates a VulkanRenderer::FrameTimings
// struct for the public getter.
//
// Extracted from VulkanRenderer.cpp during the file split.

#ifndef THREEPP_VULKAN_GPU_TIMINGS_HPP
#define THREEPP_VULKAN_GPU_TIMINGS_HPP

#include "threepp/renderers/VulkanRenderer.hpp"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    // One slot per bracketed GPU pass. timingMask records which passes wrote
    // both endpoints so readBack() can skip pools that weren't touched this
    // frame (photon emit on no-glass frames, overlay on no-overlay frames, …).
    enum TimingPass : uint32_t {
        TP_RasterGbuf   = 0,
        TP_OverlayDepth = 1,
        TP_PhotonEmit   = 2,
        TP_PathTrace    = 3,
        TP_Denoise      = 4,
        TP_TAA          = 5,
        TP_OverlayDraw  = 6,
        TP_COUNT        = 7,
    };
    inline constexpr uint32_t kTimingSlots = TP_COUNT * 2u;

    class GpuTimings {
    public:
        // Probes device timestamp support and creates one VkQueryPool per
        // frame-in-flight. Pools are sized for kTimingSlots = TP_COUNT × 2.
        GpuTimings(VulkanContext& ctx, uint32_t framesInFlight);
        ~GpuTimings();
        GpuTimings(const GpuTimings&)            = delete;
        GpuTimings& operator=(const GpuTimings&) = delete;

        // --- per-frame command-buffer operations ---

        // Call at the start of command recording. Captures the CPU record-start
        // time and GPU-resets this frame's query pool slot.
        void beginFrame(VkCommandBuffer cb, uint32_t frame);

        // GPU timestamp pairs — call begin before a pass, end after it.
        void begin(VkCommandBuffer cb, TimingPass pass, uint32_t frame);
        void end  (VkCommandBuffer cb, TimingPass pass, uint32_t frame);

        // Call at vkEndCommandBuffer to capture cpuRecordMs.
        void finishRecord();

        // --- readback (call after the per-frame fence has signaled) ---

        // Reads GPU timestamps written by the previous use of this frame's
        // pool slot and populates the internal FrameTimings.
        // pendingCpuEnsureMs: the ensureSceneBuilt wall time from the prior
        // render() call (deferred so it appears in the same timing row as the
        // GPU passes for that scene).
        void readBack(uint32_t frame, float pendingCpuEnsureMs);

        // --- out-of-band setters (called outside the cmd-buffer window) ---
        void setCpuFrameMs(float ms) { lastTimings_.cpuFrameMs = ms; }

        // --- accessors ---
        [[nodiscard]] VulkanRenderer::FrameTimings timings() const { return lastTimings_; }
        [[nodiscard]] bool supported() const { return timingsSupported_; }

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        std::vector<VkQueryPool> pools_;
        std::vector<uint32_t>   maskRecorded_;
        float timestampPeriodNs_ = 1.0f;
        bool  timingsSupported_  = false;

        VulkanRenderer::FrameTimings                   lastTimings_{};
        std::chrono::high_resolution_clock::time_point recordStartTp_{};
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_GPU_TIMINGS_HPP
