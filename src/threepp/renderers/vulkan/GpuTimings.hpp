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
    // frame (overlay on no-overlay frames, DoF when disabled, …).
    enum TimingPass : uint32_t {
        TP_RasterGbuf    = 0,
        TP_OverlayDepth  = 1,
        TP_DeferredShade = 2,// internal name for the deferred-shade compute
                             // dispatch timing bracket; bridges to the public
                             // VulkanRenderer::FrameTimings::pathTraceMs field
                             // (kept for API stability — see GpuTimings.cpp).
        TP_Denoise       = 3,
        TP_TAA           = 4,
        TP_OverlayDraw   = 5,
        TP_GbufResolve   = 6,// MSAA dominant-sample resolve (setGbufferMsaa > 1 only)
        TP_ShadeB        = 7,// MSAA dispatch B: per-sample edge shading (setGbufferMsaa > 1 only)
        TP_Dof           = 8,// thin-lens depth of field (setDepthOfField only)
        TP_Froxel        = 9,// froxel volumetrics: inject + integrate (medium-active frames only)
        TP_SensorImage   = 10,// lens distortion + sensor noise (setLensDistortion / setSensorNoise only)
        TP_COUNT         = 11,
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

        // Silence begin/end for a stretch of recording. There is ONE query
        // pool per frame-in-flight, with one slot pair per pass, so a second
        // view re-running the same passes into the same command buffer would
        // write timestamps that are already written — every one of them a
        // VUID-vkCmdWriteTimestamp2-None-03864, and the resulting numbers a
        // meaningless mix of two views. Secondary views therefore record
        // silently and lastFrameTimings() keeps meaning "the primary".
        void setSuppressed(bool s) { suppressed_ = s; }

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
        bool  suppressed_        = false;

        VulkanRenderer::FrameTimings                   lastTimings_{};
        std::chrono::high_resolution_clock::time_point recordStartTp_{};
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_GPU_TIMINGS_HPP
