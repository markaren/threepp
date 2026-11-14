// SensorState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the lidar scanner and the event camera (its detector, luma buffer and shade pipeline), and the per-
// scene opt-in for sensor-only surfaces.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct SensorState {
        // Impl saw this one through VulkanRenderer's scope.
        using EventCameraSource = VulkanRenderer::EventCameraSource;

        // Path-traced LIDAR scanner — see vulkan/LidarScanner.{hpp,cpp}.
        // Owns its own RT pipeline + SBT + descriptor set; reuses the
        // main TLAS + geomDescs + matDescs via per-scan binding updates.
        // Synchronous scan API exposed publicly via
        // VulkanRenderer::scanLidar. Constructed lazily on first use so
        // the SPV + RT pipeline cost is paid only when a user wants it.
        std::unique_ptr<vulkan::LidarScanner> lidar_;

        // GPU event-camera detector. Lazy-constructed on first
        // setEventCameraEnabled call. Owns the persistent log-history
        // image, the accumulator image, and the per-frame readback ring.
        std::unique_ptr<vulkan::EventCameraDetector> eventCam_;
        bool eventCamEnabled_ = false;
        vulkan::EventCameraDetector::Params eventCamParams_{};
        // What event_shade feeds the detector: the G-buffer Lambert proxy or
        // the presented frame. See VulkanRenderer::EventCameraSource.
        EventCameraSource eventCamSource_ = EventCameraSource::Shaded;

        // Events-only render mode. When on, recordCommandBuffer skips
        // every pass downstream of the raster G-buffer prepass: deferred
        // shade, denoise, TAA, hybrid overlay, upscale. The swapchain is
        // cleared to black so the sprite overlay (event accumulator) has a
        // clean canvas; event_shade + event_detect run in the outer
        // beginFrameAndRecord flow exactly as in the normal render path.
        // Designed for high-rate event-camera sampling
        // (~500 Hz target) where the renderer's visual output isn't
        // displayed at all — only the event accumulator readout matters.
        // Requires hybrid OR TAA to be on so the gbuf prepass actually
        // runs (event_shade reads its normal + ids attachments).
        bool eventsOnlyMode_ = false;

        // Event-camera shade pipeline. Runs immediately after the raster
        // G-buffer pass (when event camera is on) and writes a clean
        // deterministic luma buffer that the detector consumes instead of
        // the stochastically-shaded, noisier swapchain copy. Allocated
        // lazily alongside the detector; descriptor bindings refresh
        // per-frame to track the current frame's gbuf views + material/light
        // buffers.
        vulkan::Buffer        eventLumaBuf_{};
        VkDescriptorSetLayout eventShadeDsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      eventShadePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            eventShadePipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      eventShadeDescPool_       = VK_NULL_HANDLE;
        // One descriptor set per frame-in-flight. The set is rewritten every
        // frame (gbuf views + material/light buffers are per-frame), so a
        // single shared set would be updated while the other in-flight frame
        // still has it bound in a pending command buffer —
        // VUID-vkUpdateDescriptorSets-None-03047. The GPU reading the racing
        // update (wrong-frame gbuf) was the event-camera "binary flicker".
        std::array<VkDescriptorSet, kFramesInFlight> eventShadeDescSets_{};
        uint32_t              eventLumaW_ = 0;
        uint32_t              eventLumaH_ = 0;

        // User-requested sensor resolution. 0 means "track swapchain";
        // any non-zero pair pins the detector + luma buffer at that res
        // and the event_shade pass nearest-samples the (typically larger)
        // gbuf into it. Real DVS sensors live in the 128² – 640×480 range,
        // far below the swapchain — running the detector at sensor-native
        // res cuts the per-frame work by the squared scale factor.
        uint32_t eventCamUserW_ = 0;
        uint32_t eventCamUserH_ = 0;

        // Sensor-only surfaces: the per-scene opt-in for meshes on
        // VulkanRenderer::kSensorOnlyLayer. OFF means they are hit by nothing —
        // no raster in any view, TLAS instance mask 0 — so a scene that never
        // opts in senses and renders exactly as it did before the feature
        // existed. Toggling it clears sceneBuilt_: the masks live in the TLAS.
        // ON is necessary and not sufficient on the raster side: a secondary
        // view also has to ask (ViewContext::sensorSurfaces).
        bool sensorOnlySurfaces_ = false;

        // Per-slot staging for the outstanding dispatches, sized by
        // scanLidarBegin and consumed by scanLidarCollect.
        std::array<std::vector<vulkan_lidar::LidarResult>,
                   vulkan::LidarScanner::kScanSlots> lidarRaw_{};
        // Whether the dispatch in each slot was a paired clean/degraded trace,
        // i.e. whether its raw rows are two legs rather than one. Recorded at
        // dispatch because collect() cannot tell from the row count alone.
        std::array<bool, vulkan::LidarScanner::kScanSlots> lidarPaired_{};
        // Entry-index -> stable id, snapshotted per slot AT DISPATCH. A scan
        // fired with scanBegin() may be collected several frames later, by
        // which time an edit can have renumbered the entry list; translating
        // against the live table would then relabel returns that were traced
        // against the older TLAS. The snapshot is a few KB at most.
        std::array<std::vector<uint32_t>, vulkan::LidarScanner::kScanSlots> lidarStableIds_{};
    };

}// namespace threepp::vulkan::impl
