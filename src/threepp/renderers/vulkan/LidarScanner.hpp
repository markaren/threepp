// LidarScanner — secondary ray-tracing pipeline that emits one beam per
// invocation against the main renderer's TLAS, evaluates the LIDAR equation
// in a custom closest-hit shader, and writes per-beam (range, intensity,
// normal, instance) results back to the host.
//
// Owns its own pipeline + pipeline layout + descriptor set layout (unlike
// PhotonCaustics, which piggybacks on the main RT layout) — the LIDAR pass
// is decoupled from the path tracer's per-frame state, so it doesn't need
// to thread through the same push constant block. The host wires the
// shared TLAS + geom/mat buffers into LidarScanner's descriptor set just
// before each scan() call.
//
// `scan()` is synchronous: submits the RT dispatch on a private command
// buffer + fence, waits, copies results device→host, returns. Cost is a
// vkQueueSubmit + vkWaitForFences round trip — acceptable for the
// typical 10-30 Hz scan cadence a real LIDAR runs at.

#ifndef THREEPP_VULKAN_LIDAR_SCANNER_HPP
#define THREEPP_VULKAN_LIDAR_SCANNER_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"
#include "threepp/renderers/vulkan/shaders/lidar_shared.h"

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class LidarScanner {

    public:
        explicit LidarScanner(VulkanContext& ctx);
        ~LidarScanner();

        LidarScanner(const LidarScanner&) = delete;
        LidarScanner& operator=(const LidarScanner&) = delete;

        // Synchronous scan. Submits an RT dispatch of `numBeams` invocations,
        // waits for completion, and writes the per-beam results into
        // outResults[0..numBeams-1]. The caller owns the storage; this class
        // does not retain pointers.
        //
        // Bails out gracefully (writes all-miss results) when the scene is
        // not yet built (tlas == VK_NULL_HANDLE or buffers null/empty) so
        // the first frame after construction can call scan() safely.
        void scan(VkQueue queue,
                  VkAccelerationStructureKHR tlas,
                  VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                  VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                  const vulkan_lidar::LidarPushConstants& pc,
                  const vulkan_lidar::LidarBeam* beams, uint32_t numBeams,
                  vulkan_lidar::LidarResult* outResults);

    private:
        VulkanContext& ctx_;

        // Pipeline + descriptor objects.
        VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_  = VK_NULL_HANDLE;
        VkDescriptorSet       descSet_   = VK_NULL_HANDLE;
        VkPipeline            pipeline_  = VK_NULL_HANDLE;

        // Shader binding table + per-stage regions.
        Buffer                          sbtBuf_{};
        VkStridedDeviceAddressRegionKHR rgenRgn_{};
        VkStridedDeviceAddressRegionKHR missRgn_{};
        VkStridedDeviceAddressRegionKHR hitRgn_{};
        VkStridedDeviceAddressRegionKHR callRgn_{};

        // One-shot dispatch infrastructure.
        VkCommandPool   cmdPool_ = VK_NULL_HANDLE;
        VkCommandBuffer cmdBuf_  = VK_NULL_HANDLE;
        VkFence         fence_   = VK_NULL_HANDLE;

        // Per-beam buffers. beamBuf_ is host→device upload (mapped, sequential
        // write); resultBuf_ is device-local SSBO written by the rgen;
        // readbackBuf_ is host-visible, populated via vkCmdCopyBuffer after
        // the trace completes. We don't bother with a separate device-local
        // beam buffer + transfer — the trace is short-lived and the upload
        // pattern is sequential, so a host-visible SSBO performs well.
        Buffer   beamBuf_{};
        Buffer   resultBuf_{};
        Buffer   readbackBuf_{};
        uint32_t capacityBeams_ = 0;

        void createDescriptorLayout();
        void createPipeline();
        void createSbt();
        void createCommandObjects();

        // Round numBeams up to a power of two and reallocate buffers if
        // larger than current capacity. Updates the descriptor set bindings
        // for beamBuf_ + resultBuf_ since the VkBuffer handles change.
        void ensureCapacity(uint32_t numBeams);

        // Update the four shared bindings (TLAS, geomDescs, matDescs)
        // before dispatch. The beam/result bindings are updated only when
        // ensureCapacity recreates the buffers.
        void updateSceneBindings(VkAccelerationStructureKHR tlas,
                                 VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                                 VkBuffer matDescsBuffer, VkDeviceSize matDescsSize);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_LIDAR_SCANNER_HPP
