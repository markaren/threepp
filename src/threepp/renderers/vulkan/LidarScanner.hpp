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
// Dispatch is split into submit() + collect() so the caller never blocks on
// the GPU: submit() records and submits, collect() picks the results up on a
// LATER call (in practice the next scan, a frame later, by which point the
// fence has long since signalled). The old single synchronous scan() paid a
// vkQueueSubmit + vkWaitForFences round trip inline — and, worse, the caller
// had to drain the whole device first to keep the TLAS stable, which
// serialised CPU and GPU and cost ~19 ms on every scan frame of a 60 Hz app.
//
// The caller is responsible for one ordering rule: a submitted scan reads the
// renderer's TLAS and descriptor buffers, so it must complete before anything
// rebuilds them. waitPending() is the hook for that — call it once per frame
// before touching acceleration structures; in steady state the fence is
// already signalled and it costs nothing.

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

        // Record + submit an RT dispatch of `numBeams` invocations. Returns as
        // soon as the work is queued; nothing is read back here. Returns false
        // (and submits nothing) when the scene is not yet built
        // (tlas == VK_NULL_HANDLE or buffers null/empty), so the first frame
        // after construction is safe to call.
        //
        // The result slot count is remembered for the matching collect().
        bool submit(VkQueue queue,
                    VkAccelerationStructureKHR tlas,
                    VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                    VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                    VkBuffer fogUbo, VkDeviceSize fogUboSize,
                    const vulkan_lidar::LidarPushConstants& pc,
                    const vulkan_lidar::LidarBeam* beams, uint32_t numBeams);

        // Copy the results of the outstanding submit() into `outResults`,
        // waiting on its fence first (already signalled in steady state).
        // Returns the number of result SLOTS written, or 0 when nothing was
        // pending. Slot layout: outResults[beamIdx * maxReturns + returnSlot];
        // unused slots carry miss sentinels (hitInstanceId = -1) so the host
        // iterates a fixed-stride array and filters.
        //
        // The caller owns the storage and must size it for at least
        // pendingSlots() LidarResult entries.
        uint32_t collect(vulkan_lidar::LidarResult* outResults);

        // Result slots the outstanding submit will produce (0 when idle) —
        // lets the caller size its buffer before calling collect().
        [[nodiscard]] uint32_t pendingSlots() const { return pending_ ? pendingSlots_ : 0u; }
        [[nodiscard]] uint32_t pendingBeams() const { return pending_ ? pendingBeams_ : 0u; }
        [[nodiscard]] bool pending() const { return pending_; }

        // Block until an outstanding submit has completed, WITHOUT reading it
        // back (it stays pending for the next collect()). The renderer calls
        // this before it rebuilds the TLAS / descriptor buffers the in-flight
        // scan is reading. Free in steady state — the fence signalled a frame
        // ago.
        void waitPending();

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
        uint32_t capacityBeams_   = 0;// beam-buffer rows
        uint32_t capacityResults_ = 0;// result-buffer rows (= beams × maxReturns)

        // Outstanding submit() state, consumed by collect()/waitPending().
        bool     pending_        = false;
        bool     pendingWaited_  = false;// fence already observed signalled
        uint32_t pendingSlots_   = 0;
        uint32_t pendingBeams_   = 0;

        void createDescriptorLayout();
        void createPipeline();
        void createSbt();
        void createCommandObjects();

        // Round (numBeams, slotsPerBeam) up to powers of two and reallocate
        // buffers if larger than current capacity. `slotsPerBeam` is the
        // total result slots per beam = samplesPerBeam · maxReturns.
        // Updates the descriptor set bindings for beamBuf_ + resultBuf_
        // since the VkBuffer handles change.
        void ensureCapacity(uint32_t numBeams, uint32_t slotsPerBeam);

        // Update the shared bindings (TLAS, geomDescs, matDescs, fogUbo)
        // before dispatch. The beam/result bindings are updated only when
        // ensureCapacity recreates the buffers.
        void updateSceneBindings(VkAccelerationStructureKHR tlas,
                                 VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                                 VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                                 VkBuffer fogUbo, VkDeviceSize fogUboSize);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_LIDAR_SCANNER_HPP
