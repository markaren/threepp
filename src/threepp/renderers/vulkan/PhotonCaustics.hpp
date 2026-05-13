// PhotonCaustics — owns the photon-emit ray-tracing pipeline + the world-grid
// photon storage (count buffer + position/flux/dir data buffer). Emits 512×512
// caustic photons per frame from light sources, splats them into a per-cell
// grid in storage, and closest_hit's 27-cell gather samples them on near-glass
// surfaces to render refractive caustics.
//
// Extracted from VulkanRenderer.cpp during the file split. Encapsulates the
// state that used to live in `Impl` directly + the pipeline / SBT / buffer
// creation logic that was scattered across `createPhotonBuffers`,
// `createPhotonEmitPipeline`, `createPhotonSbt`, and the per-frame dispatch.

#ifndef THREEPP_VULKAN_PHOTON_CAUSTICS_HPP
#define THREEPP_VULKAN_PHOTON_CAUSTICS_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class PhotonCaustics {

    public:
        // Build the pipeline + SBT + buffers. `sharedRtLayout` is the main
        // RT pipeline's VkPipelineLayout — photon emit reuses it so the
        // host can bind the same descriptor set for both passes (matching
        // pipeline layouts is a Vulkan requirement for descriptor reuse).
        PhotonCaustics(VulkanContext& ctx, VkPipelineLayout sharedRtLayout);
        ~PhotonCaustics();
        PhotonCaustics(const PhotonCaustics&) = delete;
        PhotonCaustics& operator=(const PhotonCaustics&) = delete;

        // Storage buffers — the host wires these into the shared RT descriptor
        // set (bindings 15 = counts, 16 = position/flux/dir).
        [[nodiscard]] VkBuffer countBuffer() const { return countBuf_.handle; }
        [[nodiscard]] VkBuffer dataBuffer()  const { return dataBuf_.handle; }

        // "Has the count buffer ever been zero-filled?" Caustic gather in
        // closest_hit reads count first; if uninitialized memory aliases a
        // real glass mesh-ID, gather indexes into garbage and writes huge
        // fireflies. The host calls recordZeroFillCounts() on the first frame
        // a scene with glass appears, even if the glass isn't visible yet,
        // and the emit pass also marks the buffer initialized as a side
        // effect (it zero-fills counts before dispatching anyway). Scene
        // rebuilds invalidate the grid via markUninitialized().
        [[nodiscard]] bool isInitialized() const { return initialized_; }
        void markUninitialized() { initialized_ = false; }

        // First-frame shim: zero-fill the count buffer + insert a
        // TRANSFER → RT_SHADER barrier so caustic gather reads a valid (all
        // zero) grid. Marks initialized_ true on exit. Idempotent at the
        // host — caller gates on isInitialized() to avoid redundant calls.
        void recordZeroFillCounts(VkCommandBuffer cb);

        // Full per-frame emit pass: zero-fill counts → barrier → bind
        // pipeline + descriptor set + push constants → trace 512×512 →
        // post-trace barrier so closest_hit can read the photon buffers.
        // Caller wraps with their own timing markers.
        //
        // The push-constant payload mirrors the main RT pipeline's layout
        // exactly (same 13 uint32 slots) so the same descriptor set + layout
        // can drive both pipelines. Caller packs the values; this class
        // doesn't know what they mean.
        struct EmitPushConstants {
            uint32_t v[13];
        };
        void recordEmitPass(VkCommandBuffer cb,
                            VkDescriptorSet descSet,
                            const EmitPushConstants& push);

    private:
        VulkanContext&   ctx_;
        VkPipelineLayout sharedLayout_ = VK_NULL_HANDLE;

        // Photon-map storage (binding 15 = counts, 16 = pos/flux/dir).
        Buffer countBuf_{};
        Buffer dataBuf_{};

        // Photon-emit ray-tracing pipeline + its SBT. Shares the main RT
        // pipeline's layout so descriptor sets are interchangeable.
        VkPipeline                       emitPipeline_ = VK_NULL_HANDLE;
        Buffer                           sbtBuf_{};
        VkStridedDeviceAddressRegionKHR  rgenRgn_{};
        VkStridedDeviceAddressRegionKHR  missRgn_{};
        VkStridedDeviceAddressRegionKHR  hitRgn_{};
        VkStridedDeviceAddressRegionKHR  callRgn_{};

        bool initialized_ = false;

        // Helpers (one-time work in the constructor).
        void createBuffers();
        void createPipeline();
        void createSbt();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_PHOTON_CAUSTICS_HPP
