// FoamWorldPipeline — compute pipeline that builds the world-space foam
// texture. One dispatch per frame per DisplacedMesh; reads the three FFT
// cascades (to evaluate Jacobian foam at every texel) + the previous-frame
// foam (for decay), writes the new foam tile. Replaces the per-vertex
// foam buffer that used to live in water_displace.comp — having foam in
// world coordinates means it stays put as the ocean mesh re-tessellates.
//
// Mirrors WaterDisplacePipeline's shape; one pipeline shared across all
// DisplacedMesh instances, per-mesh descriptor sets bind the appropriate
// cascade images + the foam ping-pong pair.

#ifndef THREEPP_VULKAN_FOAM_WORLD_PIPELINE_HPP
#define THREEPP_VULKAN_FOAM_WORLD_PIPELINE_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class FoamWorldPipeline {

    public:
        // Same cap as WaterDisplacePipeline — each DisplacedMesh owns one
        // descriptor set here.
        static constexpr uint32_t kMaxOceans = 16;
        // a breaking event's life (s), shared with deferred_shade_50_water_glass.glsl's kFoamCarryLife.
        static constexpr double kFoamCarryLife = 0.5;

        // 128-byte host range: the original 112-byte prefix is unchanged,
        // currentTime is at 112, and the optional audit address is at 120.
        // Vessels share WaterDisplacePipeline's OceanHullGpu[hullCount].
        struct PushConstants {
            VkDeviceAddress disturbAddr;   // 0 = no disturbance buffer
            VkDeviceAddress wakeTrailAddr; // every vessel's trail, sliced per hull
            VkDeviceAddress hullAddr;      // OceanHullGpu[hullCount]
            uint32_t        foamRes;
            float           foamTileSize;
            float           tileSize0;
            float           tileSize1;
            float           tileSize2;
            float           waveScale;
            float           choppiness;
            uint32_t        cascadeMask;
            uint32_t        disturbCount;
            float           decay;
            // The extended ocean's fold whitecaps (its main only; upstream's main
            // has no natural deposit): 0 wherever the crest foam or breaking
            // crests own the whitecaps.
            float           natFoamScale;
            uint32_t        hullCount;     // active vessels (0 = no vessel foam)
            float           gateOffX;      // the breaking gate: group velocity x time (world m)
            float           gateOffZ;
            float           gateT;         // < 0 = no gate
            float           fineFold;      // the fine cascade's breaking fold; <= 0 = none
            float           coarseFold;    // the swell cascade's
            float           simDt;         // s since the last dispatch
            float driftX, driftZ, rowSpacing; // advection
            float carryT = 0;              // THREEPP_OCEAN_FOAM_CARRY: 1 + t / kFoamCarryLife; 0 = off
            float currentTime = 0; // current-only material foam origin
            float bandPacked = 0;  // the shade's band packing (1 + Hs + 100 x round(100 x thr)), read under THREEPP_OCEAN_FOAM_CARRY (was auditPad)
            VkDeviceAddress auditAddr = 0; // optional positive-dt diagnostic, never sampled by rendering
        };

        explicit FoamWorldPipeline(VulkanContext& ctx);
        ~FoamWorldPipeline();
        FoamWorldPipeline(const FoamWorldPipeline&) = delete;
        FoamWorldPipeline& operator=(const FoamWorldPipeline&) = delete;

        [[nodiscard]] VkDescriptorSetLayout layout() const { return dsLayout_; }
        [[nodiscard]] VkSampler sampler() const { return sampler_; }

        VkDescriptorSet allocateMeshDescriptorSet();

        // Dispatches `ceil(foamRes/8)²` workgroups of (8,8,1).
        void recordDispatch(VkCommandBuffer cb,
                            VkDescriptorSet ds,
                            const PushConstants& pc);

    private:
        VulkanContext&        ctx_;
        VkDescriptorSetLayout dsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_       = VK_NULL_HANDLE;
        VkSampler             sampler_        = VK_NULL_HANDLE;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_FOAM_WORLD_PIPELINE_HPP
