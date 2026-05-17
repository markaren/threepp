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

        // Must match foam_world.comp's `Pc` struct (80 bytes total):
        // 1 × VkDeviceAddress (8) + 18 × u32/float (72) — final `_pad` slot
        // brings the C++ struct to a multiple of 8 to match scalar-layout
        // sizing on the shader side.
        struct PushConstants {
            VkDeviceAddress disturbAddr;  // 0 = no disturbance buffer
            uint32_t        foamRes;
            float           foamTileSize;
            float           tileSize0;
            float           tileSize1;
            float           tileSize2;
            float           waveScale;
            float           choppiness;
            uint32_t        cascadeMask;
            float           hullCenterX;
            float           hullCenterZ;
            float           hullHalfLength;
            float           hullHalfBeam;
            float           hullSinYaw;
            float           hullCosYaw;
            float           forwardSpeed;
            uint32_t        disturbCount;
            float           decay;
            uint32_t        _pad;
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
