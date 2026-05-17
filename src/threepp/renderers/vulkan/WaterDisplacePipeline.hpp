// WaterDisplacePipeline — compute pipeline that samples three IFFT cascades
// (height + displacement) and writes per-vertex position / normal / foam
// into the DisplacedMesh's BLAS vertex/normal/foam buffers. One pipeline
// shared across all DisplacedMesh instances; per-mesh descriptor sets bind
// the appropriate cascade images.
//
// Per-mesh state (BLAS, cascade objects, scratch image) lives in the
// renderer's DisplacedMeshState. This class owns the shared pipeline +
// descriptor pool + sampler and exposes the dispatch primitive.
//
// Extracted from VulkanRenderer.cpp during the file split; mirrors the
// SkinningPipeline pattern.

#ifndef THREEPP_VULKAN_WATER_DISPLACE_PIPELINE_HPP
#define THREEPP_VULKAN_WATER_DISPLACE_PIPELINE_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class WaterDisplacePipeline {

    public:
        // Max simultaneous DisplacedMesh instances. Each set holds 6
        // combined-image-samplers (3 cascades × 2 images).
        static constexpr uint32_t kMaxOceans = 16;

        // Must match water_displace.comp's `Pc` struct (128 bytes total):
        // 5 × VkDeviceAddress (40) + 22 × u32/float (88).
        struct PushConstants {
            VkDeviceAddress posOut;
            VkDeviceAddress normOut;
            VkDeviceAddress foamOut;
            VkDeviceAddress disturbAddr;  // 0 = no disturbance buffer
            VkDeviceAddress wakeTrailAddr;// 0 = no historical trail
            uint32_t        vertexCount;
            uint32_t        gridDim;
            float           planeSize;
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
            float           warpCenterX;   // adaptive vertex density: see
            float           warpCenterZ;   // DisplacedMesh::MeshWarp. Shader
            float           warpHalfRange; // gates the whole feature on
            float           warpCoefA;     // warpHalfRange > 0.
            uint32_t        wakeTrailCount;// # valid samples in the trail
        };

        explicit WaterDisplacePipeline(VulkanContext& ctx);
        ~WaterDisplacePipeline();
        WaterDisplacePipeline(const WaterDisplacePipeline&) = delete;
        WaterDisplacePipeline& operator=(const WaterDisplacePipeline&) = delete;

        // Layout exposed so the renderer's DisplacedMeshState can write the
        // 6 cascade image-sampler bindings directly into the set.
        [[nodiscard]] VkDescriptorSetLayout layout() const { return dsLayout_; }
        // Sampler the renderer pairs with each cascade view at descriptor
        // write time.
        [[nodiscard]] VkSampler sampler() const { return sampler_; }

        // Allocate a per-mesh descriptor set. Pool was created without
        // FREE_DESCRIPTOR_SET_BIT; sets are released only when the entire
        // pool is destroyed (matching prior behaviour).
        VkDescriptorSet allocateMeshDescriptorSet();

        // Per-mesh dispatch — binds pipeline + descriptor set + push
        // constants and dispatches over `pc.vertexCount` in 64-thread groups.
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

#endif//THREEPP_VULKAN_WATER_DISPLACE_PIPELINE_HPP
