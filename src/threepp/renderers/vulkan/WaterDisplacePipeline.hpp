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

    // One vessel as water_displace.comp and foam_world.comp read it: the
    // DisplacedMesh::HullExclusion pose, the wake's forward speed and the
    // vessel's slice of the wake-trail buffer. Mirrors `OceanHull` in
    // ocean_cascade.glsl field for field (scalar layout, 48 bytes).
    struct OceanHullGpu {
        float    centerX;
        float    centerZ;
        float    halfLength;
        float    halfBeam;
        float    sinYaw;
        float    cosYaw;
        float    forwardSpeed;// 0 when the wake is disabled
        float    centerY;
        float    pitch;
        float    roll;
        uint32_t trailFirst;  // first WakeSample of this vessel in the trail buffer
        uint32_t trailCount;
    };
    static_assert(sizeof(OceanHullGpu) == 48, "OceanHullGpu must match ocean_cascade.glsl's OceanHull");

    class WaterDisplacePipeline {

    public:
        // Max simultaneous DisplacedMesh instances. Each set holds 6
        // combined-image-samplers (3 cascades × 2 images).
        static constexpr uint32_t kMaxOceans = 16;

        // Must match water_displace.comp's `Pc` struct (96 bytes):
        // 4 × VkDeviceAddress (32) + 16 × u32/float (64). The vessels live in
        // a per-frame buffer (hullAddr → OceanHullGpu[hullCount]); one hull
        // alone used to fill this block to the 128-byte maximum.
        struct PushConstants {
            VkDeviceAddress posOut;
            VkDeviceAddress normOut;
            VkDeviceAddress wakeTrailAddr;// every vessel's trail, sliced per hull
            VkDeviceAddress hullAddr;     // OceanHullGpu[hullCount]
            uint32_t        vertexCount;
            uint32_t        gridDimX;     // vertices along local X / Z — a
            uint32_t        gridDimZ;     // rectangle is first-class now
            float           planeSizeX;
            float           planeSizeZ;
            float           tileSize0;
            float           tileSize1;
            float           tileSize2;
            float           waveScale;
            float           choppiness;
            uint32_t        cascadeMask;
            float           warpCenterX;   // adaptive vertex density: see
            float           warpCenterZ;   // DisplacedMesh::MeshWarp. Shader
            float           warpHalfRange; // gates the whole feature on
            float           warpCoefA;     // warpHalfRange > 0.
            uint32_t        hullCount;     // active vessels (0 = none)
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
