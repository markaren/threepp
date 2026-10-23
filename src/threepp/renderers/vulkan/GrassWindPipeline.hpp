// GrassWindPipeline — compute pipeline that applies wind sway to a merged
// grass-field mesh, writing displaced positions directly into the GrassMesh's
// BLAS vertex buffer. All I/O is passed by buffer-reference device address in
// the push constants, so (unlike WaterDisplacePipeline) there are NO descriptor
// sets and no per-mesh descriptor allocation.
//
// One pipeline shared across all GrassMesh instances; per-mesh state (BLAS,
// rest-position + height-fraction buffers) lives in the renderer's
// GrassMeshState. Mirrors the WaterDisplacePipeline pattern, FFT-free.

#ifndef THREEPP_VULKAN_GRASS_WIND_PIPELINE_HPP
#define THREEPP_VULKAN_GRASS_WIND_PIPELINE_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class GrassWindPipeline {

    public:
        // Must match grass_wind.comp's `Pc` block (scalar layout): three
        // 8-byte device addresses (24) + uint + five floats (24) = 48, the
        // downwash's address (8) and the field's world matrix rows (48):
        // 104 bytes, inside the 128 every implementation guarantees.
        struct PushConstants {
            VkDeviceAddress posOut;   // displaced xyz out (= BLAS vertex buffer)
            VkDeviceAddress restIn;   // immutable rest xyz
            VkDeviceAddress hfracIn;  // per-vertex height fraction
            uint32_t        vertexCount;
            float           time;
            float           windStrength;
            float           windDirX;
            float           windDirZ;
            float           bladeHeight;  // field-local m (the downwash's lean)
            VkDeviceAddress downwashAddr; // DownwashGpu of this frame, 0 = none
            float           worldRows[12];// the field's world matrix, rows 0..2
        };

        explicit GrassWindPipeline(VulkanContext& ctx);
        ~GrassWindPipeline();
        GrassWindPipeline(const GrassWindPipeline&) = delete;
        GrassWindPipeline& operator=(const GrassWindPipeline&) = delete;

        // Bind + push + dispatch over pc.vertexCount in 64-thread groups.
        void recordDispatch(VkCommandBuffer cb, const PushConstants& pc);

    private:
        VulkanContext&   ctx_;
        VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline       pipeline_       = VK_NULL_HANDLE;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_GRASS_WIND_PIPELINE_HPP
