// VertexSanitizePipeline — the GPU replacement for the CPU finiteness scan that
// guards every BLAS build, for the one path where the CPU array does not exist:
// zero-copy vertex interop (VulkanRenderer::enableVertexInterop).
//
// One descriptor set per interop record, allocated from a dedicated pool, naming
// that record's EXPORTED position buffer. The pass rewrites non-finite vertices
// into zero-area degenerates in place, before the copy into the BLAS's own
// vertex buffer — see shaders/vertex_sanitize.comp for the repair rule and why
// repair rather than rejection.
//
// A plain SSBO binding rather than GrassWindPipeline's push-constant device
// addresses, because the buffer this reads is an exported dedicated allocation:
// createExternalBuffer allocates without VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
// so asking such a buffer for an address is invalid.

#ifndef THREEPP_VULKAN_VERTEX_SANITIZE_PIPELINE_HPP
#define THREEPP_VULKAN_VERTEX_SANITIZE_PIPELINE_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class VertexSanitizePipeline {

    public:
        // Max simultaneous interop records. One set each — unlike the tet path
        // there is no ring, because the exported buffer is a single allocation
        // the producer overwrites in place (§1.1: the ring lives on dynStaging,
        // not on the BLAS buffers).
        static constexpr uint32_t kMaxRecords = 64;

        explicit VertexSanitizePipeline(VulkanContext& ctx);
        ~VertexSanitizePipeline();
        VertexSanitizePipeline(const VertexSanitizePipeline&) = delete;
        VertexSanitizePipeline& operator=(const VertexSanitizePipeline&) = delete;

        // Allocate a set and point binding 0 at `posBuf` (the whole buffer).
        // Returns VK_NULL_HANDLE when the pool is exhausted — the caller then
        // runs interop UNVALIDATED rather than not at all, and says so.
        VkDescriptorSet allocateRecordDescriptorSet(VkBuffer posBuf);
        void            freeRecordDescriptorSet(VkDescriptorSet ds);

        // Bind + push vertexCount + dispatch in 64-thread groups.
        void recordDispatch(VkCommandBuffer cb, VkDescriptorSet ds, uint32_t vertexCount);

    private:
        VulkanContext&        ctx_;
        VkDescriptorSetLayout dsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_       = VK_NULL_HANDLE;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_VERTEX_SANITIZE_PIPELINE_HPP
