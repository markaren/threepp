// SkinningPipeline — GPU skinning compute pipeline. One descriptor set per
// SkinnedMesh, allocated from a dedicated pool (capacity = 64 simultaneous
// skinned meshes). The shader is a straight LBS — see skinning.comp.
//
// Per-mesh state (BLAS, bone-matrix buffer, descriptor set itself) lives in
// the renderer's SkinnedMeshState. This class just owns the shared pipeline
// + descriptor pool + layout and exposes the dispatch primitive.
//
// Extracted from VulkanRenderer.cpp during the file split; mirrors the
// PhotonCaustics / Denoiser / EnvPrefilter pattern.

#ifndef THREEPP_VULKAN_SKINNING_PIPELINE_HPP
#define THREEPP_VULKAN_SKINNING_PIPELINE_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class SkinningPipeline {

    public:
        // Max simultaneous skinned meshes. Each set holds 7 storage buffers.
        // Resize if you have more skinned characters.
        static constexpr uint32_t kMaxSkinnedMeshes = 64;

        explicit SkinningPipeline(VulkanContext& ctx);
        ~SkinningPipeline();
        SkinningPipeline(const SkinningPipeline&) = delete;
        SkinningPipeline& operator=(const SkinningPipeline&) = delete;

        // The renderer's SkinnedMeshState writes its 7 storage buffer
        // bindings into a set allocated here. The layout is required so the
        // renderer can fill the binding fields correctly.
        [[nodiscard]] VkDescriptorSetLayout layout() const { return dsLayout_; }

        // Allocate / free a per-mesh descriptor set. Pool was created with
        // FREE_DESCRIPTOR_SET_BIT so free is safe across remove/re-add of
        // skinned meshes.
        VkDescriptorSet allocateMeshDescriptorSet();
        void            freeMeshDescriptorSet(VkDescriptorSet ds);

        // Caller binds the pipeline once before iterating pendingSkinnedRebuilds_.
        void bindPipeline(VkCommandBuffer cb);

        // Per-mesh: bind descriptor set + push vertex count + dispatch.
        // Caller must call bindPipeline first.
        void recordDispatch(VkCommandBuffer cb,
                            VkDescriptorSet ds,
                            uint32_t        vertexCount);

    private:
        VulkanContext&        ctx_;
        VkDescriptorSetLayout dsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_       = VK_NULL_HANDLE;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_SKINNING_PIPELINE_HPP
