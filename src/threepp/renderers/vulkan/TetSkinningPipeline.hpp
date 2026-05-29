// TetSkinningPipeline — GPU tetrahedral-skinning compute pipeline for PhysX soft
// bodies. One descriptor set per tet-skinned mesh, allocated from a dedicated pool.
// The shader (tet_skinning.comp) barycentric-blends the collision-tet positions into
// the BLAS vertex/normal buffers, then the BLAS refit reads them — the same pattern
// as SkinningPipeline (bone LBS) and the ocean displacement compute.
//
// Per-mesh state (BLAS, the tet-binding buffers, the per-frame tet-position buffer,
// the descriptor set itself) lives in the renderer's TetMeshState. This class owns
// the shared pipeline + descriptor pool + layout and exposes the dispatch primitive.

#ifndef THREEPP_VULKAN_TET_SKINNING_PIPELINE_HPP
#define THREEPP_VULKAN_TET_SKINNING_PIPELINE_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class TetSkinningPipeline {

    public:
        // Max simultaneous tet-skinned meshes. Each set holds kBindingsPerSet
        // storage buffers; bump if scenes exceed it (allocateMeshDescriptorSet
        // throws a clearer error with the live count when the pool is exhausted).
        static constexpr uint32_t kMaxTetMeshes  = 256;
        // Storage buffers per set — must match tet_skinning.comp's binding count.
        static constexpr uint32_t kBindingsPerSet = 9;

        explicit TetSkinningPipeline(VulkanContext& ctx);
        ~TetSkinningPipeline();
        TetSkinningPipeline(const TetSkinningPipeline&) = delete;
        TetSkinningPipeline& operator=(const TetSkinningPipeline&) = delete;

        [[nodiscard]] VkDescriptorSetLayout layout() const { return dsLayout_; }

        VkDescriptorSet allocateMeshDescriptorSet();
        void            freeMeshDescriptorSet(VkDescriptorSet ds);
        [[nodiscard]] uint32_t liveSetCount() const { return liveSetCount_; }

        void bindPipeline(VkCommandBuffer cb);
        void recordDispatch(VkCommandBuffer cb,
                            VkDescriptorSet ds,
                            uint32_t        vertexCount);

    private:
        VulkanContext&        ctx_;
        VkDescriptorSetLayout dsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_       = VK_NULL_HANDLE;
        uint32_t              liveSetCount_   = 0;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_TET_SKINNING_PIPELINE_HPP
