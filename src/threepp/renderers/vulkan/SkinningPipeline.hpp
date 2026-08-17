// SkinningPipeline — GPU skinning compute pipeline. One descriptor set per
// SkinnedMesh, allocated from a dedicated pool (capacity = 256 simultaneous
// skinned meshes). The shader is a straight LBS — see skinning.comp.
//
// Per-mesh state (BLAS, bone-matrix buffer, descriptor set itself) lives in
// the renderer's SkinnedMeshState. This class just owns the shared pipeline
// + descriptor pool + layout and exposes the dispatch primitive.
//
// Extracted from VulkanRenderer.cpp during the file split; mirrors the
// TaaResolve / EnvPrefilter pattern.

#ifndef THREEPP_VULKAN_SKINNING_PIPELINE_HPP
#define THREEPP_VULKAN_SKINNING_PIPELINE_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class SkinningPipeline {

    public:
        // Max simultaneous skinned meshes. Each set holds 7 storage buffers.
        // Bump if scenes ever exceed; allocateMeshDescriptorSet throws a
        // clearer error with the live count when the pool is exhausted.
        static constexpr uint32_t kMaxSkinnedMeshes = 256;

        // Bone-matrix ring depth, and the reason the pool is kBoneSlots times
        // bigger than the mesh count.
        //
        // refreshSkinnedBlas memcpys the frame's bone matrices into a
        // host-visible buffer from ensureSceneBuilt, which runs BEFORE
        // renderFrame waits on inFlight[currentFrame] — so up to
        // kFramesInFlight earlier frames may still have a skinning dispatch in
        // flight reading it. One buffer means the CPU overwrites the pose a
        // queued dispatch has not consumed yet: that frame skins with the NEXT
        // frame's pose, and the picture shows one pose twice and then jumps.
        // Judder at a rock-steady 60 FPS, and Vulkan-only — GL's driver renames
        // its uniform storage under the same usage.
        //
        // kFramesInFlight + 1, not kFramesInFlight: the host write happens
        // before this frame's fence wait, so the slot belonging to frame
        // N - kFramesInFlight can still be in flight. Exactly the ring
        // TetSkinningPipeline::kPosSlots is, for exactly the same reason.
        static constexpr uint32_t kBoneSlots = 3;

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
        // Live count for diagnostics (#allocated − #freed).
        [[nodiscard]] uint32_t liveSetCount() const { return liveSetCount_; }

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
        uint32_t              liveSetCount_   = 0;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_SKINNING_PIPELINE_HPP
