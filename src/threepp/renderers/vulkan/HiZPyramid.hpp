// HiZPyramid — reverse-Z "closest depth" mip pyramid over the resolved
// G-buffer depth, rebuilt every frame for the deferred renderer's hybrid
// SSR (deferred_shade.comp binding 55).
//
// R32F, full mip chain at the render extent. Mip 0 copies the depth buffer;
// each further mip is a conservative 2×2 MAX-reduce (reverse-Z: max =
// closest — see hiz_build.comp for why conservatism must lean closer).
// The SSR walk skips whole cells the reflection ray provably stays in
// front of, giving O(log n) screen traversal instead of fixed-step marching.
//
// Follows the SkinningPipeline / ProbeGI house pattern: one class, own
// pipeline + descriptor pool, record(cb, frame). resize() is idempotent and
// recreates image/views/sets only when the extent or depth views change;
// record() lazily transitions the fresh image to GENERAL, builds the chain
// with per-mip barriers, and ends with the write→sampled-read barrier the
// shade dispatch needs — callers add nothing.

#ifndef THREEPP_VULKAN_HIZ_PYRAMID_HPP
#define THREEPP_VULKAN_HIZ_PYRAMID_HPP

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class HiZPyramid {

    public:
        HiZPyramid(VulkanContext& ctx, uint32_t framesInFlight);
        ~HiZPyramid();
        HiZPyramid(const HiZPyramid&) = delete;
        HiZPyramid& operator=(const HiZPyramid&) = delete;

        // (Re)create the pyramid for the render extent + per-frame resolved
        // depth views (sampled in DEPTH_STENCIL_READ_ONLY). Idempotent when
        // nothing changed. The caller must ensure the GPU is idle or the old
        // image unreferenced (call sites piggyback on the gbuffer resize,
        // which already waits).
        void resize(VkExtent2D extent, const VkImageView* depthViews);

        // Build the full chain for this frame's depth. Records its own
        // layout/WAR leading barrier, per-mip reduce barriers and the
        // trailing write→sampled-read barrier.
        void record(VkCommandBuffer cb, uint32_t frame);

        [[nodiscard]] bool        valid()   const { return image_ != VK_NULL_HANDLE; }
        [[nodiscard]] VkImageView view()    const { return fullView_; }// all mips, sampled
        [[nodiscard]] VkSampler   sampler() const { return sampler_; }// NEAREST, unclamped LOD

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        VkImage       image_ = VK_NULL_HANDLE;
        VmaAllocation alloc_ = VK_NULL_HANDLE;
        VkImageView   fullView_ = VK_NULL_HANDLE;
        std::vector<VkImageView> mipViews_;
        VkSampler     sampler_ = VK_NULL_HANDLE;
        VkExtent2D    extent_{0, 0};
        uint32_t      mipCount_ = 0;
        bool          needsLayoutInit_ = true;// fresh image → UNDEFINED

        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [frame * mipCount_ + mip]
        VkImageView cachedDepthViews_[8]{};// change detection for resize()

        void createPipeline();
        void destroyImage();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_HIZ_PYRAMID_HPP
