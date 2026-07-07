// HiZPyramid — reverse-Z depth mip pyramid over a G-buffer depth view.
//
// R32F, full mip chain at the render extent, two reduce flavours selected
// per record() call (see hiz_build.comp for the conservatism rules):
//   MAX (closest)  — the deferred hybrid SSR walk (deferred_shade binding
//                    55) skips whole cells the reflection ray provably
//                    stays in front of: O(log n) screen traversal.
//   MIN (farthest) — the two-phase occlusion culling test ("is everything
//                    in this region provably closer than the box?"); this
//                    instance is built MID-frame from the phase-1 depth.
// One instance per use — the two consumers need different source moments,
// so they never share a build.
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

        // (Re)create the pyramid for the render extent + per-frame depth
        // views (sampled in DEPTH_STENCIL_READ_ONLY). `msDepthViews` backs
        // the sampler2DMS binding — the RAW MS attachment when msSamples > 1
        // (occlusion culling under setGbufferMsaa reduces its samples at
        // mip 0), a 1×1 MS dummy otherwise (bound, never read). Idempotent
        // when nothing changed. The caller must ensure the GPU is idle or
        // the old image unreferenced (call sites piggyback on the gbuffer
        // resize, which already waits).
        void resize(VkExtent2D extent, const VkImageView* depthViews,
                    const VkImageView* msDepthViews, uint32_t msSamples = 1);

        // Build the full chain for this frame's depth. Records its own
        // layout/WAR leading barrier, per-mip reduce barriers and the
        // trailing write→sampled-read barrier. reduceMin selects the
        // flavour: false = closest/max (SSR), true = farthest/min (occlusion).
        void record(VkCommandBuffer cb, uint32_t frame, bool reduceMin = false);

        [[nodiscard]] bool        valid()   const { return image_ != VK_NULL_HANDLE; }
        [[nodiscard]] VkImageView view()    const { return fullView_; }// all mips, sampled
        [[nodiscard]] VkSampler   sampler() const { return sampler_; }// NEAREST, unclamped LOD
        [[nodiscard]] uint32_t    mips()    const { return mipCount_; }

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
        VkImageView cachedDepthViews_[8]{};  // change detection for resize()
        VkImageView cachedMsDepthViews_[8]{};
        uint32_t    msSamples_ = 1;

        void createPipeline();
        void destroyImage();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_HIZ_PYRAMID_HPP
