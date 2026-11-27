// BloomPyramid — the progressive bloom chain shared by BloomPass (over the
// scene's sceneHdr) and BillboardGlowPass (over its billboard target).
//
// bloom_down.comp halves its input level by level (the 13-tap filter, with
// Karis averaging and the soft-knee bright pass on the first level when asked),
// then bloom_up.comp walks back, tent-filtering each coarser level and ADDING
// it into the next finer one, so level 0 ends up holding the whole chain.
// BloomPass.hpp describes the filter.
//
// This class holds what the two passes used to carry as two copies: the two
// pipelines, their (combined sampler @0, storage image @1) set layout and
// 28-byte push block, the linear clamp-to-edge sampler, and one downsample plus
// one upsample descriptor set per level and frame in flight. The owner keeps
// the images, whose usage and transient-pool placement differ between the two,
// and the render-graph declarations that name them.

#ifndef THREEPP_VULKAN_BLOOM_PYRAMID_HPP
#define THREEPP_VULKAN_BLOOM_PYRAMID_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class BloomPyramid {

    public:
        // `label` names the pipelines and the pool in the errors a failed
        // create throws ("bloom", "bbglow").
        BloomPyramid(VulkanContext& ctx, uint32_t framesInFlight, uint32_t maxLevels, const char* label);
        ~BloomPyramid();
        BloomPyramid(const BloomPyramid&) = delete;
        BloomPyramid& operator=(const BloomPyramid&) = delete;

        // The level count for a chain over a w × h input: levels halve from
        // half extent and stop before a level's short side drops under 8 texels
        // (a 3×3 tent on a smaller level is all edge clamp, contributing only
        // blocky smear). At least 1, at most maxLevels.
        [[nodiscard]] static uint32_t levelCount(uint32_t w, uint32_t h, uint32_t maxLevels);
        // Level l of a chain over a w × h input: (w, h) >> (l + 1), at least 1.
        [[nodiscard]] static VkExtent2D levelExtent(uint32_t w, uint32_t h, uint32_t l);

        // The linear clamp-to-edge sampler the chain reads its levels through.
        [[nodiscard]] VkSampler sampler() const { return sampler_; }

        // Point frame `frame`'s sets at `input` and the `levels` consecutive
        // images starting at `level0`: downsample l reads `input` (l = 0) or
        // level l-1 and writes level l; upsample l reads level l+1 and adds into
        // level l. Every image is in GENERAL.
        void writeSets(uint32_t frame, VkImageView input, const Image2D* level0, uint32_t levels);

        struct Chain {
            VkExtent2D input{};     // the region of the input that level 0 reads
            VkExtent2D base{};      // the extent the levels halve from (levelExtent)
            uint32_t   levels = 0;
            float      threshold  = 0.f;// soft-knee cutoff, pushed to every downsample
            float      clampMax   = 0.f;// level 0's per-tap HDR cap; <= 0 = off
            uint32_t   firstLevel = 0;  // 1: level 0 runs Karis + the bright pass
        };
        // Records the downsample chain and the upsample walk-back, with a
        // compute barrier between consecutive dispatches. The barriers against
        // the input's producer and level 0's consumer are the caller's.
        void record(VkCommandBuffer cb, uint32_t frame, const Chain& chain) const;

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;
        uint32_t       maxLevels_;
        std::string    label_;

        VkSampler             sampler_    = VK_NULL_HANDLE;
        VkDescriptorSetLayout setLayout_  = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            downPipe_   = VK_NULL_HANDLE;
        VkPipeline            upPipe_     = VK_NULL_HANDLE;

        // Allocated once for maxLevels whatever the runtime chain depth (the
        // pool exists before the extent is known); only levels-1 up sets are used.
        VkDescriptorPool             descPool_ = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> downSets_;// [frame × maxLevels + level]
        std::vector<VkDescriptorSet> upSets_;  // [frame × maxLevels + level]
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_BLOOM_PYRAMID_HPP
