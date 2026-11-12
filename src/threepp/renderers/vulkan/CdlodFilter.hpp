#pragma once
#include "threepp/renderers/vulkan/water/OceanFFT.hpp"

namespace threepp::vulkan {
// Post-synthesis displacement pyramid. rg = horizontal displacement,
// b = E[|height slope|^2], a = height. No simulation image is modified.
class CdlodFilter {
public:
    CdlodFilter(VulkanContext& ctx, const water::OceanImage& height, const water::OceanImage& displacement);
    ~CdlodFilter();
    void record(VkCommandBuffer cb, float tileSize, float waveScale);
    VkImageView view() const { return view_; }
private:
    VulkanContext& ctx_;
    uint32_t dim_, levels_ = 1;
    VkImage image_{};
    VmaAllocation alloc_{};
    VkImageView view_{}, target_{};
    VkSampler sampler_{};
    VkDescriptorSetLayout dsl_{};
    VkDescriptorPool pool_{};
    VkDescriptorSet ds_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    bool initialized_ = false;
};
}
