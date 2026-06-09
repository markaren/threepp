// BloomPass — HDR bloom + tone-map/sRGB composite, the tail of the PT post
// stack.
//
// denoise.comp (resolve) writes the linear-HDR scene into sceneHdr (bound to
// the shared RT descriptor set's binding 1). This pass then:
//   1. down-samples sceneHdr to half res with a Karis-averaged 13-tap filter
//      (firefly-safe),
//   2. blurs it with a separable Gaussian (H then V, two iterations),
//   3. composites bloom + sceneHdr in linear HDR, tone-maps, sRGB-encodes,
//      and writes the LDR result to the TAA input image.
//
// Adding bloom before the tone-map curve is what makes a bright highlight
// glow far more than a mid-tone (the correct, AAA look). With bloomIntensity
// <= 0 the bloom passes are skipped and the composite reproduces the previous
// finalize output exactly.
//
// All images are per-frame-in-flight (no swapchain-image dimension — the
// composite writes the per-frame TAA input, not the swapchain directly).

#ifndef THREEPP_VULKAN_BLOOM_PASS_HPP
#define THREEPP_VULKAN_BLOOM_PASS_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class BloomPass {

    public:
        BloomPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight);
        ~BloomPass();
        BloomPass(const BloomPass&) = delete;
        BloomPass& operator=(const BloomPass&) = delete;

        // Allocate sceneHdr (render extent) + bloom ping-pong (half res).
        // Idempotent — frees existing first.
        void createImages(uint32_t width, uint32_t height);
        void destroyImages();

        // sceneHdr view per frame — bound to the shared RT set's binding 1 so
        // denoise.comp (resolve) writes the linear-HDR scene here.
        [[nodiscard]] VkImageView sceneHdrView(uint32_t frame) const {
            return sceneHdr_[frame].view;
        }
        [[nodiscard]] VkImage sceneHdrImage(uint32_t frame) const {
            return sceneHdr_[frame].image;
        }

        // Rewrite per-frame descriptor sets. External inputs (one view per
        // frame-in-flight): the G-buffer storage view (for the solid-bg sky
        // bypass) and the TAA input view (composite output target).
        struct DescriptorWriteInputs {
            const VkImageView* gbufPerFrame     = nullptr;// [framesInFlight]
            const VkImageView* taaInputPerFrame = nullptr;// [framesInFlight]
        };
        void rewriteDescriptors(const DescriptorWriteInputs& in);

        // Records the bloom chain (skipped when bloomIntensity <= 0) and the
        // composite (always). width/height = path-trace render extent.
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t width, uint32_t height,
                            uint32_t toneMapping, uint32_t exposureBits,
                            bool bgIsSolidColor, float bloomIntensity,
                            float bloomThreshold);

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;
        uint32_t       width_ = 0, height_ = 0, halfW_ = 0, halfH_ = 0;

        std::vector<Image2D> sceneHdr_;// [framesInFlight] full res rgba16f
        std::vector<Image2D> bloomA_;  // [framesInFlight] half res rgba16f
        std::vector<Image2D> bloomB_;  // [framesInFlight] half res rgba16f
        VkSampler sampler_ = VK_NULL_HANDLE;

        // Down + blur share a layout (sampler in @0, storage out @1; 16B PC).
        VkDescriptorSetLayout bloomDsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      bloomPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            downPipe_        = VK_NULL_HANDLE;
        VkPipeline            blurPipe_        = VK_NULL_HANDLE;
        // Composite layout (sampler @0,1; storage @2,3; 24B PC).
        VkDescriptorSetLayout compDsLayout_    = VK_NULL_HANDLE;
        VkPipelineLayout      compPipeLayout_  = VK_NULL_HANDLE;
        VkPipeline            compPipe_        = VK_NULL_HANDLE;

        VkDescriptorPool descPool_ = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> downSets_; // sceneHdr -> bloomA
        std::vector<VkDescriptorSet> blurHSets_;// bloomA   -> bloomB
        std::vector<VkDescriptorSet> blurVSets_;// bloomB   -> bloomA
        std::vector<VkDescriptorSet> compSets_; // sceneHdr+bloomA+gbuf -> taaInput

        Image2D createStorageSampledImage(uint32_t w, uint32_t h, const char* label);
        void    transitionFreshImage(VkImage img);
        void    createPipelines();
        void    createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_BLOOM_PASS_HPP
