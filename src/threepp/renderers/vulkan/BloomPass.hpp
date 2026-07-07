// BloomPass — HDR bloom + tone-map/sRGB composite, the tail of the PT post
// stack.
//
// denoise.comp (resolve) writes the linear-HDR scene into sceneHdr (bound to
// the shared RT descriptor set's binding 1). This pass then runs the Jimenez
// 2014 ("Next Generation Post Processing in Call of Duty: Advanced Warfare")
// progressive bloom pyramid:
//   1. downsample chain: sceneHdr → 1/2 → 1/4 → … → ~1/64 with the 13-tap
//      filter (Karis-averaged + soft-knee bright pass + per-tap clamp on the
//      FIRST level only — firefly-safe),
//   2. upsample walk-back: each coarser level is 3×3-tent-filtered and ADDED
//      into the next-finer level, accumulating every level into level 0 —
//      wide, stable, energy-conserving halos whose reach is resolution-
//      independent (a single half-res blur's radius shrinks relative to the
//      image as resolution grows),
//   3. composite: bloom (level 0, intensity normalized by level count) +
//      sceneHdr in linear HDR, tone-map, sRGB-encode → the TAA input image.
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

        // Allocate sceneHdr (render extent) + the bloom pyramid (1/2 … ~1/64,
        // level count clamped to the resolution). Idempotent — frees existing
        // first.
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
        // bloomClamp caps the per-tap HDR input to the bright pass (<= 0 = off)
        // so sub-pixel specular flicker can't pulse the halo radius.
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t width, uint32_t height,
                            uint32_t toneMapping, uint32_t exposureBits,
                            bool bgIsSolidColor, float bloomIntensity,
                            float bloomThreshold, float bloomClamp);

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;
        uint32_t       width_ = 0, height_ = 0;

        // Pyramid: level l is (width >> (l+1)) × (height >> (l+1)) — level 0
        // is half res (what the composite samples), the deepest ~1/64. The
        // upsample accumulates IN PLACE into these images (read+write), so no
        // second chain is needed.
        static constexpr uint32_t kMaxLevels = 6;
        uint32_t levels_ = 0;// actual count for the current extent
        std::vector<Image2D> sceneHdr_;// [framesInFlight] full res rgba16f
        std::vector<Image2D> pyr_;     // [framesInFlight × kMaxLevels], [f*kMaxLevels+l]
        VkSampler sampler_ = VK_NULL_HANDLE;

        // Down + up share a layout (sampler in @0, storage out @1; 28B PC).
        VkDescriptorSetLayout bloomDsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      bloomPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            downPipe_        = VK_NULL_HANDLE;
        VkPipeline            upPipe_          = VK_NULL_HANDLE;
        // Composite layout (sampler @0,1; storage @2,3; 24B PC).
        VkDescriptorSetLayout compDsLayout_    = VK_NULL_HANDLE;
        VkPipelineLayout      compPipeLayout_  = VK_NULL_HANDLE;
        VkPipeline            compPipe_        = VK_NULL_HANDLE;

        VkDescriptorPool descPool_ = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> downSets_;// [f×kMaxLevels]: (sceneHdr|pyr[l-1]) -> pyr[l]
        std::vector<VkDescriptorSet> upSets_;  // [f×kMaxLevels]: pyr[l+1] -> pyr[l] (accumulate)
        std::vector<VkDescriptorSet> compSets_;// sceneHdr+pyr[0]+gbuf -> taaInput

        Image2D createStorageSampledImage(uint32_t w, uint32_t h, const char* label);
        void    transitionFreshImage(VkImage img);
        void    createPipelines();
        void    createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_BLOOM_PASS_HPP
