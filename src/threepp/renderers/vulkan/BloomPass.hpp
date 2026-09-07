// BloomPass — the HDR bloom pyramid of the shared post stack.
//
// The shade/resolve stage writes the linear-HDR scene into sceneHdr (bound to
// the deferred shade's descriptor set binding 1). This pass then runs the Jimenez
// 2014 ("Next Generation Post Processing in Call of Duty: Advanced Warfare")
// progressive bloom pyramid:
//   1. downsample chain: sceneHdr → 1/2 → 1/4 → … → ~1/64 with the 13-tap
//      filter (Karis-averaged + soft-knee bright pass + per-tap clamp on the
//      FIRST level only — firefly-safe),
//   2. upsample walk-back: each coarser level is 3×3-tent-filtered and ADDED
//      into the next-finer level, accumulating every level into level 0 —
//      wide, stable, energy-conserving halos whose reach is resolution-
//      independent (a single half-res blur's radius shrinks relative to the
//      image as resolution grows).
//
// The tone-map / sRGB composite that consumes level 0 lives in PostComposite
// (split out so camera/display response isn't a bloom concern). Adding bloom
// before the tone-map curve there is what makes a bright highlight glow far
// more than a mid-tone (the correct, AAA look). With bloomIntensity <= 0 the
// pyramid is skipped entirely.
//
// All images are per-frame-in-flight (no swapchain-image dimension).

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
        // the shade/resolve writes the linear-HDR scene here.
        [[nodiscard]] VkImageView sceneHdrView(uint32_t frame) const {
            return sceneHdr_[frame].view;
        }
        [[nodiscard]] VkImage sceneHdrImage(uint32_t frame) const {
            return sceneHdr_[frame].image;
        }
        // Whole-struct access (image + extent + format) for the determinism
        // audit's host readback (VulkanRenderer::readSceneHdrDebug).
        [[nodiscard]] const Image2D& sceneHdrImage2D(uint32_t frame) const {
            return sceneHdr_[frame];
        }

        // Pyramid level 0 (half res, all levels accumulated) — what the
        // PostComposite samples as the bloom buffer.
        [[nodiscard]] VkImageView bloomView(uint32_t frame) const {
            return pyr_[frame * kMaxLevels].view;
        }
        // Runtime level count for the current extent — PostComposite divides
        // bloomIntensity by it so the summed pyramid lands at the same
        // overall energy a single-level bloom put out for the same slider.
        [[nodiscard]] uint32_t levels() const { return levels_; }

        // Rewrite the per-frame down/up descriptor sets (internal images
        // only). Call after createImages.
        void rewriteDescriptors();

        // Records the bloom pyramid; no-op when bloomIntensity <= 0.
        // width/height = render extent. bloomClamp caps the per-tap HDR input
        // to the bright pass (<= 0 = off) so sub-pixel specular flicker can't
        // pulse the halo radius. Ends with the pyramid writes barriered
        // visible to the next compute consumer.
        void recordPyramid(VkCommandBuffer cb, uint32_t frame,
                           uint32_t width, uint32_t height,
                           float bloomIntensity, float bloomThreshold,
                           float bloomClamp);

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;
        uint32_t       width_ = 0, height_ = 0;

        // Pyramid: level l is (width >> (l+1)) × (height >> (l+1)) — level 0
        // is half res (what the composite samples). The upsample accumulates
        // IN PLACE into these images (read+write), so no second chain is needed.
        //
        // levels_ is derived from the extent (see resize): the chain grows until
        // the smaller dimension drops below 8 px. kMaxLevels only bounds the
        // preallocated array/pool capacity — it must be >= the deepest chain any
        // supported resolution needs, or bloom is silently TRUNCATED there,
        // shrinking the widest tap's screen coverage (a 4K source blooms less
        // broadly than a 1080p one). Levels wanted: 1080p→7, 4K→8, 8K→9. Only
        // `levels_` images are ever created, so a generous cap costs a handful
        // of unused descriptor-set slots, not memory/images.
        static constexpr uint32_t kMaxLevels = 9;
        uint32_t levels_ = 0;// actual count for the current extent
        std::vector<Image2D> sceneHdr_;// [framesInFlight] full res rgba16f
        std::vector<Image2D> pyr_;     // [framesInFlight × kMaxLevels], [f*kMaxLevels+l]
        VkSampler sampler_ = VK_NULL_HANDLE;

        // Down + up share a layout (sampler in @0, storage out @1; 28B PC).
        VkDescriptorSetLayout bloomDsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      bloomPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            downPipe_        = VK_NULL_HANDLE;
        VkPipeline            upPipe_          = VK_NULL_HANDLE;

        VkDescriptorPool descPool_ = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> downSets_;// [f×kMaxLevels]: (sceneHdr|pyr[l-1]) -> pyr[l]
        std::vector<VkDescriptorSet> upSets_;  // [f×kMaxLevels]: pyr[l+1] -> pyr[l] (accumulate)

        Image2D createStorageSampledImage(uint32_t w, uint32_t h, const char* label);
        void    transitionFreshImage(VkImage img);
        void    createPipelines();
        void    createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_BLOOM_PASS_HPP
