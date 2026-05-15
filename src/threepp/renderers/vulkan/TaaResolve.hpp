// TaaResolve — temporal anti-aliasing resolve pass.
//
// Reads the denoise output (`inputView(frame)` — denoise targets this image
// when TAA is active), reprojects last frame's history via the raster
// G-buffer's motion vector, blends with neighborhood-AABB clamp, writes the
// result to the swapchain + a fresh history slot for next frame.
//
// Extracted from VulkanRenderer.cpp during the file split. Owns its
// pipeline, descriptor set layout + pool + sets, sampler, input/history
// image ping-pong. External deps (raster G-buffer views, swapchain views)
// are passed in at descriptor-write time.

#ifndef THREEPP_VULKAN_TAA_RESOLVE_HPP
#define THREEPP_VULKAN_TAA_RESOLVE_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class TaaResolve {

    public:
        // `cmdPool` is used internally for one-shot image layout transitions
        // (UNDEFINED → GENERAL) at image creation time. Pipeline + layout +
        // sampler + descriptor pool + descriptor sets are allocated here;
        // images are deferred to `createImages` so we don't need the surface
        // size yet.
        TaaResolve(VulkanContext& ctx,
                   VkCommandPool cmdPool,
                   uint32_t imageCount,
                   uint32_t framesInFlight);
        ~TaaResolve();
        TaaResolve(const TaaResolve&) = delete;
        TaaResolve& operator=(const TaaResolve&) = delete;

        // Allocate input images at the render extent and history images at
        // the output (swapchain) extent. When the two extents differ the
        // resolve runs as a temporal upsampler; when equal it is a plain
        // 1:1 TAA resolve. Idempotent — frees existing images first. Resets
        // history-valid to false (the freshly-allocated slots are undefined).
        void createImages(uint32_t inWidth, uint32_t inHeight,
                           uint32_t outWidth, uint32_t outHeight);
        void destroyImages();

        // Rewrite all descriptor sets. Caller supplies the external view
        // sources from the raster G-buffer pass + the swapchain. Must be
        // called after createImages + after the raster G-buffer has been
        // allocated (its views must be valid). Both per-frame arrays are
        // indexed by frame-in-flight slot; swapchain views by swapchain
        // image index.
        struct DescriptorWriteInputs {
            VkSampler          gbufSampler         = VK_NULL_HANDLE;
            const VkImageView* gbufMotionPerFrame  = nullptr;// [framesInFlight]
            const VkImageView* gbufIdsPerFrame     = nullptr;// [framesInFlight]
            const VkImageView* swapchainViews      = nullptr;// [imageCount]
        };
        void rewriteDescriptors(const DescriptorWriteInputs& inputs);

        // Per-frame dispatch. Records barrier on input/history images, binds
        // pipeline + descriptor set + push constants, dispatches over the
        // OUTPUT extent in 8×8 groups (each thread reconstructs one full-res
        // pixel; the input may be lower-res). Auto-flips history-valid to
        // true after the first dispatch.
        void recordResolve(VkCommandBuffer cb,
                           uint32_t frame,
                           uint32_t imageIndex,
                           uint32_t inWidth,
                           uint32_t inHeight,
                           uint32_t outWidth,
                           uint32_t outHeight,
                           float blendAlpha);

        // Denoise writes its output here when TAA is active (replaces the
        // direct-to-swapchain write of non-TAA mode).
        [[nodiscard]] VkImageView inputView(uint32_t frame) const {
            return inputImagesPP_[frame].view;
        }
        [[nodiscard]] VkImage inputImage(uint32_t frame) const {
            return inputImagesPP_[frame].image;
        }

        // History images — accessed for inter-frame barriers in the caller's
        // pre-RT block (TAA writes them, denoise / next-frame TAA reads).
        [[nodiscard]] VkImage historyImage(uint32_t slot) const {
            return historyImagesPP_[slot].image;
        }

        // First-frame history is undefined. Caller sets this to false on
        // resetAccumulation; the first recordResolve after that uses
        // alpha=1 so we don't bleed garbage into history.
        void invalidateHistory() { historyValid_ = false; }
        [[nodiscard]] bool historyValid() const { return historyValid_; }

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       imageCount_;
        uint32_t       framesInFlight_;

        // Input image per frame-in-flight (denoise's target) — sized to the
        // path-trace RENDER extent. BGRA8_UNORM to match denoise.comp's
        // rgba8 output and the swapchain channel order.
        std::vector<Image2D> inputImagesPP_;
        // History ping-pong — sized to the OUTPUT (swapchain) extent, so it
        // accumulates the temporal upsampler's reconstructed full-res image.
        // RGBA16F (higher precision than the rgba8 input) so the running
        // mix() doesn't re-quantize to uint8 each frame, which produced
        // visible iso-luminance "lines" on smooth specular surfaces.
        std::array<Image2D, 2> historyImagesPP_{};
        VkSampler sampler_ = VK_NULL_HANDLE;

        VkDescriptorSetLayout dsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_       = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> descSets_;

        bool historyValid_ = false;

        // Internal helpers.
        Image2D createStorageSampledImage(uint32_t w, uint32_t h, VkFormat format,
                                          const char* label);
        void    transitionFreshImage(VkImage img);
        void    createPipeline();
        void    createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_TAA_RESOLVE_HPP
