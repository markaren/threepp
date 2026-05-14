// Denoiser — spatial path-trace denoiser. Owns two compute pipelines (à-trous
// filter + finalize tone-map / sRGB) plus the swapchain-sized ping-pong
// storage they consume (filtered rgba32f × 2, moments R32F × 2).
//
// Reuses the RT pipeline's set-0 descriptor set layout (rtDsLayout) verbatim
// so the renderer can bind one descriptor set per frame and drive raygen +
// atrous + finalize from it. The host wires bindings 20 (filtered count=2)
// and 33/34 (moments write/read) into that set via the view accessors below;
// accumImage / gbufImage / outImage stay owned by the renderer.
//
// Extracted from VulkanRenderer.cpp during the file split; mirrors the
// TaaResolve / PhotonCaustics pattern.

#ifndef THREEPP_VULKAN_DENOISER_HPP
#define THREEPP_VULKAN_DENOISER_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class Denoiser {

    public:
        // `sharedRtDsLayout` is the RT pipeline's set-0 layout — the denoise
        // pipeline layout reuses it so descriptor sets are interchangeable.
        // The renderer constructs Denoiser after createRtPipeline so the
        // layout is valid here. `cmdPool` is used for one-shot
        // UNDEFINED → GENERAL image transitions at createImages time.
        Denoiser(VulkanContext& ctx,
                 VkDescriptorSetLayout sharedRtDsLayout,
                 VkCommandPool cmdPool);
        ~Denoiser();
        Denoiser(const Denoiser&) = delete;
        Denoiser& operator=(const Denoiser&) = delete;

        // Allocate filtered (rgba32f × 2) and moments (R32F × 2) storage
        // images at the given size. Idempotent — frees existing images
        // first. Caller still clears moments contents via
        // clearGbufImages (variance reads from a stale slot otherwise).
        void createImages(uint32_t width, uint32_t height);
        void destroyImages();

        // Per-frame dispatch. `frameDescriptorSet` is the renderer's RT
        // descriptor set for the current frame-in-flight (bindings 20 /
        // 33 / 34 already wired via the view accessors below). Inserts
        // the RT_SHADER → COMPUTE_SHADER memory barrier internally,
        // then runs the two à-trous passes (stride 1, 2) gated on
        // `denoiseEnabled`, and finally the finalize pass (tone-map +
        // sRGB → outImage) which always runs. `extent` is the dispatch
        // size (swapchain extent).
        void recordDispatch(VkCommandBuffer cb,
                            VkDescriptorSet frameDescriptorSet,
                            VkExtent2D      extent,
                            bool            denoiseEnabled,
                            uint32_t        toneMapping,
                            uint32_t        exposureBits);

        // View / image accessors for descriptor wiring and pre-RT
        // barriers in the renderer. Binding 20 takes both filtered
        // slots (count=2). Binding 33 reads moments[writeSlot], 34
        // reads moments[readSlot] — the renderer continues to compute
        // writeSlot / readSlot itself.
        [[nodiscard]] VkImageView filteredView(uint32_t slot) const {
            return filteredImagesPP_[slot].view;
        }
        [[nodiscard]] VkImageView momentsView(uint32_t slot) const {
            return momentsImagesPP_[slot].view;
        }
        [[nodiscard]] VkImage momentsImage(uint32_t slot) const {
            return momentsImagesPP_[slot].image;
        }

        // Per-pixel primary-surface albedo. Written by closest_hit at the
        // primary hit only (gated on payload.inFlags scatter==0). The atrous
        // shader divides the temporal-accumulated radiance by this before
        // filtering and multiplies back after — surfaces with high-frequency
        // texture detail (paint, weathering) confuse the luminance edge stop
        // when filtering in radiance space; demodulating makes the filter
        // see pure illumination, which is smoother, so the lum-stop can let
        // the filter actually average out fireflies. Zero (sentinel) on
        // non-shaded primaries (sky, metal, glass, emissive) — atrous treats
        // those as "no demod available" and filters in radiance space.
        [[nodiscard]] VkImageView albedoView() const { return albedoImage_.view; }
        [[nodiscard]] VkImage    albedoImage() const { return albedoImage_.image; }

    private:
        VulkanContext&        ctx_;
        VkDescriptorSetLayout sharedDsLayout_   = VK_NULL_HANDLE;
        VkCommandPool         cmdPool_          = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_   = VK_NULL_HANDLE;
        VkPipeline            atrousPipeline_   = VK_NULL_HANDLE;
        VkPipeline            finalizePipeline_ = VK_NULL_HANDLE;

        std::array<Image2D, 2> filteredImagesPP_{};
        std::array<Image2D, 2> momentsImagesPP_{};
        Image2D                albedoImage_{};

        Image2D createStorageImage2D(uint32_t w, uint32_t h,
                                     VkFormat format, const char* label);
        void    transitionFreshImage(VkImage img);
        void    createPipelines();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DENOISER_HPP
