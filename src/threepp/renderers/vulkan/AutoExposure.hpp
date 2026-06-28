// AutoExposure — GPU log2-luma histogram + CPU EMA for automatic exposure.
//
// Each frame: recordDispatch() clears the per-frame histogram SSBO with
// vkCmdFillBuffer, then dispatches lum_histogram.comp over sceneHdr (rgba16f).
// tick() runs on the CPU at the START of the FOLLOWING frame (after the
// framesInFlight fence guarantees the prior GPU slot is retired), reads the
// histogram, computes a weighted-percentile luminance and applies an
// asymmetric EMA (fast constriction, slow dilation) to exposure_.

#ifndef THREEPP_VULKAN_AUTO_EXPOSURE_HPP
#define THREEPP_VULKAN_AUTO_EXPOSURE_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class AutoExposure {
    public:
        AutoExposure(VulkanContext& ctx, uint32_t framesInFlight);
        ~AutoExposure();
        AutoExposure(const AutoExposure&) = delete;
        AutoExposure& operator=(const AutoExposure&) = delete;

        // Rewrite per-frame descriptor sets. Call after sceneHdr views change
        // (initial creation and on resize). `sceneHdrViews` must have
        // `framesInFlight` entries.
        void rewriteDescriptors(const VkImageView* sceneHdrViews);

        // Zero this frame's histogram SSBO then dispatch lum_histogram.comp.
        // Callers must ensure sceneHdr writes are visible (compute→compute
        // SHADER_WRITE→SHADER_READ barrier) before calling this.
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t width, uint32_t height);

        // Read the previous frame's histogram (currentFrame - 1, safe via fence
        // lag), compute weighted-percentile EV and advance the EMA.
        // dt = seconds since last frame.
        void tick(uint32_t currentFrame, float dt);

        // Current adapted exposure multiplier (1.0 = no change).
        [[nodiscard]] float exposure() const { return exposure_; }

        // ── Knobs (write at any time; take effect on the next tick()) ────────
        float adaptSpeed  = 2.0f; // EV per second (constriction); dilation is 0.5×
        float minEV       = -3.0f;// exposure floor (EV)
        float maxEV       =  3.0f;// exposure ceiling (EV)
        float lowPercent  = 0.40f;// exclude darkest 40% of pixels from average
        float highPercent = 0.95f;// exclude brightest 5% of pixels from average

    private:
        static constexpr uint32_t kBins   = 128;
        static constexpr float    kEvMin  = -8.0f;
        static constexpr float    kEvMax  =  8.0f;

        VulkanContext& ctx_;
        uint32_t      framesInFlight_;
        float         exposure_  = 1.0f;
        float         currentEV_ = 0.0f;// log2(exposure_)

        struct HistBuf {
            Buffer    buf{};
            uint32_t* ptr = nullptr;// persistent mapped pointer (HOST_COHERENT)
        };
        std::vector<HistBuf> histBufs_;// [framesInFlight]

        VkSampler             sampler_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsLayout_  = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_= VK_NULL_HANDLE;
        VkPipeline            pipe_      = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_  = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> descSets_;// [framesInFlight]

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_AUTO_EXPOSURE_HPP
