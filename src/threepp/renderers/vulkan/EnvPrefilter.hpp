// EnvPrefilter — GGX-prefilters an equirect HDR environment map into a
// mip chain (PMREM). Mip 0 is the source mirror; mip k is convolved with
// GGX(α = (k / (N-1))²) by dispatching prefilter_env.comp once per mip.
//
// One pipeline + descriptor pool shared across env uploads; descriptor sets
// are allocated per-mip each upload (pool is reset between uploads).
//
// Extracted from VulkanRenderer.cpp during the file split; mirrors the
// PhotonCaustics / TaaResolve / Denoiser pattern.

#ifndef THREEPP_VULKAN_ENV_PREFILTER_HPP
#define THREEPP_VULKAN_ENV_PREFILTER_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class EnvPrefilter {

    public:
        // Cap mip count regardless of source size. With α = (k/(N-1))² and
        // N = 8, the last mip is α = 1 (fully diffuse hemisphere) and
        // intermediate mips cover the roughness range with even spacing.
        static constexpr uint32_t kMaxEnvMips = 8;

        // `cmdPool` is used for the one-shot upload + dispatch command
        // buffer that buildPmrem submits per env upload.
        EnvPrefilter(VulkanContext& ctx, VkCommandPool cmdPool);
        ~EnvPrefilter();
        EnvPrefilter(const EnvPrefilter&) = delete;
        EnvPrefilter& operator=(const EnvPrefilter&) = delete;

        // Allocate an env Image2D with a full GGX-prefiltered mip chain.
        // Uploads `pixels` (R32G32B32A32_SFLOAT equirect, `byteSize` bytes)
        // to mip 0, then dispatches prefilter_env.comp for mips 1..N-1.
        // The returned image is in SHADER_READ_ONLY_OPTIMAL layout and owns
        // its sampler (LINEAR mipmap, LOD-clamp to mipLevels-1) ready for
        // closest_hit sampling. Caller owns the returned Image2D and is
        // responsible for destroying it via destroyImage2D.
        [[nodiscard]] Image2D buildPmrem(uint32_t w, uint32_t h,
                                         const float* pixels,
                                         VkDeviceSize byteSize);

    private:
        VulkanContext&        ctx_;
        VkCommandPool         cmdPool_         = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsLayout_        = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_  = VK_NULL_HANDLE;
        VkPipeline            pipeline_        = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_        = VK_NULL_HANDLE;
        VkSampler             srcSampler_      = VK_NULL_HANDLE;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_ENV_PREFILTER_HPP
