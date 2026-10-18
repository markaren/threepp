// SkyBake — threepp::Sky on the deferred renderer.
//
// A Sky is a Mesh: a box carrying a ShaderMaterial that shades the Preetham
// daylight model (three.js's Sky). The GL renderer draws that dome. The
// deferred renderer has no ShaderMaterial path and keeps ONE environment image
// for the sky it shows and the light it lights with, so the scene gather drops
// the box (VulkanCoreScene: as a mesh it sealed the scene in a dark room —
// every pixel the box's inside, every sun and GI ray stopped by it) and this
// pass bakes the model into that image instead: sky_preetham.comp writes mip 0,
// prefilter_env.comp mips it, exactly as an HDRI is mipped.
//
// The image is persistent and rewritten IN PLACE inside the frame's command
// buffer whenever the Sky's uniforms change — a sun that moves every frame is
// the ordinary case (examples/lights/directional.cpp) — with none of what a
// new scene.environment costs: no device drain, no temporal-history wipe, no
// descriptor rewrite. The Impl's envImage holds a non-owning copy of the image
// while the sky is active (Impl::dropEnvImage knows the difference).
//
// The one-sun policy (EnvSunPolicy) holds for the sky as for an HDRI: with
// extraction wanted the mips are prefiltered from the DISC-FREE sky (a disc
// of 1e4 x the sky cannot be Monte Carlo prefiltered) and its energy is the
// EnvSunExtract PreethamSky's preethamSun reports — a scene with its own
// DirectionalLight keeps that; one without gets the Sky's sun injected.

#ifndef THREEPP_VULKAN_SKY_BAKE_HPP
#define THREEPP_VULKAN_SKY_BAKE_HPP

#include "threepp/renderers/common/PreethamSky.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class EnvPrefilter;
    class VulkanContext;

    class SkyBake {

    public:
        // 512 x 256 texels: the model is smooth, and its disc is one texel
        // (0.7 deg) — the background and true mirrors show a bright dot where
        // the GL dome draws a crisp disc.
        static constexpr uint32_t kWidth = 512;
        static constexpr uint32_t kHeight = 256;
        // Prefilter samples per texel. A smooth, disc-free source integrates
        // cleanly with far fewer than an HDRI's 512, and this runs every frame
        // the sun moves.
        static constexpr uint32_t kPrefilterSamples = 128;

        SkyBake(VulkanContext& ctx, EnvPrefilter& prefilter);
        ~SkyBake();
        SkyBake(const SkyBake&) = delete;
        SkyBake& operator=(const SkyBake&) = delete;

        // The env image — every mip in SHADER_READ_ONLY_OPTIMAL once record()
        // has run — allocated on the first call, with no GPU work. Owned here
        // for the renderer's life; callers copy the handles, never retire them.
        const Image2D& image();

        // Record the bake into `cb`: mip 0 and the prefiltered mips, in place,
        // every mip left in SHADER_READ_ONLY_OPTIMAL for the frame's readers.
        // sunFreeMips: the mips integrate the sky WITHOUT its disc, so mip 0 is
        // written twice — the disc-free source first, the sky with its disc
        // last.
        void record(VkCommandBuffer cb, const PreethamSkyParams& p, bool sunFreeMips);

    private:
        VulkanContext& ctx_;
        EnvPrefilter& prefilter_;
        VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline pipeline_ = VK_NULL_HANDLE;
        VkDescriptorPool descPool_ = VK_NULL_HANDLE;
        Image2D image_{};
        std::vector<VkImageView> mipViews_;      // one single-level storage view per mip
        VkDescriptorSet bakeSet_ = VK_NULL_HANDLE;// mip 0 as the bake's storage image
        std::vector<VkDescriptorSet> mipSets_;   // the prefilter's sets, mipSets_[mip - 1]
        bool firstUse_ = true;                   // the image is still UNDEFINED

        void createPipeline();
        void createImage();
        void dispatchBake(VkCommandBuffer cb, const PreethamSkyParams& p, bool withDisc) const;
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_SKY_BAKE_HPP
