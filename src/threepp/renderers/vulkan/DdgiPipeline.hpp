// DdgiPipeline — owns the DDGI probe field: the octahedral irradiance atlas
// (ping-pong), the per-frame ray-radiance scratch buffer, the DDGI uniform,
// and the two pipelines that drive it (a ray-tracing "update" pass that casts
// probe rays + a compute "blend" pass that folds them into the atlas).
//
// Deliberately SELF-CONTAINED: it builds its own descriptor set layouts rather
// than reusing the main RT descriptor set, so standing it up touches none of
// the renderer's shared descriptor infrastructure. The renderer constructs it
// unconditionally but only dispatches the passes when ddgiEnabled_ is set —
// so until that flag flips, this class allocates resources + builds pipelines
// and otherwise does nothing.
//
// Scaffolding stage: the constructor allocates + builds; per-frame dispatch
// (recordUpdate / recordBlend), the scene-AABB grid sizing, the UBO upload,
// and the sampling-side wiring land in subsequent increments.

#ifndef THREEPP_VULKAN_DDGI_PIPELINE_HPP
#define THREEPP_VULKAN_DDGI_PIPELINE_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>

namespace threepp::vulkan {

    class VulkanContext;

    class DdgiPipeline {

    public:
        DdgiPipeline(VulkanContext& ctx, VkCommandPool cmdPool);
        ~DdgiPipeline();
        DdgiPipeline(const DdgiPipeline&) = delete;
        DdgiPipeline& operator=(const DdgiPipeline&) = delete;

        // Atlas / grid geometry (read by the renderer when it wires + dispatches).
        [[nodiscard]] uint32_t atlasWidth()  const { return atlasW_; }
        [[nodiscard]] uint32_t atlasHeight() const { return atlasH_; }
        [[nodiscard]] int      totalProbes() const { return totalProbes_; }

    private:
        static constexpr uint32_t kPingPong = 2;

        VulkanContext& ctx_;
        VkCommandPool  cmdPool_ = VK_NULL_HANDLE;

        // Probe grid + octahedral atlas geometry (defaults from vulkan_shared.h;
        // the grid origin/spacing get sized to the scene AABB in a later pass).
        int probesX_ = 0, probesY_ = 0, probesZ_ = 0;
        int totalProbes_ = 0, raysPerProbe_ = 0;
        int irrRes_ = 0, border_ = 0, tileSide_ = 0, tilesPerRow_ = 0, tileRows_ = 0;
        uint32_t atlasW_ = 0, atlasH_ = 0;

        // Resources.
        std::array<Image2D, kPingPong> irradiance_{}; // ping-pong octahedral irradiance atlas
        Buffer rayRadiance_{};                        // per (probe,ray) radiance + hitDist
        Buffer ddgiUbo_{};                            // grid params + per-frame ray rotation

        // Update (ray tracing) pipeline — own layout, not the shared RT set.
        VkDescriptorSetLayout updateDsLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout      updateLayout_   = VK_NULL_HANDLE;
        VkPipeline            updatePipeline_ = VK_NULL_HANDLE;
        Buffer                          sbtBuf_{};
        VkStridedDeviceAddressRegionKHR rgenRgn_{};
        VkStridedDeviceAddressRegionKHR missRgn_{};
        VkStridedDeviceAddressRegionKHR hitRgn_{};
        VkStridedDeviceAddressRegionKHR callRgn_{};

        // Blend (compute) pipeline.
        VkDescriptorSetLayout blendDsLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout      blendLayout_   = VK_NULL_HANDLE;
        VkPipeline            blendPipeline_  = VK_NULL_HANDLE;

        // Descriptor pool + sets (allocated now; populated when DDGI is enabled).
        VkDescriptorPool descPool_  = VK_NULL_HANDLE;
        VkDescriptorSet  updateSet_ = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kPingPong> blendSets_{};

        void computeGridDims();
        void createImages();
        void createBuffers();
        void createUpdatePipeline();
        void createSbt();
        void createBlendPipeline();
        void createDescriptors();
        void transitionFreshImage(VkImage img);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DDGI_PIPELINE_HPP
