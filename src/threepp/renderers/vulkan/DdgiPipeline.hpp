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

        // Wired into the shared RT descriptor set so closest_hit's sampleDDGI
        // can read this frame's irradiance atlas + the grid uniform. The atlas
        // is read as a storage image (GENERAL layout) — no sampler needed.
        [[nodiscard]] VkImageView irradianceView(uint32_t frame) const {
            return irradiance_[frame % kPingPong].view;
        }
        [[nodiscard]] VkBuffer uboBuffer() const { return ddgiUbo_.handle; }

        // Override the probe-grid placement (world AABB the grid spans). Until
        // a scene-AABB auto-sizer lands, the renderer can call this; otherwise
        // the constructor's defaults are used.
        void setGrid(const float originXYZ[3], const float spacingXYZ[3]);

        // Per-frame passes. recordUpdate casts the probe rays into `tlas`
        // (env sampled in the miss shader) and fills the ray-radiance buffer;
        // recordBlend folds them into the irradiance atlas. Both are no-ops at
        // the renderer level until ddgiEnabled_ — the renderer simply doesn't
        // call them otherwise. `frame` is the frame-in-flight index (parity
        // selects the double-buffered ray buffer / descriptor set / atlas).
        void recordUpdate(VkCommandBuffer cb, VkAccelerationStructureKHR tlas,
                          VkImageView envView, VkSampler envSampler, uint32_t frame);
        void recordBlend(VkCommandBuffer cb, uint32_t frame);

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

        // Probe-grid world placement (defaults set in computeGridDims; the
        // renderer may override via setGrid once it knows the scene AABB).
        float gridOrigin_[3]  = {0.f, 0.f, 0.f};
        float gridSpacing_[3] = {1.f, 1.f, 1.f};

        // Resources. The ray buffer + per-frame descriptor sets are
        // double-buffered by frame parity so frame N+1's update can't clobber
        // data frame N's blend is still reading. The atlas is per-frame for the
        // same reason on the sampling side. The UBO is static (grid params,
        // identity ray rotation for now) so a single copy is race-free.
        std::array<Image2D, kPingPong> irradiance_{};   // per-frame octahedral irradiance atlas
        std::array<Buffer,  kPingPong> rayRadiance_{};  // per (probe,ray) radiance + hitDist
        Buffer ddgiUbo_{};                              // grid params (static for now)
        bool   uboWritten_ = false;

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

        // Descriptor pool + sets (allocated up front; their contents are
        // written lazily on the first recordUpdate/recordBlend per parity).
        VkDescriptorPool descPool_ = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kPingPong> updateSets_{};
        std::array<VkDescriptorSet, kPingPong> blendSets_{};
        std::array<bool, kPingPong> updateWired_{};
        std::array<bool, kPingPong> blendWired_{};

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
