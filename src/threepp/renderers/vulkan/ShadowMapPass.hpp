// ShadowMapPass — one view's shadow maps (VulkanRenderer::ShadowMode::ShadowMaps).
//
// Per frame slot: the sun's cascades (a D32 2D array, one layer per cascade),
// the spot lamps' maps (a D32 2D array, one layer per slot) and the point
// lamps' cubes (a D32 cube array, six layers per slot), the GpuShadowMapsUbo
// the receivers read (shaders/shadow_maps.glsl), and the caster list the
// maps are drawn from: a DrawInfo buffer and indirect commands in the
// G-buffer's own format (gbuffer_indirect.vert's DrawInfo), bound through a
// raster descriptor set of its own at binding 4.
//
// The renderer fills it each frame (VulkanCoreShadows.cpp: the fit, the lamp
// cameras, the caster list) and the graph's "shadowMaps" pass records it:
// one dynamic-rendering depth pass per layer, the caster list drawn into
// each, reverse-Z, cleared to 0 (far).
//
// Slots are reallocated one at a time, each after its own fence, so a size or
// count change never touches an image a frame in flight still reads; each
// reallocation takes a new generation number, which is how the deferred set
// knows its bindings 80-83 are stale.

#ifndef THREEPP_VULKAN_SHADOW_MAP_PASS_HPP
#define THREEPP_VULKAN_SHADOW_MAP_PASS_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    // A layered D32 depth target: the whole image as one sampled view (2D
    // array, or cube array for a cube-compatible image) and, when it is
    // rendered into, one 2D view per layer.
    struct ShadowDepthArray {
        VkImage                  image = VK_NULL_HANDLE;
        VmaAllocation            alloc = VK_NULL_HANDLE;
        VkImageView              view  = VK_NULL_HANDLE;
        std::vector<VkImageView> layerViews;
        uint32_t                 size   = 0;
        uint32_t                 layers = 0;
        bool                     cube   = false;
    };
    // layers: for a cube, 6 x cubes. renderable: the per-layer views; otherwise
    // TRANSFER_DST usage (a dummy is cleared, never drawn).
    ShadowDepthArray createShadowDepthArray(VulkanContext& ctx, uint32_t size, uint32_t layers, bool cube,
                                            bool renderable, const char* name);
    void destroyShadowDepthArray(VulkanContext& ctx, ShadowDepthArray& a);

    class ShadowMapPass {
    public:
        ShadowMapPass(VulkanContext& ctx, uint32_t framesInFlight, VkDescriptorSetLayout rasterLayout);
        ~ShadowMapPass();
        ShadowMapPass(const ShadowMapPass&)            = delete;
        ShadowMapPass& operator=(const ShadowMapPass&) = delete;

        struct Sizes {
            uint32_t cascadeSize = 0, cascades = 0;
            uint32_t spotSize = 0, spots = 0;  // spots = layers (0 = no array)
            uint32_t pointSize = 0, points = 0;// points = cubes (0 = no array)
            bool operator==(const Sizes&) const = default;
        };
        // After the call slot `frame` holds images of exactly these sizes.
        // Reallocates that slot only; the caller has waited for its fence.
        void ensure(uint32_t frame, const Sizes& sizes);

        struct Slot {
            ShadowDepthArray cascades, spots, points;
            Sizes            sizes{};
            Buffer           ubo;     // GpuShadowMapsUbo, host-visible, mapped
            void*            uboMapped = nullptr;
            Buffer           drawInfo;// DrawInfoGpu[], host-visible
            Buffer           cmds;    // VkDrawIndirectCommand[], host-visible
            VkDescriptorSet  rasterSet = VK_NULL_HANDLE;
            bool             matTexWritten = false;
            uint64_t         gen = 0;// nonzero once allocated; new on each reallocation
        };
        [[nodiscard]] Slot& slot(uint32_t frame) { return slots_[frame]; }
        [[nodiscard]] const Slot& slot(uint32_t frame) const { return slots_[frame]; }

        // Grow the slot's caster buffers to hold `draws` records; true when the
        // DrawInfo buffer moved (binding 4 of the slot's raster set was then
        // rewritten here).
        bool ensureDrawCapacity(uint32_t frame, uint32_t draws);
        // Bindings 2 (MaterialDesc[]) and, when matTex is given, 3 (the
        // bindless albedo maps) of the slot's raster set: the shadow fragment
        // shader's alpha test reads them.
        void writeRasterMaterials(uint32_t frame, VkBuffer materials, const VkDescriptorImageInfo* matTex,
                                  uint32_t matTexCount);

        // This frame's layers, in record order: the cascades, then the spots,
        // then six faces per point cube.
        enum class Target : uint32_t { Cascade, Spot, PointFace };
        struct Layer {
            float    vp[16];// world -> clip, reverse-Z; a point face's has its y pre-flipped
            Target   target = Target::Cascade;
            uint32_t layer  = 0;// layer of its array
        };
        std::vector<Layer> layers;
        // The caster list's cull-mode buckets: [BACK, FRONT, NONE].
        struct Group {
            uint32_t offset = 0, count = 0;
        };
        std::array<Group, 3> groups{};
        uint32_t             totalDraws = 0;
        // vkCmdSetDepthBias for the raster side (reverse-Z: negative pushes the
        // stored depth away from the light).
        float depthBiasConstant = 0.f, depthBiasSlope = 0.f;

        // Records every layer; the graph has put the images in
        // DEPTH_ATTACHMENT_OPTIMAL.
        void record(VkCommandBuffer cb, uint32_t frame, VkPipeline pipeline, VkPipelineLayout layout) const;

    private:
        VulkanContext&        ctx_;
        VkDescriptorPool      pool_ = VK_NULL_HANDLE;
        std::vector<Slot>     slots_;
        uint64_t              genCounter_ = 0;

        void freeImages(Slot& s);
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_SHADOW_MAP_PASS_HPP
