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
//
// The maps are dead between frames (drawn before the G-buffer, read up to the
// particle lighting), so they live in the renderer's TransientPool when it has
// one: live from TransientPhase::Gbuffer to Filter of their view, one frame
// slot each, so the two slots' maps share memory and the post chain's scratch
// may sit in it too.

#ifndef THREEPP_VULKAN_SHADOW_MAP_PASS_HPP
#define THREEPP_VULKAN_SHADOW_MAP_PASS_HPP

#include "threepp/renderers/vulkan/AliasPacker.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace threepp::vulkan {

    class TransientPool;
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
    // TRANSFER_DST usage (a dummy is cleared, never drawn). pool: bound into it
    // with `span` and `slots` (TransientPool), else an allocation of its own.
    ShadowDepthArray createShadowDepthArray(VulkanContext& ctx, uint32_t size, uint32_t layers, bool cube,
                                            bool renderable, const char* name, TransientPool* pool = nullptr,
                                            TransientSpan span = 0, uint32_t slots = kTransientAllSlots);
    void destroyShadowDepthArray(VulkanContext& ctx, ShadowDepthArray& a);

    class ShadowMapPass {
    public:
        // pool: the renderer's TransientPool, or null (THREEPP_VK_NO_ALIAS);
        // view: the view's id, for the maps' span. retireImage: the renderer's
        // retire queue, which a reallocated static cache is handed to (both
        // slots read it, so the other one may still have it in flight).
        ShadowMapPass(VulkanContext& ctx, uint32_t framesInFlight, VkDescriptorSetLayout rasterLayout,
                      TransientPool* pool, uint32_t view, std::function<void(Image2D&&)> retireImage);
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
        // Reallocates that slot only; the caller has waited for its fence. A
        // static cache that changes size is retired, not destroyed.
        void ensure(uint32_t frame, const Sizes& sizes);
        // ensure(frame, sizes) would allocate nothing: the slot holds them.
        [[nodiscard]] bool ready(uint32_t frame, const Sizes& sizes) const {
            return slots_[frame].gen != 0 && slots_[frame].sizes == sizes;
        }

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

        // Grow the slot's caster buffers to hold `draws` records and `cmds`
        // indirect commands; true when the DrawInfo buffer moved (binding 4 of
        // the slot's raster set was then rewritten here).
        bool ensureDrawCapacity(uint32_t frame, uint32_t draws, uint32_t cmds);
        // Bindings 2 (the material table), 5 (the GeometryDescs, whose flags
        // word names an entry's record) and, when matTex is given, 3
        // (the bindless albedo maps) of the slot's raster set: the shadow
        // fragment shader's alpha test reads them.
        void writeRasterMaterials(uint32_t frame, VkBuffer materials, VkBuffer geometryDescs,
                                  const VkDescriptorImageInfo* matTex, uint32_t matTexCount);

        // This frame's layers, in record order: the cascades, then the spots,
        // then six faces per point cube.
        enum class Target : uint32_t { Cascade, Spot, PointFace };
        // How a layer is drawn. Draw: cleared, every caster drawn. Rebuild:
        // cleared, the static casters drawn and the result copied into the
        // cache's layer `cacheLayer`, then the moving casters on top. Cached:
        // the cache's layer copied in, then the moving casters on top. A
        // cascade's frozen box is what makes the cache valid (fitShadowMaps).
        enum class Mode : uint32_t { Draw, Rebuild, Cached };
        struct Layer {
            float    vp[16];// world -> clip, reverse-Z (a point face is drawn without the y flip)
            Target   target     = Target::Cascade;
            uint32_t layer      = 0;    // layer of its array
            bool     skip       = false;// the view cannot see it: cleared, nothing drawn
            Mode     mode       = Mode::Draw;
            uint32_t cacheLayer = 0;
        };
        std::vector<Layer> layers;
        // Per layer, the caster list's buckets: [BACK, FRONT, NONE] drawn
        // depth only, then the same three through the alpha test (a cutout
        // caster within the cutout ray distance, see buildShadowDrawData);
        // `stat` for the casters that hold still, `dyn` for the moving ones
        // (deformers, recently moved meshes), which a cached layer draws
        // alone. Each is a range of the slot's command buffer; the commands
        // name their DrawInfo record by firstInstance, so a caster that
        // reaches several layers has one record and one command per layer.
        struct Group {
            uint32_t offset = 0, count = 0;
        };
        static constexpr uint32_t kGroups = 6;
        struct LayerDraws {
            std::array<Group, kGroups> stat{}, dyn{};
        };
        std::vector<LayerDraws> layerGroups;// [layers.size()]
        uint32_t                totalDraws = 0;// DrawInfo records
        uint32_t                totalCmds  = 0;// indirect commands over every layer

        // The static cache of the sun's cascades: one persistent D32 layer per
        // cascade (its own allocation, never aliased), in GENERAL for its whole
        // life, holding the static casters' depth for the frozen box in
        // `cascadeCache[i]`. Sized with the slots' cascade arrays (ensure).
        ShadowDepthArray cache;
        struct CascadeCache {
            bool     valid = false;
            float    c[3]{}, r = 0.f, texel = 0.f, far = 0.f;// the frozen box
            float    sunDir[3]{};
            uint64_t staticGen = 0;// the static caster set it was drawn for
            float    vp[16]{};
        };
        std::array<CascadeCache, 4> cascadeCache{};
        // The cascades' stale-tolerant mode (VulkanRenderer::
        // setShadowCacheStaleFrames), where a changed static caster set no
        // longer redraws a cascade at once. Per entry, bit c: a static caster
        // when cascade c was last drawn (in its cache, or outside its box);
        // bit 4 + c: it reached cascade c's box then. A caster that leaves the
        // static set (moves, hides, stops casting) clears its bits and, if it
        // reached a box, leaves its old depth in that cache: the cascade is
        // stale. A static caster without its bit (it settled or started
        // casting since) is drawn over the cache each frame, which makes the
        // cascade stale too. A cascade stale since staleSince[c] (a frame
        // serial, 0 = not stale) is redrawn after the set number of frames.
        // `onTop`: the entries drawn over the caches this frame, in span
        // order (the moving casters and the static ones without their bit).
        std::vector<uint8_t>    casterSnap;
        std::array<uint64_t, 4> staleSince{};
        uint64_t                snapLayout = ~0ull;// the entry numbering the bits index
        std::vector<uint32_t>   onTop;
        // The lamps' caches: one 2D layer per spot slot (spotSize) and per
        // point-cube face (pointSize), same life and layout as `cache`. A
        // lamp layer is valid while its light clip matrix is the same bits
        // (the lamp has not moved, its cone and range hold) and the static
        // caster set is the one it was drawn for; a layer the view cannot see
        // keeps its entry and is served again when it comes back.
        ShadowDepthArray spotCache, pointCache;
        struct LampCache {
            bool     valid     = false;
            uint64_t staticGen = 0;
            float    vp[16]{};
        };
        std::vector<LampCache> spotCacheState, pointFaceCacheState;// [spots], [6 * points]
        void invalidateCache() {
            for (auto& c : cascadeCache) c.valid = false;
            for (auto& c : spotCacheState) c.valid = false;
            for (auto& c : pointFaceCacheState) c.valid = false;
        }
        // vkCmdSetDepthBias for the raster side (reverse-Z: negative pushes the
        // stored depth away from the light).
        float depthBiasConstant = 0.f, depthBiasSlope = 0.f;

        // Records every layer; the graph has put the images in
        // DEPTH_ATTACHMENT_OPTIMAL. depthOnly: no fragment stage (groups 0-2);
        // alphaTest: shadow_depth.frag's cutout test (groups 3-5).
        void record(VkCommandBuffer cb, uint32_t frame, VkPipeline depthOnly, VkPipeline alphaTest,
                    VkPipelineLayout layout);

    private:
        VulkanContext&        ctx_;
        TransientPool*        transient_ = nullptr;
        uint32_t              view_      = 0;
        std::function<void(Image2D&&)> retireImage_;
        VkDescriptorPool      pool_ = VK_NULL_HANDLE;
        std::vector<Slot>     slots_;
        uint64_t              genCounter_ = 0;
        bool cacheReady_ = false, spotCacheReady_ = false, pointCacheReady_ = false;// each cache image's one-time move to GENERAL was recorded

        void freeImages(Slot& s);
        void retireCache(ShadowDepthArray& c);
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_SHADOW_MAP_PASS_HPP
