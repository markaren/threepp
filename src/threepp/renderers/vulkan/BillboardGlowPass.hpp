// BillboardGlowPass — a second, much smaller bloom pyramid, for the content the
// scene's pyramid structurally cannot see.
//
// plans/particle-atmosphere.md F4, item 1.
//
// ── THE PROBLEM ─────────────────────────────────────────────────────────────
// ParticleField billboards composite AFTER recordUpscaleAndPost, on the
// swapchain, in the overlay slot. F3 note 1 argues that placement at length and
// it is worth restating, because everything here is shaped by NOT giving it up:
// a 3-px spark that crosses 20 px in a frame is the exact content a temporal
// resolve mis-handles, so drawing it outside TAA/DLSS/FSR is not a convenience,
// it is why field billboards need no disocclusion tuning at all.
//
// BloomPass runs long before that point, on sceneHdr, so the sparks' radiance
// is not in its input and no amount of brightening a quad will make it bloom.
// Moving the composite earlier would buy the bloom and forfeit the temporal
// property. Instead this pass gives the billboards a pyramid of their own:
//
//   1. the glow-enabled fields are drawn a SECOND time, into `src` — a
//      half-display-extent linear-HDR colour target (the billboard fragment
//      stage skips its display transform when it renders there);
//   2. the SHARED bloom_down.comp / bloom_up.comp are run over `src` into a
//      quarter-extent pyramid — the same Jimenez progressive chain, the same
//      SPIR-V, no new shader;
//   3. the renderer's fullscreen composite draw (particlefield_glow.frag) adds
//      level 0 into the swapchain INSIDE the overlay render-pass instance, next
//      to the sharp quads. The composite point does not move.
//
// ── WHY HALF EXTENT, AND WHY NO DEPTH ───────────────────────────────────────
// A glow is by definition low-frequency; the pyramid's finest level is already
// a quarter of the frame, so feeding it a half-extent source loses nothing that
// survives the first 13-tap downsample. It costs a quarter of the memory a
// full-extent target would, and it is allocated LAZILY — a scene with no glow
// field never creates it.
//
// The source pass carries NO depth attachment, so a spark behind a wall still
// contributes its halo. That is a deliberate trade rather than an oversight: a
// depth test at half extent needs a half-extent depth buffer (the overlay's
// unjittered depth is full extent, and a render area must fit its attachments),
// which is a second image and a reduction pass to fill it, for an artefact that
// bloom's own defining behaviour — bleeding over silhouettes — half-hides
// anyway. If an occluded emitter ever reads wrong, the fix is that reduction,
// not a full-extent target.
//
// ── DESCRIPTOR DISCIPLINE ───────────────────────────────────────────────────
// Every set here is allocated once and written once per resize, never per
// frame: the images are per-frame-in-flight and the sets that name them are
// too, so nothing is ever written while a frame naming it is in flight
// (VUID-03047). The pass owns its own pool for exactly that reason — it shares
// nothing with the overlay's per-frame texture pool, whose reset-discipline bug
// F3 note 4 records.

#ifndef THREEPP_VULKAN_BILLBOARD_GLOW_PASS_HPP
#define THREEPP_VULKAN_BILLBOARD_GLOW_PASS_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class BillboardGlowPass {

    public:
        BillboardGlowPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight);
        ~BillboardGlowPass();
        BillboardGlowPass(const BillboardGlowPass&) = delete;
        BillboardGlowPass& operator=(const BillboardGlowPass&) = delete;

        // Allocate (or re-allocate) for a DISPLAY extent. Idempotent: a second
        // call with the same extent is a no-op, so the renderer may call it on
        // the first frame a glow field appears and every frame after.
        // False when the extent is degenerate.
        bool ensureImages(uint32_t displayWidth, uint32_t displayHeight);
        void destroyImages();

        [[nodiscard]] bool ready() const { return levels_ > 0; }

        // The half-extent linear-HDR colour target the billboards render into.
        [[nodiscard]] VkImage     srcImage(uint32_t frame) const { return src_[frame].image; }
        [[nodiscard]] VkImageView srcView(uint32_t frame) const { return src_[frame].view; }
        [[nodiscard]] VkExtent2D  srcExtent() const { return {srcW_, srcH_}; }

        // Level 0 of the pyramid — what the composite draw samples. The set is
        // pre-written per frame-in-flight; the renderer binds it and adds.
        [[nodiscard]] VkDescriptorSet compositeSet(uint32_t frame) const {
            return compositeSets_[frame];
        }
        [[nodiscard]] VkDescriptorSetLayout compositeSetLayout() const {
            return compositeDsLayout_;
        }
        // Runtime level count. The composite divides its intensity by this, so
        // the summed pyramid carries the same total energy whatever the extent
        // made the chain depth — the same normalisation PostComposite applies
        // to the scene pyramid.
        [[nodiscard]] uint32_t levels() const { return levels_; }

        // Barrier the freshly-rendered `src` visible to compute, then run the
        // progressive downsample + upsample walk-back. Leaves level 0 barriered
        // for a FRAGMENT-stage sampled read (the composite draw), which is the
        // one place this differs from BloomPass's compute-to-compute chain.
        void recordPyramid(VkCommandBuffer cb, uint32_t frame, float threshold);

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;
        uint32_t       srcW_ = 0, srcH_ = 0;
        uint32_t       dispW_ = 0, dispH_ = 0;

        // Shallower than BloomPass's nine: the source is already half extent and
        // the chain starts at a quarter, so five levels reach ~1/64 of the frame
        // — wider than any spark's halo has business being.
        static constexpr uint32_t kMaxLevels = 5;
        uint32_t levels_ = 0;
        std::vector<Image2D> src_;// [framesInFlight]
        std::vector<Image2D> pyr_;// [framesInFlight × kMaxLevels]
        VkSampler sampler_ = VK_NULL_HANDLE;

        // The pyramid pipelines. Same SPIR-V, same 28 B push block and same
        // (sampler @0, storage @1) layout as BloomPass — reuse of the shaders,
        // not of the class, because BloomPass's images and descriptor sets are
        // welded to sceneHdr and the scene extent.
        VkDescriptorSetLayout bloomDsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      bloomPipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            downPipe_        = VK_NULL_HANDLE;
        VkPipeline            upPipe_          = VK_NULL_HANDLE;

        // Composite: one combined image sampler on pyramid level 0.
        VkDescriptorSetLayout compositeDsLayout_ = VK_NULL_HANDLE;

        VkDescriptorPool descPool_ = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> downSets_;
        std::vector<VkDescriptorSet> upSets_;
        std::vector<VkDescriptorSet> compositeSets_;

        Image2D createImage(uint32_t w, uint32_t h, VkImageUsageFlags usage, const char* label);
        void    transitionFreshImage(VkImage img);
        void    createPipelines();
        void    createDescriptors();
        void    rewriteDescriptors();
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_BILLBOARD_GLOW_PASS_HPP
