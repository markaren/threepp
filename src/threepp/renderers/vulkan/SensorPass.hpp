// SensorPass — the camera's final image formation: lens distortion + sensor
// noise, applied to the FINISHED frame on its way to the swapchain.
//
// This runs LAST, after the overlay pass, and the reason is correctness rather
// than convenience (see sensor_image.comp for the full argument):
//
//   * a lens bends everything the camera sees, including the particle
//     billboards, lines and wireframe the overlay pass composites onto the
//     swapchain AFTER the TAA resolve. Warping earlier (in RCAS) distorted the
//     scene but not those, so overlays slid off the geometry they belong to;
//   * TAA and the DLSS/FSR upscalers average successive frames, so noise added
//     before the resolve gets filtered back out.
//
// It also keeps the overlay's depth test honest: that test runs against the
// raster G-buffer's depth in UNDISTORTED screen space, which is where the
// overlay geometry still is when the warp happens afterwards.
//
// MECHANISM. A gather warp cannot run in place, so `record` snapshots the
// finished swapchain image into an owned scratch (one vkCmdCopyImage), then
// dispatches sensor_image.comp to sample that scratch and write the swapchain.
// This costs a display-extent copy per frame — but ONLY while a lens or noise
// is actually configured; `active()` is false otherwise and the caller skips
// the pass entirely, leaving the default path untouched.
//
// Snapshotting the swapchain rather than retargeting the upstream passes is
// deliberate: TaaResolve's RCAS/finalize descriptors are swapchain-image
// indexed, and the overlay pass attaches the swapchain view directly, so
// redirecting them all would have meant threading an alternate render target
// through several passes that have no other reason to know about one.

#ifndef THREEPP_VULKAN_SENSOR_PASS_HPP
#define THREEPP_VULKAN_SENSOR_PASS_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class SensorPass {

    public:
        SensorPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight);
        ~SensorPass();
        SensorPass(const SensorPass&) = delete;
        SensorPass& operator=(const SensorPass&) = delete;

        // Lens + sensor state for one frame. Filled by the caller from the
        // renderer's lens/noise settings (Impl::buildSensorParams).
        struct Params {
            bool     distortActive = false;
            bool     noiseActive   = false;
            uint32_t lensModel     = 0u;         // threepp::LensModel
            float    radial[4]     = {0, 0, 0, 0};// k1..k4
            float    tangential[2] = {0, 0};      // p1, p2
            // Normalized intrinsics (fx/W, fy/H, cx/W, cy/H) — resolution
            // independent, so render-extent values stay correct here at the
            // display extent.
            float    normK[4]  = {1.f, 1.f, 0.5f, 0.5f};
            float    overscan  = 1.f;
            uint32_t frameSeed = 0u;
            float    fullWell      = 20000.f;
            float    readNoise     = 3.f;
            float    darkElectrons = 0.f;
            float    prnu          = 0.f;
            float    isoGain       = 1.f;

            [[nodiscard]] bool active() const { return distortActive || noiseActive; }
        };

        // (Re)allocate the snapshot scratch at the swapchain extent. Idempotent;
        // a no-op when the size is unchanged. Safe to call every frame — the
        // caller does, lazily, so a renderer that never sets a lens or noise
        // never pays the allocation.
        void resize(uint32_t width, uint32_t height);

        // Snapshot `swapImage` into the scratch, then warp/noise it back into
        // `swapView`. The swapchain image must be in GENERAL on entry and is
        // left in GENERAL. No-op when `p.active()` is false.
        void record(VkCommandBuffer cb, uint32_t frame,
                    VkImage swapImage, VkImageView swapView,
                    uint32_t width, uint32_t height, const Params& p);

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;

        VkSampler sampler_ = VK_NULL_HANDLE;// LINEAR clamp — the warp resamples

        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        std::vector<Image2D> snapshot_;// [framesInFlight]
        uint32_t width_ = 0, height_ = 0;
        // The swapchain view each frame slot's descriptor currently points at;
        // the swapchain image index varies per frame, so the destination write
        // is refreshed only when it actually changes.
        std::vector<VkImageView> boundDst_;

        void createPipeline();
        void createDescriptorPool();
        void destroySnapshots();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_SENSOR_PASS_HPP
