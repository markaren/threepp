// DofPass — thin-lens depth of field on the linear-HDR scene.
//
// Runs between the shade/denoise (sceneHdr final) and the bloom pyramid /
// PostComposite, so defocused highlights still bloom and tone-map as HDR.
// Consumes the raster G-buffer depth, which the hybrid prepass rasterizes
// for TAA anyway.
//
// Jimenez-style half-res scatter-as-gather (4 dispatches, ~fixed cost):
//   1. dof_coc     — half-res prefilter + signed thin-lens CoC from depth
//                    (closest-of-4 wins, keeps thin silhouettes near-field)
//   2. dof_tile ×2 — 8×8 tile max (near, far) + 3×3 dilate: bounds the
//                    gather radius so a big bokeh a tile away is found
//   3. dof_gather  — 49-tap disc → far field (background-classified,
//                    normalized) + near field (premult coverage that spills
//                    over in-focus background, as defocused foregrounds do)
//   4. dof_combine — full-res: exact per-pixel CoC re-eval, far blend, near
//                    composite; read-modify-writes sceneHdr in place.
//
// CoC comes from the CAMERA, not a strength slider: aperture = the f-number
// from setCameraExposure, focal length from the camera's vertical FOV on a
// 24 mm full-frame sensor, focus plane = setFocusDistance. cocScale =
// f²/(N·(S−f)) · (renderH/sensorH) · 0.5 is precomputed CPU-side per frame.
//
// The half-res scratch images are a SINGLE set (not per frame-in-flight):
// they are written and consumed inside one command buffer, and the leading
// pipeline barrier's first-synchronization-scope covers the previous
// frame's submission-order work on the queue, so cross-frame reuse is
// ordered without doubling ~12 MB of VRAM.

#ifndef THREEPP_VULKAN_DOF_PASS_HPP
#define THREEPP_VULKAN_DOF_PASS_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class DofPass {

    public:
        DofPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight);
        ~DofPass();
        DofPass(const DofPass&) = delete;
        DofPass& operator=(const DofPass&) = delete;

        // (Re)create the half-res scratch for `width × height` (render
        // extent) if it changed, and (re)write every descriptor set. One
        // view / buffer per frame-in-flight.
        struct ResizeInputs {
            const VkBuffer*    cameraUbos       = nullptr;// [framesInFlight]
            const VkImageView* depthPerFrame    = nullptr;// [framesInFlight] raster D32, read-only layout
            const VkImageView* sceneHdrPerFrame = nullptr;// [framesInFlight] rgba16f GENERAL
        };
        void resize(uint32_t width, uint32_t height, const ResizeInputs& in);

        [[nodiscard]] bool valid() const { return half_.image != VK_NULL_HANDLE; }

        // Record the 5 dispatches (leading sceneHdr write→read barrier
        // included; the trailing write is covered by the bloom/PostComposite
        // leading barriers). width/height = the frame's region render
        // extent; cocScale/focusDist/maxCocPx as documented above.
        void record(VkCommandBuffer cb, uint32_t frame,
                    uint32_t width, uint32_t height,
                    float cocScale, float focusDist, float maxCocPx);

    private:
        VulkanContext& ctx_;
        VkCommandPool  cmdPool_;
        uint32_t       framesInFlight_;
        uint32_t       width_ = 0, height_ = 0;// full-res extent the images fit
        uint32_t       halfW_ = 0, halfH_ = 0;
        uint32_t       tilesW_ = 0, tilesH_ = 0;

        Image2D half_{};// rgb + signed coc (half-res px), rgba16f
        Image2D far_{}; // far field, rgba16f
        Image2D near_{};// near field + coverage, rgba16f
        Image2D tileA_{};// (maxNear, maxFar) per 8×8 tile, rg16f
        Image2D tileB_{};// dilated tiles, rg16f

        VkSampler nearestSampler_ = VK_NULL_HANDLE;// texelFetch users
        VkSampler linearSampler_  = VK_NULL_HANDLE;// half-res field upsample

        VkDescriptorSetLayout cocLayout_     = VK_NULL_HANDLE;
        VkDescriptorSetLayout tileLayout_    = VK_NULL_HANDLE;
        VkDescriptorSetLayout gatherLayout_  = VK_NULL_HANDLE;
        VkDescriptorSetLayout combineLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout cocPipeLayout_     = VK_NULL_HANDLE;
        VkPipelineLayout tilePipeLayout_    = VK_NULL_HANDLE;
        VkPipelineLayout gatherPipeLayout_  = VK_NULL_HANDLE;
        VkPipelineLayout combinePipeLayout_ = VK_NULL_HANDLE;
        VkPipeline cocPipe_     = VK_NULL_HANDLE;
        VkPipeline tilePipe_    = VK_NULL_HANDLE;
        VkPipeline gatherPipe_  = VK_NULL_HANDLE;
        VkPipeline combinePipe_ = VK_NULL_HANDLE;

        VkDescriptorPool descPool_ = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> cocSets_;    // [framesInFlight]
        VkDescriptorSet              tileSet0_ = VK_NULL_HANDLE;// half → tileA
        VkDescriptorSet              tileSet1_ = VK_NULL_HANDLE;// tileA → tileB
        VkDescriptorSet              gatherSet_ = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> combineSets_;// [framesInFlight]

        Image2D createImage(uint32_t w, uint32_t h, VkFormat format, const char* label);
        void    destroyImages();
        void    createPipelines();
        void    createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DOF_PASS_HPP
