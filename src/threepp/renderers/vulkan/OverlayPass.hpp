// OverlayPass — ortho/HUD overlay rendering (Sprites, Lines, Meshes).
//
// Manages the ortho sprite pipeline, the ortho line/mesh pipelines, per-frame
// descriptor pools for per-sprite atlas binding, and the three geometry caches
// (sprite atlas, sprite quad geometry, line/mesh geometry). recordOrthoOverlay
// is the only entry point; it is self-contained and replaces the same-named
// Impl method it was extracted from.
//
// Atlas texture creation is delegated back to the caller via SampledImageCreator
// because createSampledImage2D is shared across 10+ Impl call sites and cannot
// move here alone. The lambda supplied at construction captures Impl's
// beginOneShot/endAndSubmitOneShot transparently.
//
// Extracted from VulkanRenderer.cpp during the file split.

#ifndef THREEPP_VULKAN_OVERLAY_PASS_HPP
#define THREEPP_VULKAN_OVERLAY_PASS_HPP

#include "threepp/renderers/vulkan/VulkanFrameTypes.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/textures/Texture.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace threepp {
    class Object3D;
    class Camera;
}

namespace threepp::vulkan {

    class VulkanContext;

    class OverlayPass {
    public:
        // Callback type that wraps Impl::createSampledImage2D. Called from
        // ensureSpriteAtlasTexture when a new or stale atlas needs to be
        // uploaded; the lambda stored here captures Impl* and forwards to the
        // shared helper so OverlayPass does not need beginOneShot/endOneShot.
        using SampledImageCreator = std::function<Image2D(
                uint32_t w, uint32_t h, VkFormat fmt,
                const void* pixels, VkDeviceSize byteSize,
                VkFilter filter,
                VkSamplerAddressMode addrU,
                VkSamplerAddressMode addrV,
                const char* debugName)>;

        OverlayPass(VulkanContext& ctx, uint32_t framesInFlight, SampledImageCreator uploadFn);
        ~OverlayPass();
        OverlayPass(const OverlayPass&)            = delete;
        OverlayPass& operator=(const OverlayPass&) = delete;

        // Record the entire ortho/HUD overlay into `cb`.
        // cb       — the per-frame command buffer (already open, in GENERAL layout).
        // frame    — which frame-in-flight slot to use (descriptor pool, etc.).
        // imageIndex — which swapchain image/view to render into.
        // scene / camera — the ortho HUD scene and its camera.
        // screenSpaceOnly — when true, only sprites with Sprite::screenSpace=true
        //                   are drawn (used for the automatic screen-space sprite
        //                   compositing after the PT body).
        // regionW == 0 → full frame. Otherwise the overlay is clipped to the
        // swapchain sub-rect (regionX, regionY, regionW, regionH) — used for
        // split-screen secondary panes (overlay-only, drawn beside a PT pane).
        void record(VkCommandBuffer cb, uint32_t frame, uint32_t imageIndex,
                    Object3D& scene, Camera& camera, bool screenSpaceOnly,
                    uint32_t regionX = 0, uint32_t regionY = 0,
                    uint32_t regionW = 0, uint32_t regionH = 0);

    private:
        // Cached uploaded sprite atlas. Keyed on Texture*; liveCheck detects
        // pointer recycle; textureVersion mirrors Texture::version() so
        // setText()-triggered re-rasterisation forces a re-upload.
        struct SpriteAtlasRec {
            Image2D      image{};
            unsigned int textureVersion = ~0u;
            uint32_t     width          = 0;
            uint32_t     height         = 0;
            std::weak_ptr<Texture> liveCheck;
        };

        // Per-BufferGeometry vertex/index upload for Sprite quads.
        struct SpriteGeomRec {
            Buffer   vertex;
            Buffer   index;
            uint32_t indexCount = 0;
            std::weak_ptr<BufferGeometry> liveCheck;
        };

        // Lazy pipeline setup — called from record() on first use.
        void createSpriteOverlayPipeline();
        void createOrthoLinePipelines();
        void createOrthoPointPipeline();

        // Cache helpers — called from record() on each draw.
        const SpriteAtlasRec* ensureSpriteAtlasTexture(const std::shared_ptr<Texture>& texSp);
        const SpriteGeomRec*  ensureSpriteGeometryUploaded(const BufferGeometry* geom);
        const LineRec*        ensureLineGeometryUploaded(const BufferGeometry* geom);

        VulkanContext&      ctx_;
        uint32_t            framesInFlight_;
        SampledImageCreator uploadFn_;

        static constexpr uint32_t kMaxSpritesPerFrame = 64;

        // Sprite pipeline
        VkDescriptorSetLayout spriteDescSetLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      spritePipelineLayout_  = VK_NULL_HANDLE;
        VkPipeline            overlaySpritePipeline_ = VK_NULL_HANDLE;

        // Ortho line / mesh pipelines (overlay.vert/frag, depth-off)
        VkPipelineLayout orthoLinePipelineLayout_      = VK_NULL_HANDLE;
        VkPipeline       orthoLineListPipeline_        = VK_NULL_HANDLE;
        VkPipeline       orthoLineStripPipeline_       = VK_NULL_HANDLE;
        VkPipeline       orthoMeshPipeline_            = VK_NULL_HANDLE;
        VkPipeline       orthoMeshTransparentPipeline_ = VK_NULL_HANDLE;

        // Ortho point pipeline (overlay_point.vert/frag, POINT_LIST, pos+color
        // vertex bindings, depth-off). Reuses orthoLinePipelineLayout_.
        VkPipeline       orthoPointListPipeline_       = VK_NULL_HANDLE;

        // Per-frame descriptor pools reset at the top of each record() call.
        std::vector<VkDescriptorPool> spriteDescPools_;

        // Texture + geometry caches
        std::unordered_map<const Texture*,        SpriteAtlasRec> spriteAtlasCache_;
        std::unordered_map<const BufferGeometry*, SpriteGeomRec>  spriteGeomCache_;
        std::unordered_map<const BufferGeometry*, LineRec> lineGeomCache_;
        uint64_t overlayFrameCounter_ = 0;
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_OVERLAY_PASS_HPP
