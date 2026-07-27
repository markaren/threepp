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
    struct OverlayRecordScratch;// per-record draw lists, reused across frames

    class OverlayPass {
    public:
        // Callback type that wraps Impl::createSampledImage2DInFrame. Called
        // from ensureSpriteAtlasTexture when a new or stale atlas needs to be
        // uploaded: the upload is RECORDED into the frame's own command
        // buffer (before the overlay's render-pass instance opens) instead of
        // a one-shot submit — a mid-record one-shot's vkQueueWaitIdle drains
        // every in-flight frame, which turned each HUD-text re-rasterization
        // (ammo counters) into a 40-50 ms hitch.
        using SampledImageCreator = std::function<Image2D(
                VkCommandBuffer cb,
                uint32_t w, uint32_t h, VkFormat fmt,
                const void* pixels, VkDeviceSize byteSize,
                VkFilter filter,
                VkSamplerAddressMode addrU,
                VkSamplerAddressMode addrV,
                const char* debugName)>;

        // Hands a stale sprite-atlas Image2D back to the renderer's frame-serial
        // retire queue instead of a per-swap vkDeviceWaitIdle. The lambda
        // captures Impl and forwards to Impl::retire. Optional: if unset,
        // ensureSpriteAtlasTexture falls back to drain+destroy.
        using RetireImageFn = std::function<void(Image2D&&)>;

        OverlayPass(VulkanContext& ctx, uint32_t framesInFlight, SampledImageCreator uploadFn,
                    RetireImageFn retireFn = {});
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
        //                   compositing after the deferred-shaded frame body).
        // regionW == 0 → full frame. Otherwise the overlay is clipped to the
        // swapchain sub-rect (regionX, regionY, regionW, regionH) — used for
        // split-screen secondary panes (overlay-only, drawn beside the
        // primary deferred-render pane).
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
        // ensureSpriteAtlasTexture records any needed (re)upload into `cb`,
        // so it must run OUTSIDE a render-pass instance. Pass
        // cb == VK_NULL_HANDLE for a lookup-only call (inside the pass):
        // cache hits return normally, anything needing an upload returns
        // nullptr (record()'s pre-pass hoist has already uploaded every
        // atlas in this frame's draw list, so that shouldn't happen).
        const SpriteAtlasRec* ensureSpriteAtlasTexture(const std::shared_ptr<Texture>& texSp,
                                                       VkCommandBuffer cb);
        const SpriteGeomRec*  ensureSpriteGeometryUploaded(const BufferGeometry* geom);
        const LineRec*        ensureLineGeometryUploaded(const BufferGeometry* geom);

        VulkanContext&      ctx_;
        uint32_t            framesInFlight_;
        SampledImageCreator uploadFn_;
        RetireImageFn       retireFn_;

        // Draw lists gathered each record() call. Held here (not local vectors)
        // so their heap storage is reused frame to frame instead of realloc'd.
        std::unique_ptr<OverlayRecordScratch> scratch_;

        static constexpr uint32_t kMaxSpritesPerFrame = 64;

        // Sprite pipeline
        VkDescriptorSetLayout spriteDescSetLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      spritePipelineLayout_  = VK_NULL_HANDLE;
        VkPipeline            overlaySpritePipeline_ = VK_NULL_HANDLE;

        // Ortho line / mesh pipelines (overlay.vert/frag, depth-off)
        VkPipelineLayout orthoLinePipelineLayout_      = VK_NULL_HANDLE;
        VkPipeline       orthoLineListPipeline_        = VK_NULL_HANDLE;
        VkPipeline       orthoLineStripPipeline_       = VK_NULL_HANDLE;
        // Vertex-colored variants (overlay_color shaders, pos + color
        // bindings) — without them a GridHelper/AxesHelper in a composed
        // secondary pane loses its vertex colors and draws in the flat
        // material color (white, for the helpers' default materials).
        VkPipeline       orthoLineListColoredPipeline_  = VK_NULL_HANDLE;
        VkPipeline       orthoLineStripColoredPipeline_ = VK_NULL_HANDLE;
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
