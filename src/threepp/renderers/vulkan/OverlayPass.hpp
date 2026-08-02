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

        // Same contract for the geometry caches' vertex/index/color/normal
        // buffers. The overlay re-uploads and evicts these MID-FRAME (a
        // detection-box overlay rebuilds its line geometry every frame), while
        // command buffers from the previous frames-in-flight can still be
        // reading them — freeing inline is a use-after-free with zero margin.
        // Optional: if unset, the buffer paths fall back to drain+destroy.
        using RetireBufferFn = std::function<void(Buffer&&)>;

        OverlayPass(VulkanContext& ctx, uint32_t framesInFlight, SampledImageCreator uploadFn,
                    RetireImageFn retireFn = {}, RetireBufferFn retireBufferFn = {});
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
        // swapchain sub-rect (regionX, regionY, regionW, regionH) — the
        // split-screen secondary pane, drawn beside (or docked over) the
        // primary deferred-render pane. A regioned pane is a VIEW, not a HUD:
        // its rect is cleared to the scene background and meshes with float
        // normals draw depth-tested and sun+ambient lit (the lit pane —
        // matches what GL's forward path does for a scissored render()).
        // Lines / Points / Sprites / normal-less meshes composite on top
        // through the flat path as before.
        void record(VkCommandBuffer cb, uint32_t frame, uint32_t imageIndex,
                    Object3D& scene, Camera& camera, bool screenSpaceOnly,
                    uint32_t regionX = 0, uint32_t regionY = 0,
                    uint32_t regionW = 0, uint32_t regionH = 0);

        // Environment for the lit pane's sky. Pass the renderer's envImage
        // view/sampler when the scene's environment is a REAL equirect
        // texture; VK_NULL_HANDLE keeps the solid background clear (a
        // background COLOUR must stay a verbatim clear — the deferred
        // solid-bg bypass shows that colour display-referred, and tone
        // mapping it here would make the two views disagree on "empty").
        // Call before record(); the view must outlive the frame (it does —
        // env swaps go through the retire queue).
        void setPaneEnvironment(VkImageView view, VkSampler sampler, float exposure);

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
        void createLitMeshPipelines();

        // Lit split-screen pane depth buffer: (re)create at the swapchain
        // extent and transition to DEPTH_ATTACHMENT_OPTIMAL. Must be called
        // OUTSIDE a rendering scope (records a barrier into cb). A stale
        // extent's image goes through retireFn_ — frames still in flight may
        // reference it.
        void ensurePaneDepth(VkCommandBuffer cb);

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

        // Free a cached geometry buffer that an in-flight frame may still be
        // reading: hands it to the renderer's frame-serial retire queue, or
        // drains the device first when no callback was wired. Zeroes `b`, and
        // is a no-op on an already-null buffer. NOT for teardown — the
        // destructor runs with the device idle and destroys inline.
        void retireBuffer(Buffer& b);

        VulkanContext&      ctx_;
        uint32_t            framesInFlight_;
        SampledImageCreator uploadFn_;
        RetireImageFn       retireFn_;
        RetireBufferFn      retireBufferFn_;

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

        // Lit split-screen pane (overlay_mesh_lit shaders): meshes in a
        // secondary scissored render() draw depth-tested and sun+ambient lit
        // over a cleared background, instead of the flat fills the HUD path
        // uses. Own layout — 128-byte PC (mvp + normal-matrix columns + color)
        // plus a per-FIF light UBO the flat pipelines don't carry.
        VkDescriptorSetLayout        paneLightSetLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool             paneLightDescPool_  = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> paneLightSets_;// per FIF, written once
        std::vector<Buffer>          paneLightUbos_;// per FIF, 48 B host-visible
        VkPipelineLayout litMeshPipelineLayout_      = VK_NULL_HANDLE;
        VkPipeline       litMeshPipeline_            = VK_NULL_HANDLE;
        VkPipeline       litMeshTransparentPipeline_ = VK_NULL_HANDLE;
        // The pane's own depth image (D32, swapchain extent). The rect in use
        // is cleared each pane record; a dedicated image because the main
        // frame's depth attachments all carry meaning across passes.
        Image2D paneDepth_{};
        bool    paneDepthInitialized_ = false;// first use transitions from UNDEFINED

        // Lit pane sky (overlay_pane_sky shaders): fullscreen equirect draw
        // under the meshes, sampling the renderer's envImage. Set state comes
        // from setPaneEnvironment each frame; per-FIF descriptor sets are
        // rewritten lazily when the env view changes (a slot is only touched
        // once its fence has passed, so no in-flight set is ever updated).
        void createPaneSkyPipeline();
        VkImageView   paneEnvView_    = VK_NULL_HANDLE;
        VkSampler     paneEnvSampler_ = VK_NULL_HANDLE;
        float         paneEnvExposure_ = 1.f;
        VkDescriptorSetLayout        paneSkySetLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool             paneSkyDescPool_  = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> paneSkySets_;       // per FIF
        std::vector<VkImageView>     paneSkyWrittenView_;// what each set holds
        VkPipelineLayout paneSkyPipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline       paneSkyPipeline_       = VK_NULL_HANDLE;

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
