// OverlayState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the post-TAA overlay: the overlay pass, its pipelines (lines, points, meshes, depth prepass and import,
// MSAA inject), overlay fog, colour grade, and the line and morphed-point geometry caches.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_OVERLAYSTATE_HPP
#define THREEPP_VULKAN_STATE_OVERLAYSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct OverlayState {
        bool     overlayColorGrade_ = false;          // setOverlayColorGrade
        bool     overlayDepthFromGbuffer_ = false;    // setOverlayDepthFromGbuffer

        // Hybrid raster overlay (wireframe + Line/LineSegments). Runs after
        // TAA resolve, draws onto the swapchain with the G-buffer's depth
        // attachment as a read-only depth source so overlays are correctly
        // occluded by the traced/rasterized scene geometry. No descriptor
        // sets — the only input is a push constant (mvp + color).
        VkPipelineLayout overlayPipelineLayout      = VK_NULL_HANDLE;
        VkPipeline       overlayWireframePipeline   = VK_NULL_HANDLE;
        // Solid-fill counterpart for MeshBasicMaterial-style overlays — flat
        // color, depth-tested, but rendered as filled triangles instead of
        // wireframe lines. Selected per-draw based on the material's
        // `wireframe` flag (true → wireframe pipeline, false → basic).
        VkPipeline       overlayBasicPipeline       = VK_NULL_HANDLE;
        // Alpha-blended counterpart to overlayBasicPipeline. Same state
        // except colorBlendAttachmentState's blendEnable=TRUE with
        // SRC_ALPHA / ONE_MINUS_SRC_ALPHA. Selected when the mesh's
        // material has `transparent == true`.
        VkPipeline       overlayBasicTransparentPipeline = VK_NULL_HANDLE;
        // Line-topology pipelines for Line / LineSegments objects. Same
        // overlay shaders, only input-assembly topology differs:
        //   LineSegments → LINE_LIST  (vertices in pairs)
        //   Line         → LINE_STRIP (connected polyline)
        // Lines are never part of the traced/rasterized scene; they're
        // collected separately from Mesh entries and walked in the overlay
        // record loop.
        VkPipeline       overlayLineListPipeline    = VK_NULL_HANDLE;
        VkPipeline       overlayLineStripPipeline   = VK_NULL_HANDLE;
        // Per-vertex color counterparts. Use overlay_color.vert/.frag
        // (location 1 = inColor) and require a 2-binding vertex input.
        // Picked when the Line's geometry has a "color" attribute AND the
        // material has vertexColors == true (matches three.js semantics).
        VkPipeline       overlayLineListColoredPipeline  = VK_NULL_HANDLE;
        VkPipeline       overlayLineStripColoredPipeline = VK_NULL_HANDLE;
        // Blended counterparts to the four line pipelines above, which all
        // share the wireframe pipeline's blend-OFF state — that silently
        // dropped Material::transparent/opacity/blending for Line overlays
        // (an additive streamline drew as an opaque near-black stroke).
        // Selection mirrors GLState::setBlending's gate: blend only when
        // NOT (Normal && !transparent). Indexed [colored][strip][mode] with
        // mode 0 = alpha (SRC_ALPHA, 1-SRC_ALPHA), 1 = additive
        // (SRC_ALPHA, ONE) — the same factors the GL backend programs.
        VkPipeline       overlayLineBlendPipelines[2][2][2] = {};
        // Vertex-coloured mesh fill (overlay_color shaders, TRIANGLE_LIST,
        // dynamic cull like overlayBasicPipeline). Lets unlit transparent
        // vertex-coloured basics take the kSnapUiBlend overlay route instead
        // of landing opaque in the G-buffer (see snapMeshFlags). Indexed
        // [0] opaque, [1] alpha-blended, [2] additive.
        VkPipeline       overlayMeshColoredPipelines[3] = {};
        // Additive counterpart to overlayBasicTransparentPipeline, so a flat
        // Blending::Additive basic mesh matches the GL backend too.
        VkPipeline       overlayBasicAdditivePipeline = VK_NULL_HANDLE;
        // Vertex-coloured wireframe (overlay_color shaders, POLYGON_MODE_LINE,
        // blend-off like the flat wireframe). HemisphereLightHelper-style
        // helpers paint a "color" attribute on a wireframe basic material;
        // the flat wireframe pipeline drew them in the material colour
        // (default white) instead.
        VkPipeline       overlayWireframeColoredPipeline = VK_NULL_HANDLE;
        // Points pipeline — POINT_LIST topology. Always vertex-coloured;
        // a Points object without a "color" attribute renders as plain
        // material colour (vertex colour defaults to white in that case
        // because we still bind the geometry's color buffer if present,
        // and skip the draw if it's missing in the dispatch path below).
        // Writes gl_PointSize from PointsMaterial::size, encoded in the
        // push constant's color.w slot for this pipeline only.
        VkPipeline       overlayPointListPipeline        = VK_NULL_HANDLE;
        // Depth prepass that fills rasterGbufs[f].unjitDepth using the
        // unjittered VP. Reuses the raster pipeline's descriptor set + push
        // constants (same camera UBO, same model matrix push). Runs after
        // recordRasterGbufPass and only renders non-overlay geometry.
        VkPipeline       overlayDepthPrepassPipeline = VK_NULL_HANDLE;
        // The prepass's cheap form (overlay_depth_import.frag): when the only
        // overlay content is ParticleField billboards, the G-buffer's depth is
        // resampled into the overlay depth attachment by one fullscreen
        // triangle instead of rasterizing the scene a second time at display
        // resolution — see overlayDepthImportsGbuffer(). One descriptor set
        // per frame in flight naming that slot's G-buffer depth, rewritten
        // only when the view changes (imageViewSerial), from the post-fence
        // window of its own slot (prepareOverlayDepthImport).
        VkPipeline            overlayDepthImportPipeline_  = VK_NULL_HANDLE;
        VkPipelineLayout      overlayDepthImportLayout_    = VK_NULL_HANDLE;
        VkDescriptorSetLayout overlayDepthImportSetLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      overlayDepthImportPool_      = VK_NULL_HANDLE;
        VkSampler             overlayDepthImportSampler_   = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> overlayDepthImportSets_{};
        std::array<uint64_t, kFramesInFlight>        overlayDepthImportSerials_{};

        // ── Hardware-MSAA overlay ───────────────────────────────────────────
        // The overlay renders post-TAA, so its vector edges (SVG fills, lines,
        // wireframes, points) get no AA from TAA (deliberately after it, to
        // avoid ghosting) and none from the swapchain (Vulkan has no
        // multisampled swapchains). So when Canvas antialiasing > 1 the whole
        // overlay pass rasterizes into a multisampled color+depth pair and
        // vkCmdEndRendering RESOLVES it onto the swapchain — the exact
        // mechanism GL gets for free from a 4x default framebuffer.
        //
        // Because the swapchain already holds the composited scene and the
        // MS target starts undefined, the first draw in the pass is a
        // fullscreen "scene inject" that seeds every sample from a 1-sample
        // copy of the swapchain (a pass may not sample its own target).
        // That makes every blend state in the pass composite against the true
        // background and makes untouched pixels resolve back unchanged.
        //
        // overlaySamples_ == 1 keeps the original direct-to-swapchain path
        // (no inject, no scratch copy, no resolve).
        uint32_t              overlaySamples_    = 0;// 0 = not yet resolved
        VkSampleCountFlagBits overlaySampleBits_ = VK_SAMPLE_COUNT_1_BIT;
        Image2D               overlayMsColor_{};  // swapchain format, N samples
        Image2D               overlayMsDepth_{};  // D32_SFLOAT, N samples
        Image2D               overlayAaScratch_{};// 1-sample swapchain copy (inject source)
        VkDescriptorSetLayout overlayInjectSetLayout_      = VK_NULL_HANDLE;
        VkDescriptorPool      overlayInjectPool_           = VK_NULL_HANDLE;
        VkDescriptorSet       overlayInjectSet_            = VK_NULL_HANDLE;
        VkPipelineLayout      overlayInjectPipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            overlayInjectPipeline_       = VK_NULL_HANDLE;

        // ── Overlay-pass fog (Phase 2b) ─────────────────────────────────────
        // The post-TAA overlay draws world-space ParticleSystem billboards
        // (chimney smoke etc.) that never saw the unified air-fog / murk medium.
        // particle.frag now binds this per-frame snapshot at SET 1 (the texture
        // stays at set 0) and applies the closed-form fog per fragment. The
        // world-space Sprite pipeline shares particlePipelineLayout_ but its
        // shader does not reference set 1, so it simply ignores the (still-bound)
        // fog set. std140 — mirrors particle.frag's OverlayFog block.
        // Layout moved to VulkanGpuLayouts.hpp.
        using GpuOverlayFogUbo = vulkan::impl::GpuOverlayFogUbo;
        VkDescriptorSetLayout overlayFogDescSetLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      overlayFogDescPool_      = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> overlayFogDescSets_{};
        std::array<Buffer, kFramesInFlight>          overlayFogUbos_{};

        // Definition moved to VulkanSceneTypes.hpp.
        using LineEntry = vulkan::impl::LineEntry;
        std::vector<LineEntry> lastVisibleLines_;

        // Ortho/HUD overlay pass. Owns the sprite and
        // ortho-line pipelines, atlas/geometry caches, and per-frame descriptor
        // pools. Constructed after ctx is available (same constraint as gpuTimings_).
        std::unique_ptr<vulkan::OverlayPass> overlayPass_;

        // Hybrid raster overlay: layer index for opt-in overlay objects
        // (alongside auto-detected wireframe materials + Line/LineSegments).
        // -1 disables layer-based selection.
        int overlayLayer_ = -1;

        // Line geometry cache for the 3D hybrid overlay (recordCommandBuffer's
        // line-draw section). Keyed on raw BufferGeometry*; geomId in the
        // entry guards against recycled-pointer aliasing. (OverlayPass owns a separate
        // cache of the same shape for the ortho path — these are NOT shared.)
        //
        // The counter is advanced once per submitted frame in endFrame, which
        // also runs sweepLineGeomCache — eviction matters for apps that
        // rebuild Line/Points geometry each frame (trajectory, scan and
        // detection-box overlays), which would otherwise leak a buffer set
        // per geometry.
        std::unordered_map<const BufferGeometry*, vulkan::LineCacheEntry> lineGeomCache_;
        uint64_t overlayFrameCounter_ = 0;// lastTouch reference for lineGeomCache_ entries

        // A Points whose material sets morphTargets: GL blends the geometry's
        // morph targets in the vertex shader (morphtargets_sphere rides its
        // points on the morphing sphere that way); the overlay point pipeline
        // reads plain positions, so the blend happens here. Per OBJECT, not per
        // geometry: the geometry, and so its LineRec, is typically shared with
        // the morphing Mesh, and the two need not carry the same influences.
        // One buffer per frame in flight, written while recording slot
        // currentFrame, i.e. past that slot's fence wait, so the frame that
        // last read it has finished. Swept with lineGeomCache_.
        struct MorphedPointsRec {
            std::array<Buffer, kFramesInFlight>             vertex{};
            std::array<std::vector<float>, kFramesInFlight> influences{};// what each slot holds
            std::array<uint32_t, kFramesInFlight>           posVersion{};
            unsigned int       geomId    = 0;
            uint64_t           lastTouch = 0;
            std::vector<float> blended;
        };
        std::unordered_map<const Points*, MorphedPointsRec> morphedPointsCache_;

        std::vector<uint32_t> overlayEntryScratch_;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_OVERLAYSTATE_HPP
