// FieldBillboardState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns ParticleField billboards: the billboard pipeline layout and the sun snapshot, the glow pass, the HDR
// layer and its pipelines, and the soft-depth pass.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct FieldBillboardState {
        // ── ParticleField billboards (plans/particle-atmosphere.md F-D) ─────
        // A sibling of the two above rather than a variant of them, drawn in
        // the same slot: the post-upscaler overlay render pass, right after
        // the legacy billboard loop. That position is deliberate:
        //
        //   • After the upscalers, the quads are never fed to TAA / DLSS /
        //     FSR — a 3-px spark crossing 20 px in a frame is exactly the
        //     content those filters mis-handle, and an additive glow has no
        //     depth or motion vector to give them anyway.
        //   • It is where transparents already composite, so field billboards
        //     land in the same order relative to lines, wireframe and legacy
        //     particles.
        //   • The pass already binds the unjittered depth read-only, so the
        //     quads get correct occlusion against scene geometry for free.
        //
        // Zero vertex bindings and zero uniform buffers: the quad is
        // vertex-less, the positions come by buffer_reference and the per-field
        // appearance record comes by device address. The one descriptor is the
        // optional sprite texture, which reuses the legacy path's set layout,
        // its per-frame pool and its 1x1 white default.
        // (plans/particle-volumetric-sprites briefly gave this pipeline a SET 1
        // — the field's r16f density mirror, marched by the vertex stage. R8
        // moved the marches into particlefield_transmit.comp, which is where
        // the image is bound now, so the exception closed again and the draw
        // side is back to one set.)
        // The sprites themselves are drawn by BillboardHdrLayer's two
        // pipelines (additive and alpha-over), which share this layout, and by
        // the glow leg's pipeline below.
        VkPipelineLayout fieldBillboardPipelineLayout_ = VK_NULL_HANDLE;
        // The scene's one sun, snapshotted by updateLightsUbo for the
        // billboard slice — the brightest DirectionalLight in the lights UBO
        // (already the env-sun/scene-sun decision EnvSunPolicy made), plus the
        // summed ambient. Kept as three host vectors so the pass needs no
        // lights descriptor set. Defaults: no sun, no ambient, so a `lit`
        // field in a scene with neither draws black rather than undefined.
        float            bbSunDirWorld_[3]  = {0.f, 1.f, 0.f};
        float            bbSunRadiance_[3]  = {0.f, 0.f, 0.f};
        float            bbAmbient_[3]      = {0.f, 0.f, 0.f};
        // Billboard-only bloom chain. Created lazily on the first frame a
        // field asks for a glow — a scene with no sparks allocates no target,
        // compiles no pipeline and records no pass.
        std::unique_ptr<vulkan::BillboardGlowPass> billboardGlow_;
        bool billboardGlowReadyThisFrame_ = false;
        // The primary's field billboards, summed in linear HDR and tone-mapped
        // once (BillboardHdrLayer.hpp): the only way field billboards are
        // drawn. Created on the first frame a field draws billboards; a scene
        // without one has nothing of it. (A secondary view's own layer lives on
        // its ViewContext.)
        std::unique_ptr<vulkan::BillboardHdrLayer> billboardHdrLayerPass_;
        // The pipelines every view's layer draws and composites with, built
        // with the other field billboard pipelines (createFieldBillboardPipeline).
        std::unique_ptr<vulkan::BillboardHdrLayerPipelines> bbLayerPipes_;
        // The prepare window's answer (a field with a billboard
        // representation, targets allocated), then the graph's (the overlay
        // pass and its depth prepass run).
        bool billboardHdrLayerReady_     = false;
        bool billboardHdrLayerThisFrame_ = false;
        // This frame: the front passes were added (addFieldBillboardFrontPasses),
        // the soft-depth buffer they built, and whether the post composite
        // (upscaler path) blends the layer rather than the overlay pass.
        bool     bbFrontAdded_     = false;
        bool     bbLayerInPost_    = false;
        VkBuffer bbFrontSoftDepth_ = VK_NULL_HANDLE;
        // BillboardRepr::softDistance: the linear scene depth the soft fade
        // reads, one buffer per (view, frame in flight). Created on the first
        // frame a visible field asks for a soft edge; until then nothing of it
        // exists and no pass is recorded.
        std::unique_ptr<vulkan::BillboardSoftDepth> billboardSoftDepth_;

        // Billboard glow pipelines.
        //   • ...GlowPipeline_ draws the same quads into BillboardGlowPass's
        //     half-extent linear-HDR target: rgba16f, one sample, no depth
        //     attachment (see that class's header for why occlusion is given
        //     up at this resolution).
        //   • ...CompositePipeline_ is the fullscreen additive draw that folds
        //     the finished pyramid into the swapchain, inside the overlay
        //     render-pass instance — the composite point does not move, so the
        //     glow stays outside TAA/upscaling like the billboards themselves.
        VkPipeline       fieldBillboardGlowPipeline_ = VK_NULL_HANDLE;
        VkPipelineLayout fieldGlowCompositeLayout_   = VK_NULL_HANDLE;
        VkPipeline       fieldGlowCompositePipeline_ = VK_NULL_HANDLE;
    };

}// namespace threepp::vulkan::impl
