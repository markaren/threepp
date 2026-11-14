// ParticleState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns particles: ParticleSystem billboards (pipelines, textures, lit-particle buffers), world-space sprites,
// and the ParticleField pass with its density volume.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct ParticleState {
        // ── ParticleSystem billboard pass ───────────────────────────────────
        // The Vulkan backend has no generic ShaderMaterial path, so the particle
        // Mesh (a custom billboard ShaderMaterial whose quad is expanded in the
        // vertex shader) is drawn here, in the post-TAA overlay block: depth-
        // tested against unjitDepth, composited onto the swapchain. Two blend
        // variants chosen per material at draw time. See createParticlePipeline /
        // the particle draw loop in the overlay block, and particle.vert/.frag.
        VkDescriptorSetLayout particleDescSetLayout_  = VK_NULL_HANDLE;
        VkPipelineLayout      particlePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline particlePipelineNormal_   = VK_NULL_HANDLE;// alpha blend, depth-test on
        VkPipeline particlePipelineAdditive_ = VK_NULL_HANDLE;// additive blend, depth-test off

        // Per-frame combined-image-sampler pool, reset at the top of the draw
        // loop (mirrors OverlayPass::spriteDescPools_).
        std::array<VkDescriptorPool, kFramesInFlight> particleDescPools_{};
        // 1×1 white default bound when a particle system has no texture.
        Image2D particleWhiteTex_{};
        static constexpr uint32_t kMaxParticleTexPerFrame = 64;

        // Definition moved to VulkanSceneTypes.hpp.
        using ParticleGeomRec = vulkan::impl::ParticleGeomRec;
        std::unordered_map<const BufferGeometry*, ParticleGeomRec> particleGeomCache_;

        // ── Lit particles (particle_light.comp) ────────────────────────────
        // Per-frame world-space particle centers (host-written, one vec4 per
        // particle across ALL visible systems, in overlay draw order) + the
        // per-particle {light,T}/{fogAdd} results the compute pass writes and
        // particle.vert reads (set 1 binding 1 of the billboard pipeline).
        // Eagerly allocated in createParticlePipeline (fixed footprint, ~1.5 MB
        // total). Systems past kMaxLitParticles fall back to the unlit path via
        // the kUnlitBase firstInstance sentinel (KEEP IN SYNC with particle.vert).
        static constexpr uint32_t kMaxLitParticles = 16384;
        static constexpr uint32_t kUnlitBase       = 0x40000000u;
        std::array<Buffer, kFramesInFlight> particleCenterBufs_{};// host-visible SSBO, kMaxLitParticles × vec4
        std::array<Buffer, kFramesInFlight> particleLightBufs_{}; // device-local SSBO, kMaxLitParticles × 2 × vec4
        VkDescriptorPool particleIoDescPool_ = VK_NULL_HANDLE;    // particle_light.comp set 1
        std::array<VkDescriptorSet, kFramesInFlight> particleIoDescSets_{};
        // Filled by prepareParticleLighting() each frame; consumed by the
        // subclass scene-dispatch hook (compute) + the overlay particle loop
        // (per-draw firstInstance base).
        uint32_t particleLightCount_ = 0;// live particles prepped this frame (0 = pass off)
        std::unordered_map<const void*, uint32_t> particleLitBase_;// mesh → base particle index

        // CPU-side staging for the centers upload (avoids per-frame realloc).
        std::vector<float> particleCenterScratch_;

        // Definition moved to VulkanSceneTypes.hpp.
        using ParticleTexRec = vulkan::impl::ParticleTexRec;
        std::unordered_map<const Texture*, ParticleTexRec> particleTexCache_;

        // ── World-space Sprite billboards (screenSpace == false) ────────────
        // 3D-positioned camera-facing sprites (e.g. the TPS shooter's impact
        // "particles"). The Vulkan renderer only composites screen-space (HUD)
        // sprites via OverlayPass; world-space ones are drawn here in the
        // depth-tested overlay block. Reuses particlePipelineLayout_ +
        // particleDescSetLayout_ + particleDescPools_ + particleTexCache_ (the
        // push-constant size, set-0 sampler layout, and texture cache all match);
        // only the pipeline (perspective billboard vertex shader + interleaved
        // pos/uv vertex input) and the shared static quad are new.
        VkPipeline spriteWorldPipeline_ = VK_NULL_HANDLE;
        // Shared canonical sprite quad (4 interleaved pos.xyz+uv.xy verts + 6
        // indices) — identical for every Sprite, so one static copy serves all.
        Buffer spriteQuadVtx_{};
        Buffer spriteQuadIdx_{};
        // Definition moved to VulkanSceneTypes.hpp.
        using WorldSpriteEntry = vulkan::impl::WorldSpriteEntry;
        std::vector<WorldSpriteEntry> lastVisibleSprites_;

        // ── ParticleField (phase 0: buffers + descriptor, no consumer) ──────
        // The device-side position rings and the per-frame FieldDesc SSBO for
        // every threepp::ParticleField in the scene. ALWAYS ON and free when
        // there are no fields: the whole pass is O(fields), and a scene without
        // one does a single empty-vector test per frame.
        std::unique_ptr<vulkan::ParticleFieldPass> particleFieldPass_;
        // (field, its index in lastVisibleEntries_). Written only by a full
        // scene expansion — a field appearing or disappearing is structural, so
        // the snapshot fast path never runs with this stale.
        std::vector<std::pair<ParticleField*, uint32_t>> particleFields_;
        // Reused scratch for the per-frame Rec list handed to the pass.
        std::vector<vulkan::ParticleFieldPass::Rec> particleFieldRecs_;

        // ── ParticleField density volumes (phase 2, plan §3.3) ──────────────
        // Bindings 67/68 of EVERY view's deferred set. Both handles are owned
        // here, not by ParticleFieldPass, because they must be valid from the
        // renderer's first descriptor write — which happens before the pass is
        // lazily constructed, and keeps happening on scenes that never get one.
        //
        //   particleDensityUbos_  — per-FIF ParticleDensityUbo (std140, 160 B),
        //                           rewritten every frame; handle never changes.
        //   particleDensityDummy_ — 1x1x1 R32_UINT in GENERAL, bound to every
        //                           array slot no live volume occupies. Same
        //                           "always bound, harmlessly unused" idiom as
        //                           the ocean/MSAA dummies.
        Buffer  particleDensityUbos_[kFramesInFlight]{};
        Image2D particleDensityDummy_{};
        // Its r16f twin for binding 69 (the shade's linear-sampled mirrors).
        Image2D particleDensityLinDummy_{};
        // Any field contributed density this frame. Forces heteroActive, opens
        // the froxel gate with no clustered lights, and sets the shade's flags
        // bit 11 — the three gates plan §3.3 calls "real, small, easy to miss".
        bool     particleDensityActiveThisFrame_ = false;
        // ParticleFieldPass::densityGeneration() each FIF's deferred set was
        // last written with. Not equal ⇒ that set names a stale (possibly
        // retired) volume view and must be rewritten before it is recorded.
        uint64_t particleDensityDescGen_[kFramesInFlight]{};
    };

}// namespace threepp::vulkan::impl
