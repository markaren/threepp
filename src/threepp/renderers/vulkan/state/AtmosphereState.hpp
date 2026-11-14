// AtmosphereState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the participating media: air fog and its march, underwater murk, the medium, clouds, height fog, the
// deferred volumetric fog knobs and the rotor downwash.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct AtmosphereState {
        bool     fogEnabledThisFrame_ = false;// scene.fog present (froxel-volumetrics gate)

        // Layout (and its doc comment) moved to VulkanGpuLayouts.hpp.
        using GpuFogUbo = vulkan::impl::GpuFogUbo;
        std::array<Buffer, kFramesInFlight> fogUbos{};
        float    fogAnisotropy_ = 0.0f;
        float    fogWaterSurfaceY_ = 1e30f;
        float    murkDensity_ = 0.0f;               // setUnderwaterMurk (0 = off); the clearest channel of murkSigma_
        float    murkSigma_[3] = {0.0f, 0.0f, 0.0f};// setUnderwaterMurk extinction per channel (all = murkDensity_ in the scalar form)
        float    murkColor_[3] = {1.0f, 1.0f, 1.0f};// setUnderwaterMurk tint
        float    murkAmbientFalloff_ = 0.0f;        // setUnderwaterAmbientFalloff (0 = off)
        bool     murkSpriteForeshortening_ = true;  // setUnderwaterSpriteForeshortening
        // ── Resolved unified fog medium for THIS frame (computed by updateFogUbo,
        // consumed by updateCloudUbo + the froxel gate) ──────────────────────────
        // Phase 2 "one knob": scene.fog is the primary control — when present it
        // supplies the medium density (FogExp2.density / linear-Fog span) + colour
        // and the froxel volumetrics run in heterogeneous mode with a near-uniform
        // default profile (baseY 0, huge falloff). setHeightFog is the advanced
        // profile control: it sets baseY/falloff/noise; its density is used only
        // when scene.fog is absent (back-compat for existing mist users). When
        // both are set, scene.fog's density wins (heightFog.density ignored).
        bool  mediumActiveThisFrame_ = false;// air medium present → run froxels hetero
        float mediumDensityThisFrame_ = 0.0f;
        float mediumBaseYThisFrame_   = 0.0f;
        float mediumFalloffThisFrame_ = 1.0e6f;
        float mediumNoiseThisFrame_   = 0.0f;
        // Air-fog single-scattering tint (= scene.fog colour when present, else
        // white — mirrors the shade pass's medAlbedo). The overlay-fog snapshot
        // fades particles toward this (× no env, the accepted particle-domain
        // approximation). Consumed by the overlay particle draw in the record.
        float mediumTintThisFrame_[3] = {1.0f, 1.0f, 1.0f};
        // Near-uniform default profile falloff (m) when scene.fog drives the
        // medium without an explicit setHeightFog profile. exp(-y/H) ≈ 1 over any
        // real scene extent, and heightFogOpticalDepth clamps y<baseY to full
        // density → a homogeneous medium (the degenerate case of the height-fog
        // integral). The sky path detects this magnitude to avoid a divergent
        // infinite-ray optical depth (see applySkyFog).
        static constexpr float kUniformFogFalloff = 1.0e6f;

        // Layout (and its doc comment) moved to VulkanGpuLayouts.hpp.
        using GpuCloudUbo = vulkan::impl::GpuCloudUbo;
        std::array<Buffer, kFramesInFlight> cloudUbos{};
        bool  cloudsEnabled_   = false;
        // Bumped by setClouds on enable / material reconfigure (NOT on identical
        // per-frame re-sets — the fjord demo calls setClouds every frame).
        // Wrapped &1023: the aux .a channel is fp16, int-exact only to 2048.
        int   cloudEpoch_      = 1;
        float cloudCoverage_   = 0.45f;
        float cloudDensity_    = 1.0f;
        float cloudBottomY_    = 600.0f;
        float cloudTopY_       = 1400.0f;
        float cloudEvolveSpeed_= 1.0f;
        float cloudWind_[3]    = {8.0f, 0.0f, 2.0f};
        // Near-field heterogeneous height fog (VulkanRenderer::setHeightFog).
        bool  heightFogEnabled_    = false;
        float heightFogDensity_    = 0.02f;
        float heightFogBaseY_      = 0.0f;
        float heightFogFalloff_    = 80.0f;
        float heightFogNoiseAmount_= 0.6f;
        float heightFogTopNoise_      = 0.0f;
        float heightFogTopNoiseScale_ = 150.0f;
        float heightFogTopNoiseSpeed_ = 4.0f;

        uint32_t fogMarchSteps_   = 16u;
        float    fogMarchRtRange_ = 0.f;
        // setFogMarchHalfRes: fog_march.comp at half res. THREEPP_VK_FOG_HALF_RES=1
        // starts it on (an A/B on an app that does not set the knob).
        bool     fogMarchHalfRes_ = vulkan::env::flag("THREEPP_VK_FOG_HALF_RES");

        // Deferred volumetric spot-light beams (ray-marched single scattering in
        // deferred_shade.comp). σ = 0 disables (the march is skipped entirely).
        float deferredVolDensity_ = 0.f;
        float deferredVolAniso_   = 0.55f;
        // The directional sun shafts / aerial glow are ALWAYS on when the fog medium
        // is present (froxels own the near field, volumetricDirScatter the > 512 m
        // tail). There is no public toggle; this stays true and only feeds the
        // shade/froxel volFog flag.
        bool  deferredVolFog_     = true;
        // setUnlitFogScatter: the unlit branch of the shade takes the volumetric
        // in-scatter too (shade flags bit 28). Off = extinction and ambient only.
        bool  unlitFogScatter_    = false;
        // ── Rotor downwash (setDownwash; scenes/Downwash.hpp) ──────────────
        // uploadDownwash packs the enabled sources into this slot's block once
        // per frame, after the fence wait; downwashAddr_ is then its device
        // address (0 = no downwash this frame) and downwashCount_ the packed
        // count. The G-buffer (RasterCameraUbo.downwash), the grass wind push
        // block and the particle emitters' aux records read these.
        std::shared_ptr<Downwash> downwash_;
        std::array<Buffer, kFramesInFlight> downwashBufs_{};
        VkDeviceAddress downwashAddr_  = 0;
        uint32_t        downwashCount_ = 0;
        // Last frame's centres by source slot (the particles' motion vectors);
        // reset whenever the list's version changes.
        std::array<Vector3, 8> downwashPrevCenters_{};
        uint64_t downwashPrevVersion_ = ~0ull;
        const Downwash* downwashPrevOwner_ = nullptr;

        bool  fogShadowOpaqueCutouts_ = true; // setFogShadowOpaqueCutouts (shade flags bit 29); off = exact
    };

}// namespace threepp::vulkan::impl
