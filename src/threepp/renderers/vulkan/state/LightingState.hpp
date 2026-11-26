// LightingState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns what the deferred shade lights with: the lights UBO and clusters, the emissive triangles and their
// cache, the sun's size and softness, and the shade toggles (AO, ReSTIR DI, denoise, firefly clamp).
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_LIGHTINGSTATE_HPP
#define THREEPP_VULKAN_STATE_LIGHTINGSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct LightingState {
        // Per-FIF deferred-descriptor refresh flag. Set (all slots) when a
        // material texture is swapped in place (refreshDirtyMaterialTextures) or
        // an env texture is swapped: the new image view must be rebound in the
        // deferred/probe descriptor sets, but those sets can't be updated while
        // in flight (no descriptorBindingUpdateAfterBind on this device). Same
        // idiom as geomDescsDirty_/matDescsDirty_: the frame-begin path rewrites
        // ONLY the current slot's set once its fence has signaled, so no device
        // stall. In-flight frames sample the retired old view for ≤1 more frame
        // (it stays alive in the retire queue) — correct behavior.
        std::array<bool, kFramesInFlight> deferredDescDirty_{};

        // Scene lights mirrored to a per-frame UBO. Scalar block layout means
        // the C++ structs map directly (no std140 vec3→vec4 padding).
        // Bounds moved to VulkanImplCommon.hpp (GpuLightsUbo sizes its arrays
        // with them); forwarded so every use site stays unqualified.
        static constexpr auto kMaxDirLights   = vulkan::impl::kMaxDirLights;
        static constexpr auto kMaxPointLights = vulkan::impl::kMaxPointLights;
        static constexpr auto kMaxSpotLights  = vulkan::impl::kMaxSpotLights;
        static constexpr auto kMaxRectLights  = vulkan::impl::kMaxRectLights;

        // GPU-layout mirrors moved to VulkanGpuLayouts.hpp (their layout
        // comments and sizeof static_asserts travel with them).
        using GpuDirLight   = vulkan::impl::GpuDirLight;
        using GpuPointLight = vulkan::impl::GpuPointLight;
        using GpuSpotLight  = vulkan::impl::GpuSpotLight;
        using GpuRectLight  = vulkan::impl::GpuRectLight;
        using GpuLightsUbo  = vulkan::impl::GpuLightsUbo;
        std::array<Buffer, kFramesInFlight> lightsUbos{};

        // ── Clustered lights (deferred) ─────────────────────────────────────
        // All scene point/spot lights, power-sorted, in one unified record;
        // the UBO's 8-per-type arrays above keep the strongest 8 (same sort)
        // for the paths screen-space clusters can't serve (secondary ray
        // hits — reflection/GI, volumetric beams, probes). cluster_build.comp
        // culls the list into per-cell index rows of a 16×8×24 screen-tile ×
        // exponential-Z grid; deferred_shade's analytic split loops only its
        // own cell.
        // KEEP IN SYNC with the ClusterLight structs in cluster_build.comp +
        // deferred_shade.comp (scalar layout, 64 bytes).
        static constexpr uint32_t kMaxClusterLights   = 256;
        static constexpr uint32_t kClusterCells       = 16 * 8 * 24;
        static constexpr uint32_t kClusterMaxPerCell  = 24;
        // Layout moved to VulkanGpuLayouts.hpp.
        using GpuClusterLight = vulkan::impl::GpuClusterLight;
        using GpuShadowMapsUbo = vulkan::impl::GpuShadowMapsUbo;
        std::array<Buffer, kFramesInFlight> clusterLightsBuffers{};// host-visible mapped (CPU fills per frame)
        std::array<Buffer, kFramesInFlight> clusterGridBuffers{};  // device-local (cluster_build writes)
        uint32_t clusterLightCountThisFrame_ = 0;

        // Frame counter driving Halton jitter + blue-noise offset for the
        // deferred shade's stochastic GI / soft-shadow sampling. Genuinely
        // shared: it is the sequence CLOCK, not per-view history. Every view
        // in a frame samples the same Halton/blue-noise phase.
        uint32_t sampleIndex = 0;

        // (The per-slot all-identity flag was replaced by the version pair
        //  motionScratchVersion_ / motionUploadedVersion_ — see the members
        //  next to prevWorldByEntry_.)
        // Per-frame emissive-triangle CDF buffer. Each entry is 4 vec4 (64 B):
        //   v0.xyz/area, v1.xyz/cumPower, v2.xyz/power, emission.rgb/_pad.
        // Built fresh each frame from the visible scene; size = numEmissiveTris.
        // Capacity is grown 2× and a descriptor rewrite triggers when the
        // VkBuffer handle changes. Uploaded by buildAndUploadEmissiveTris.
        std::array<Buffer, kFramesInFlight> emissiveTriBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> emissiveTriBufferCapacity{};
        uint32_t emissiveTriCountThisFrame_ = 0;
        float    emissiveTotalPowerThisFrame_ = 0.0f;
        // Per-NEE firefly clamp; pushed to shaders as float bits in slot [11].
        // 1e30f sentinel disables the clamp (set via setFireflyClamp(0)).
        float    fireflyClamp_ = 30.f;
        // Directional-light angular RADIUS (degrees) for the deferred renderer's
        // soft sun shadows — pushed to deferred_shade as tan(radians(deg)) in PC
        // slot [16]. 0 = exact hard 1-ray shadow (old behaviour). Default 0.0°
        // (the real sun subtends ~0.27°): thin occluders cast a stable narrow
        // penumbra instead of a per-frame lit/shadow coin flip under TAA jitter.
        float    sunAngularRadiusDeg_ = 0.0f;

        // The deferred shade's ray budget (setSunSoftDistance, setFogMarchSteps,
        // setFogMarchRtRange: the lights UBO's tail) and setCountRays: per frame
        // in flight a device-local buffer of kRayCatCount uints the shade bumps
        // through its device address (zeroed at the head of the frame's command
        // buffer, recordRayCountClear), copied at its tail into a host-mapped
        // one (recordRayCountCopy) that readRayCounts reads after the fence.
        // Device-local because the atomics, aimed at host memory, cost the
        // shade 17 ms a frame.
        float    sunSoftDistance_ = 0.f;

        // Cached CDF blob (16 floats per tri) reused across frames when no
        // emissive mesh moved + entries-list size unchanged. The CPU walk in
        // buildAndUploadEmissiveTris is the dominant per-frame cost on
        // texture-heavy scenes like Bistro; reusing the cache makes
        // camera-only motion a memcpy instead of a re-trace of every tri.
        std::vector<float> cachedEmissiveData_;
        uint32_t cachedEmissiveTriCount_ = 0;
        float    cachedEmissiveTotalPower_ = 0.0f;
        size_t   cachedEmissiveEntryCount_ = static_cast<size_t>(-1);
        // Entries the last walk found emissive, as meshMovedBits_ words: only a
        // moved emitter invalidates the cache (buildAndUploadEmissiveTris).
        std::vector<uint32_t> cachedEmissiveEntryBits_;
        uint32_t cachedEmissiveVersion_ = 0;
        // The range gate's exposure term the cached header holds
        // (emissiveGateK(); the emitter NEE skips a point no light reaches).
        float cachedEmissiveGateK_ = 0.0f;

        // Objects whose emissive glows but is not sampled as a light
        // (setEmissiveCastsLight), keyed by Object3D::id like classIds_. The
        // emitter walk skips them; GeometryDesc.flags bit 6 tells the ray hits
        // to keep their emission. glowOnlyVersion_ is bumped on every change so
        // the cached emitter list above is rebuilt.
        std::unordered_set<unsigned int> glowOnlyObjects_;
        uint32_t glowOnlyVersion_       = 0;
        uint32_t cachedGlowOnlyVersion_ = 0;

        std::array<uint32_t, kFramesInFlight> emissiveBufferVersion_{};

        // THREEPP_DENOISE=0 disables the deferred SVGF denoiser from the
        // environment — an A/B discriminator for "is this artifact shading or
        // temporal/denoise?" without plumbing a flag through every example.
        bool denoiseEnabled_ = vulkan::env::flag("THREEPP_DENOISE", true);

        // Deferred GI path: ON activates stochastic 1-bounce GI (colour bleed) +
        // temporal accumulation + à-trous. OFF falls back to the deterministic
        // AO+far≈sky approximation. setDenoise(false) → clean fallback.
        // Ray-traced env ambient occlusion / GI. ON: gives the "dirty realistic"
        // ray-traced grounding (contact darkening + 1-bounce GI). Uses the
        // deterministic Fibonacci 64-sample gather (clean + settles), no
        // per-frame flicker.
        bool  deferredAO_ = true;

        bool  indirectOpaqueCutouts_  = false; // setIndirectOpaqueCutouts (shade flags bit 31 clear, probe flags bit 3 clear when on); off (the default) = the GI, probe and AO rays alpha-test cutouts
        // Procedural direction-space star field on deferred sky pixels (0 = off).
        float deferredStarIntensity_ = 0.f;

        // ReSTIR DI master toggle. When false, chit's primary RIS branch is
        // bypassed and the legacy per-light NEE classic loops run instead
        // (same pattern as bounces). Forwarded to the deferred shade via
        // pc.motionFlags bit 4 each frame.
        bool restirDIEnabled_ = true;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_LIGHTINGSTATE_HPP
