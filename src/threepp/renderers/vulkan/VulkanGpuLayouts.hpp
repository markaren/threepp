// VulkanGpuLayouts — the host-side mirrors of the GLSL uniform/storage
// blocks the Vulkan renderer uploads. Split out of VulkanCoreImpl.hpp; each
// struct is aliased back into VulkanRenderer::Impl at its original spot so
// every reference site (Impl methods and the VulkanCore*.cpp TUs) is
// unchanged.
//
// These are LAYOUT CONTRACTS with the shaders: the comments carry the layout
// rules and the sizeof static_asserts are the guard. Never reorder, repad or
// resize a member without editing the matching GLSL block.

#ifndef THREEPP_VULKAN_GPU_LAYOUTS_HPP
#define THREEPP_VULKAN_GPU_LAYOUTS_HPP

#include "VulkanImplCommon.hpp"

#include <cstdint>

namespace threepp::vulkan::impl {

    struct GpuDirLight {
        float direction[3];
        float color[3];
    };
    struct GpuPointLight {
        float position[3]; float range;
        float color[3];    float decay;
        float radius;      // physical source radius (world units) → RT soft shadows; 0 = hard
    };
    struct GpuSpotLight {
        float position[3];   float range;
        float color[3];      float decay;
        float direction[3];  // toward target (emission direction)
        float cosAngleOuter; // cos(angle)
        float cosAngleInner; // cos(angle * (1-penumbra))
        float radius;        // physical source radius (world units) → RT soft shadows; 0 = hard
    };
    struct GpuRectLight {
        float position[3];
        float halfU[3];  // world right  * width/2
        float halfV[3];  // world up     * height/2
        float normal[3]; // emission direction into scene
        float color[3];
    };
    struct GpuLightsUbo {
        float       ambient[3];
        uint32_t    dirCount;
        GpuDirLight dirLights[kMaxDirLights];
        uint32_t    pointCount;
        uint32_t    spotCount;
        uint32_t    rectCount;
        GpuPointLight pointLights[kMaxPointLights];
        GpuSpotLight  spotLights[kMaxSpotLights];
        GpuRectLight  rectLights[kMaxRectLights];
        // HemisphereLight, split in two so most shaders need no layout change:
        // each hemi's angular MEAN 0.5*(sky+ground) is folded into ambient
        // above (fog/cloud/water/particle/probe shaders declare only a PREFIX
        // of this block and keep seeing the energy), while these rows carry
        // the zero-mean remainder 0.5*(sky-ground)⊗up. The deferred surface
        // shade reconstructs the exact GL term mix(ground, sky,
        // 0.5*dot(N,up)+0.5) as ambient + rows·N; per-channel rows let
        // several hemis with different up axes sum exactly. Appended LAST so
        // every earlier offset is unchanged.
        float hemiDeltaR[3];
        float hemiDeltaG[3];
        float hemiDeltaB[3];
    };
    static_assert(sizeof(GpuDirLight)   == 24);
    static_assert(sizeof(GpuPointLight) == 36);
    static_assert(sizeof(GpuSpotLight)  == 56);
    static_assert(sizeof(GpuRectLight)  == 60);
    static_assert(sizeof(GpuLightsUbo)  == 1232);

    struct GpuClusterLight {
        float position[3];   float range;         // range 0 = infinite (three.js)
        float color[3];      float decay;         // color premultiplied by intensity
        float direction[3];  float cosAngleOuter; // spot cone; points carry -1.1/-1.05 (cone test → 1)
        float cosAngleInner;
        float radius;        // physical source radius (soft shadows)
        float cullRadius;    // conservative influence radius (range, or the atten<eps solve)
        float type;          // 0 = point, 1 = spot
    };
    static_assert(sizeof(GpuClusterLight) == 64);

    // Homogeneous fog (participating media). FogExp2.density maps directly
    // to sigma_t; linear Fog (near/far) is converted to an equivalent
    // density. Enabled flag = 0 short-circuits all fog work in the shaders.
    // anisotropy is the Henyey-Greenstein g for single-scattering.
    struct GpuFogUbo {
        float sigmaT[3];     // per-channel extinction (1/world unit)
        float enabled;       // 1.0 = fog active, 0.0 = disabled
        float color[3];      // inscatter tint (sRGB-linear)
        float anisotropy;    // HG g, clamped [-0.95, 0.95] by setFogAnisotropy
        float waterSurfaceY; // world-Y of the water surface; 1e30 = no limit
        float worldUp[3];    // world up axis (= camera.up) for sky aerial perspective
        // Unified fog medium (setHeightFog / resolved scene.fog) params,
        // MIRRORED from GpuCloudUbo so the deferred FILTER recombines
        // (deferred_filter_common.glsl, which binds only this fog UBO — not
        // the CloudUbo) can carry the same hetero extinction the shade pass
        // applies. 0 density = no air medium. Phase 2: scene.fog now feeds
        // these (its density/profile), so the froxel hetero path is the ONE
        // air medium; the sigmaT/color/enabled fields above are the medium's
        // beam-σ / albedo / present-flag for the volumetric consumers.
        float hfDensity;     // air-medium σ_t at baseY
        float hfBaseY;       // air-medium base world Y
        float hfFalloff;     // air-medium exponential height scale (m); huge ≈ uniform
        // Underwater murk (setUnderwaterMurk) — a SEPARATE homogeneous medium
        // clipped to BELOW waterSurfaceY (the water body's own absorption),
        // decoupled from the air fog in Phase 2 so a scene can hold clear air
        // above the waterline and murk below (the fjord). 0 density = off.
        float murkDensity;   // murk σ_t (1/m); 0 = off
        float murkColor[3];  // murk inscatter tint (sRGB-linear)
    };
    static_assert(sizeof(GpuFogUbo) == 76);

    // Volumetric cloud layer (VulkanRenderer::setClouds) + near-field
    // heterogeneous height fog (VulkanRenderer::setHeightFog). Both ride the
    // one binding-58 scalar UBO (they share wind + timeSec). clouds.enabled
    // == 0 short-circuits the far cloud march; heteroActive == 0 keeps the
    // froxel volumetrics on today's homogeneous path (off = free /
    // image-identical). Layout matches deferred_shade.comp / cloud_march /
    // froxel_inject / froxel_integrate's scalar CloudUbo block exactly.
    struct GpuCloudUbo {
        float enabled;      // 1.0 = far cloud march active
        float coverage;     // 0 = clear .. 1 = overcast
        float density;      // density multiplier
        float bottomY;      // shell base (world Y)
        float topY;         // shell top (world Y)
        float evolveSpeed;  // shape churn rate
        float timeSec;      // wall-clock seconds (wind scroll + evolution)
        float heteroActive; // 1.0 = heterogeneous near-field froxels (height fog on)
        float wind[3];      // m/s xz drift (y ignored)
        float hfDensity;    // height-fog σ_t at baseY (0 = height fog off)
        float hfBaseY;      // height-fog base world Y
        float hfFalloff;    // height-fog exponential height scale (m)
        float hfNoiseAmount;// 0 = smooth analytic .. 1 = fully noise-modulated
        float shadowActive; // 1.0 = cloud shadow map valid this frame (clouds on)
        float epoch;        // history generation — cloud_march rejects prev-epoch
                            // history (first-enable garbage, reconfigured decks)
    };
    static_assert(sizeof(GpuCloudUbo) == 68);

    struct DebugResolvePC {
        uint32_t view;      // 1 = normal, 2 = motion, 3 = ids, 4 = albedo
        uint32_t width;
        uint32_t height;
        uint32_t gbufWidth;
        uint32_t gbufHeight;
        float    motionGain;
    };

    struct GpuOverlayFogUbo {
        float fogActive;     // >0.5 = a medium is present this frame
        float hfDensity;     // air-medium σ_t at baseY (0 = no air fog)
        float hfBaseY;
        float hfFalloff;     // huge ≈ uniform
        float murkDensity;   // underwater-murk σ_t (0 = off)
        float waterSurfaceY; // world Y of the water surface (murk clip)
        float camWorldY;     // camera world Y
        float _pad0;
        float viewToWorldY[3];// world-Y row of the inverse-view
        float _pad1;
        float fogInscatter[3];// LINEAR air-fog in-scatter radiance (fade target)
        float _pad2;
        float murkInscatter[3];// LINEAR murk in-scatter radiance (fade target)
        float _pad3;
        // Lit particles (particle_light.comp): the LIT fragment path needs
        // the scene's display transform to land in the same domain as the
        // tonemapped background it alpha-blends over.
        float litActive;   // >0.5 = particle_light.comp ran this frame
        float exposure;    // FULL tone-map exposure (currentExposure())
        float toneMapMode; // threepp::ToneMapping as float (frag casts back)
        float _pad4;
    };
    static_assert(sizeof(GpuOverlayFogUbo) == 96);

    // Host mirror of gbuffer_indirect.vert's DrawInfo struct. Tight-
    // packed (136 bytes — the static_assert below is the authority — all
    // members naturally aligned to ≤ 8) so it matches the GLSL `scalar`
    // block layout used in the shader.
    struct DrawInfoGpu {
        float    model[16];        // 64
        uint64_t posAddr;          // 8
        uint64_t nrmAddr;          // 8
        uint64_t uvAddr;           // 8
        uint64_t prevPosAddr;      // 8
        uint64_t indexAddr;        // 8 (0 → non-indexed)
        uint64_t colorAddr;        // 8 (0 → no per-vertex color / vertexColors off)
        uint32_t instanceCustomIndex;
        uint32_t flags;            // bits 0..7 render flags | bits 8..15 semantic class id
        uint32_t indexed;
        float    polygonOffset;    // clip-z depth bias (reverse-Z: + = toward near = on top)
        uint32_t stableId;         // stable per-object instance id (-> outIds.y)
        uint32_t packedAttrs;      // BlasRecord::packedMask (also keeps 8-byte array stride)
    };
    static_assert(sizeof(DrawInfoGpu) == 136,
                  "DrawInfoGpu layout drifted from gbuffer_indirect.vert");

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_GPU_LAYOUTS_HPP
