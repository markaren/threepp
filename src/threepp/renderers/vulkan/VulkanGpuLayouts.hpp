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

#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kMaxShadow*

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
        float shadowSlot;  // cube layer in the point shadow maps (ShadowMapPass), -1 = none
    };
    struct GpuSpotLight {
        float position[3];   float range;
        float color[3];      float decay;
        float direction[3];  // toward target (emission direction)
        float cosAngleOuter; // cos(angle)
        float cosAngleInner; // cos(angle * (1-penumbra))
        float radius;        // physical source radius (world units) → RT soft shadows; 0 = hard
        float shadowSlot;    // layer in the spot shadow maps (ShadowMapPass), -1 = none
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
        // three.js scene.environmentRotation / environmentIntensity and
        // scene.backgroundRotation / backgroundIntensity / backgroundBlurriness,
        // with GLRenderer's conventions (9804c236). Appended LAST, like the hemi
        // rows. The rotations are WORLD -> ENV (the transpose of the Euler's
        // rotation), stored as three GLSL mat3 columns. sampleEnvLod applies
        // envRot + envIntensity to every environment read; the visible sky
        // applies bgRot + bgIntensity and reads the PMREM at LOD
        // bgBlurriness * (mips - 1). envKnobFlags bit 0 = envRot is not the
        // identity, bit 1 = bgRot is not the identity: the shaders skip the
        // multiply otherwise, so a default scene samples exactly as before.
        float    envRot[9];
        float    envIntensity;
        float    bgRot[9];
        float    bgIntensity;
        float    bgBlurriness;
        uint32_t envKnobFlags;
        float    envKnobPad[2];
    };
    static_assert(sizeof(GpuDirLight)   == 24);
    static_assert(sizeof(GpuPointLight) == 40);
    static_assert(sizeof(GpuSpotLight)  == 60);
    static_assert(sizeof(GpuRectLight)  == 60);
    // splat_common.glsl declares a prefix of this block and skips to the
    // environment knobs with a float array sized from it: a size change here
    // moves that skip (it ends 96 bytes before the end).
    static_assert(sizeof(GpuLightsUbo)  == 1392);

    struct GpuClusterLight {
        float position[3];   float range;         // range 0 = infinite (three.js)
        float color[3];      float decay;         // color premultiplied by intensity
        float direction[3];  float cosAngleOuter; // spot cone; points carry -1.1/-1.05 (cone test → 1)
        float cosAngleInner;
        float radius;        // physical source radius (soft shadows)
        float cullRadius;    // conservative influence radius (range, or the atten<eps solve)
        float type;          // 0 = point, 1 = spot
        float shadowSlot;    // layer in the point (cube) or spot shadow maps, by type; -1 = none
    };
    static_assert(sizeof(GpuClusterLight) == 68);

    // The shadow maps of one view (ShadowMapPass), set 0 binding 83 of the
    // deferred shade's set: shadow_maps.glsl's ShadowMapsUbo, scalar layout.
    // Clip matrices are world -> clip with the reverse-Z projection the raster
    // camera uses (reverseZVk); the shaders turn clip y into the texture's v
    // as the raster's y flip wrote it. All zero (flags 0) when the maps are
    // not in use: every lookup then answers "no map here".
    struct GpuShadowMapsUbo {
        float    cascadeVP[kMaxShadowCascades][16];
        float    cascadeSphere[kMaxShadowCascades][4];// xyz centre, w radius (world): the pick
        float    cascadeParams[kMaxShadowCascades][4];// x texel (m), y depth units per metre, z PCF radius (texels), w 0
        float    spotVP[kMaxShadowSpots][16];
        float    pointPosFar[kMaxShadowPoints][4];// xyz light position, w far
        // Per lamp slot, spots first, then points at kMaxShadowSpots + slot:
        // bias (depth units, LightShadow::bias), normal offset (m, 0 = auto),
        // PCF radius (texels), intensity.
        float    lampParams[kMaxShadowSpots + kMaxShadowPoints][4];
        // Per lamp slot: x = one texel's angle (tan) at unit distance, y near,
        // z far, w map size.
        float    lampTexel[kMaxShadowSpots + kMaxShadowPoints][4];
        uint32_t cascadeCount;
        uint32_t spotCount;
        uint32_t pointCount;
        uint32_t flags;// bit 0 = the sun's cascades are valid this frame
        float    sunBias;        // LightShadow::bias, depth units
        float    sunNormalBias;  // LightShadow::normalBias, metres (0 = the texel-scaled default)
        float    sunIntensity;   // LightShadow::intensity
        float    sunDir[3];      // toward the sun (dir light 0)
        float    cascadeMapSize;
    };
    static_assert(sizeof(GpuShadowMapsUbo) == 1964);

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
        float murkDensity;   // murk σ_t (1/m); 0 = off. The clearest channel of murkSigma
        float murkColor[3];  // murk inscatter tint (sRGB-linear)
        // Appended, so a shader block that stops at murkColor still lines up.
        float murkSigma[3];  // murk σ_t per channel; all three = murkDensity for the scalar murk
        float murkAmbientK;  // setUnderwaterAmbientFalloff; 0 = the ambient terms ignore depth
    };
    static_assert(sizeof(GpuFogUbo) == 92);

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
    // packed (144 bytes — the static_assert below is the authority — all
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
        uint64_t uv2Addr;          // 8 ("uv2", TEXCOORD_1, float; 0 → none, the aoMap reads uv)
        uint32_t instanceCustomIndex;
        uint32_t flags;            // bits 0..7 render flags | bits 8..15 semantic class id
        uint32_t indexed;
        float    polygonOffset;    // clip-z depth bias (reverse-Z: + = toward near = on top)
        uint32_t stableId;         // stable per-object instance id (-> outIds.y)
        uint32_t packedAttrs;      // BlasRecord::packedMask (also keeps 8-byte array stride)
    };
    static_assert(sizeof(DrawInfoGpu) == 144,
                  "DrawInfoGpu layout drifted from gbuffer_indirect.vert");

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_GPU_LAYOUTS_HPP
