// Single source of truth for Vulkan path-tracer constants and MaterialDesc
// layout. Included by VulkanRenderer.cpp (host) and by the path-tracer
// shaders (closest_hit, closest_hit_alpha, photon_chit, etc.) via glslang's
// `#extension GL_GOOGLE_include_directive`. Cross-language: the C++ compiler
// defines __cplusplus and sees the C++ branch; GLSL preprocessor doesn't,
// so it sees the GLSL branch.
//
// Adding or removing a MaterialDesc field requires editing only this file —
// every consumer picks up the change through a clean rebuild. The
// static_assert in the C++ branch catches drift between the two structs.

#ifndef THREEPP_VULKAN_SHARED_H
#define THREEPP_VULKAN_SHARED_H

// Bindless material-texture array size. Must match descriptor pool size and
// `albedoMaps[kMaxMaterialTextures]` in every shader. Bumping requires a
// clean rebuild so every translation unit picks up the new size.
#define kMaxMaterialTextures 2048

// Photon-map cell hash space. kPhotonGridSize cells × kPhotonsPerCell slots
// each form the storage for caustic photon emit + gather.
#define kPhotonGridBits  16
#define kPhotonGridSize  (1u << kPhotonGridBits)
#define kPhotonsPerCell  8u

// Photon emit raygen dimensions: kPhotonEmitDim × kPhotonEmitDim paths/frame.
// 256² = 65 536 photons/frame. Earlier 512² was 4× this — visibly diminishing
// returns past ~64 K because per-cell capacity (kPhotonsPerCell = 8) saturates
// quickly on hot caustic patches and overflow scaling absorbs the rest.
#define kPhotonEmitDim   256

// World-space grid cell size (metres) — same value used by photon_emit.rgen
// when depositing and closest_hit.rchit when gathering. They must agree or
// photons land in cells the gather can't find.
#define kGatherRadius    0.15

#ifdef __cplusplus

#include <cstdint>

namespace threepp::vulkan_pt {

    struct MaterialDesc {
        float albedo[3];
        float roughness;
        float metalness;
        float emissive[3];
        float emissiveIntensity;
        int32_t albedoTexIndex;
        int32_t roughnessTexIndex;
        int32_t metalnessTexIndex;
        int32_t normalTexIndex;
        float normalScale[2];
        float alphaCutoff;
        float transmission;
        float ior;
        int32_t transmissionTexIndex;
        float clearcoat;
        float clearcoatRoughness;
        int32_t clearcoatTexIndex;
        int32_t clearcoatRoughnessTexIndex;
        float attenuationColor[3];
        float attenuationDistance;
        int32_t emissiveTexIndex;
        float specularIntensity;
        float specularColor[3];
        float sheenColor[3];
        float sheenRoughness;
        // Side enum (matches threepp::Side): 0 = Front, 1 = Back, 2 = Double.
        // Drives the chit pass-through gate (wrong-side hits skip the surface)
        // and the raster gbuffer cull mode (BACK / FRONT / NONE respectively).
        int32_t sideMode;
        float uvTransform[9];
        int32_t occlusionTexIndex;
        float uvTransformNormal[9];
        float uvTransformRoughMetal[9];
        float uvTransformEmissive[9];
        float uvTransformOcclusion[9];
        float uvTransformClearcoat[9];
        float uvTransformClearcoatRough[9];
        float uvTransformTransmission[9];
        float iridescence;
        float iridescenceIOR;
        float iridescenceThicknessNm;
        float dispersion;
        float thickness;
        int32_t thinWalled;
    };

    // Catches silent layout drift: if any field is added/removed/reordered
    // above, the size changes and this fires. Update the GLSL `MaterialDesc`
    // mirror below to match before bumping the expected size.
    static_assert(sizeof(MaterialDesc) == 464,
                  "MaterialDesc size changed — update the GLSL mirror in this file too.");
}

#else  // GLSL

struct MaterialDesc {
    vec3  albedo;
    float roughness;
    float metalness;
    vec3  emissive;
    float emissiveIntensity;
    int   albedoTexIndex;
    int   roughnessTexIndex;
    int   metalnessTexIndex;
    int   normalTexIndex;
    vec2  normalScale;
    float alphaCutoff;
    float transmission;
    float ior;
    int   transmissionTexIndex;
    float clearcoat;
    float clearcoatRoughness;
    int   clearcoatTexIndex;
    int   clearcoatRoughnessTexIndex;
    vec3  attenuationColor;
    float attenuationDistance;
    int   emissiveTexIndex;
    float specularIntensity;
    vec3  specularColor;
    vec3  sheenColor;
    float sheenRoughness;
    // 0 = Front (cull back), 1 = Back (cull front), 2 = Double (no cull).
    int   sideMode;
    mat3  uvTransform;
    int   occlusionTexIndex;
    mat3  uvTransformNormal;
    mat3  uvTransformRoughMetal;
    mat3  uvTransformEmissive;
    mat3  uvTransformOcclusion;
    mat3  uvTransformClearcoat;
    mat3  uvTransformClearcoatRough;
    mat3  uvTransformTransmission;
    float iridescence;
    float iridescenceIOR;
    float iridescenceThicknessNm;
    float dispersion;
    float thickness;
    int   thinWalled;
};

#endif  // __cplusplus

#endif  // THREEPP_VULKAN_SHARED_H
