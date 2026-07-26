// Single source of truth for Vulkan renderer constants and the MaterialDesc
// layout. Included by VulkanRenderer.cpp (host) and by the Vulkan shaders
// (deferred_shade.comp, event_shade.comp, probe_update.comp, …) via glslang's
// `#extension GL_GOOGLE_include_directive`. Cross-language: the C++ compiler
// defines __cplusplus and sees the C++ branch; the GLSL preprocessor doesn't,
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

// TLAS instance visibility groups (VkAccelerationStructureInstanceKHR.mask).
// Opaque + alpha-CUTOUT instances carry kRayMaskOpaque; alpha-BLEND and
// transmissive instances (text decals, alpha quads, glass — anything whose
// MaterialDesc has alphaCutoff < 0 or transmission > 0, except water) carry
// kRayMaskAlpha INSTEAD. Pure-visibility occlusion queries (env/sky gather,
// GI bounces, emissive-NEE shadow tests) trace with cullMask = kRayMaskOpaque
// so a decal's transparent quad never blocks IBL/GI/emissive light — the HW
// skips those instances entirely, no per-candidate alpha test needed. Every
// radiance/primary trace keeps cullMask 0xFF and sees both groups.
#define kRayMaskOpaque 0x01u
#define kRayMaskAlpha  0x02u

// Per-instance flag word — packed host-side (VulkanCoreIndirect.cpp) into
// DrawInfo.flags, carried through gbuffer.vert into the gbuffer IDs
// attachment's .z channel (rgba16ui — truncates to 16 bits, so flags live in
// bits 0..7 and the semantic CLASS id in bits 8..15). THE canonical bit
// layout; shader consumers go through the instance_flags.glsl accessors
// instead of raw masks. Bits 1..2 are reserved (documented for transmissive/
// thin-walled in an earlier design, never packed — kept so re-introducing
// them can't silently collide).
#define kInstFlagWater       0x01u// DisplacedMesh (FFT water / displaced surface)
#define kInstFlagSkinned     0x08u// GPU-skinned mesh
#define kInstFlagDoubleSided 0x10u// Side::Double material (±N = same surface)
#define kInstFlagDeformer    0x20u// persistent per-frame deformer (tet soft body)
#define kInstFlagTexAnim     0x40u// per-frame texture animation (Material::textureAnimatedHint)

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
        // Drives the ray-hit pass-through gate in lidar.rchit / probe_update.comp
        // (wrong-side hits skip the surface)
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
        // Stable per-Material-asset index, deduplicated by Material* pointer
        // when the matDescs buffer is built (VulkanRenderer.cpp). Adjacent
        // meshes that share one Material C++ object get the SAME value, so
        // a reproject/reuse consumer can accept cross-mesh-same-material
        // taps — kills the visible seam at tiled-wall boundaries during
        // camera motion. mesh-asset/material-asset only; not a hash.
        uint32_t materialAssetIdx;
        // Tiled world-anchored detail albedo (MaterialWithDetailMap):
        // LINEAR-space texture, 0.5 = neutral; gbuffer.frag modulates the
        // base albedo by mix(1, 2*detail, strength*fade). -1 = none.
        // Raster primary visibility only — ray-hit shading skips it.
        int32_t detailTexIndex;
        float detailRepeat;  // repeats per world meter (worldPos.xz anchored)
        float detailStrength;// 0..1
        // Detail NORMAL + ROUGHNESS layer (shares detailTexIndex's world-XZ
        // stochastic projection). -1 = none. Raster G-buffer only.
        int32_t detailNormalTexIndex;
        float detailNormalScale;  // tangent xy perturbation scale
        float detailRoughStrength;// 0..1 roughness modulation strength
        float _padDetail;
        // Foliage translucency / two-sided subsurface (MaterialWithTranslucency):
        // strength 0..1 + tint. Raster primary visibility only — deferred_shade
        // adds a wrap back-light + forward-scatter sun term plus a small back-N
        // ambient term; ray-hit shading (probe GI, reflections, lidar) skips it.
        // translucency == 0 → bit-exact no-op for all existing content.
        float translucencyColor[3];
        float translucency;
    };

    // Catches silent layout drift: if any field is added/removed/reordered
    // above, the size changes and this fires. Update the GLSL `MaterialDesc`
    // mirror below to match before bumping the expected size.
    static_assert(sizeof(MaterialDesc) == 512,
                  "MaterialDesc size changed - update the GLSL mirror in this file too.");
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
    uint  materialAssetIdx;
    int   detailTexIndex;// tiled world-anchored detail albedo; -1 = none
    float detailRepeat;  // repeats per world meter (worldPos.xz anchored)
    float detailStrength;// 0..1
    int   detailNormalTexIndex;// detail normal+roughness; -1 = none
    float detailNormalScale;   // tangent xy perturbation scale
    float detailRoughStrength; // 0..1 roughness modulation strength
    float _padDetail;
    vec3  translucencyColor;// foliage two-sided subsurface tint
    float translucency;     // 0 = off (raster primary shading only)
};

#endif  // __cplusplus

#endif  // THREEPP_VULKAN_SHARED_H
