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

// Emitter COVERAGE mode (emissive_lights.glsl): scenes with at most this many
// emissive INSTANCES get a per-light table at the tail of the EmTri buffer and
// every light is sampled every time; above it the coherent NEE falls back to
// the global power-CDF pick. Host (buildAndUploadEmissiveTris) writes the
// table only under this cap; the shaders read it only under this cap.
#define kEmissiveCoverMaxLights 8

// TLAS instance visibility groups (VkAccelerationStructureInstanceKHR.mask).
// Opaque + alpha-CUTOUT instances carry kRayMaskOpaque; alpha-BLEND and
// transmissive instances (text decals, alpha quads, glass — anything whose
// MaterialDesc has alphaCutoff < 0 or transmission > 0, except water) carry
// kRayMaskAlpha INSTEAD. Pure-visibility occlusion queries (env/sky gather,
// GI bounces, emissive-NEE shadow tests) trace with cullMask = kRayMaskOpaque
// so a decal's transparent quad never blocks IBL/GI/emissive light — the HW
// skips those instances entirely, no per-candidate alpha test needed. Every
// radiance/primary trace keeps cullMask 0xFF and sees every group.
//
// kRayMaskNoShadow is the same idea for OPAQUE geometry that must not occlude:
// a first-person viewmodel (MeshEntry::camAttached — anything parented under
// the camera). It rasterizes, shades and reflects normally, but since no
// occlusion cullMask includes 0x04 it casts no shadow and blocks no sky/GI
// light — otherwise the sun paints a pair of floating hands and a gun on the
// ground beside the player.
// kRayMaskSensorOnly is geometry that exists FOR THE SENSORS and for nothing
// else: a surface baked out of a Gaussian-splat scan (threepp::splats::
// bakeSurface), which must return lidar ranges without appearing in the
// picture the splat rasterizer already draws there. No radiance trace includes
// it — that is what kRayMaskAll is for, the "everything the camera may see"
// mask that the primary/reflection/refraction traces use where they used to
// pass 0xFF. The LIDAR pass keeps cullMask 0xFF and is therefore the only
// consumer that sees the group. Instances carry mask 0 (hit by nothing) until
// VulkanRenderer::setSensorOnlySurfaces(true) opts the scene in.
#define kRayMaskOpaque     0x01u
#define kRayMaskAlpha      0x02u
#define kRayMaskNoShadow   0x04u
#define kRayMaskSensorOnly 0x08u
#define kRayMaskAll        0x07u

// Per-instance flag word — packed host-side (VulkanCoreIndirect.cpp) into
// DrawInfo.flags, carried through gbuffer.vert into the gbuffer IDs
// attachment's .z channel (rgba16ui — truncates to 16 bits, so flags live in
// bits 0..7 and the semantic CLASS id in bits 8..15). THE canonical bit
// layout; shader consumers go through the instance_flags.glsl accessors
// instead of raw masks. Bits 1..2 are reserved (documented for transmissive/
// thin-walled in an earlier design, never packed — kept so re-introducing
// them can't silently collide).
#define kInstFlagWater       0x01u// DisplacedMesh (FFT water / displaced surface)
#define kInstFlagSplat       0x02u// Gaussian-splat pixel (composited by SplatPass,
                                  // not rasterized — its motion vector comes from
                                  // an alpha-weighted EXPECTED depth, so it is
                                  // exact for an opaque cloud and approximate
                                  // wherever several depths mix)
#define kInstFlagSkinned     0x08u// GPU-skinned mesh
#define kInstFlagDoubleSided 0x10u// Side::Double material (±N = same surface)
#define kInstFlagDeformer    0x20u// persistent per-frame deformer (tet soft body)
#define kInstFlagTexAnim     0x40u// per-frame texture animation (Material::textureAnimatedHint)
#define kInstFlagMoving      0x80u// moved-sticky (mirrors GeometryDesc.flags bit 0 at draw
                                  // time). Makes the PREV ids texel self-describing for the
                                  // moving-mesh trailing-edge guards: prev ids .x indexes the
                                  // prev frame's draw list, so geoms[pid-1] reads the WRONG
                                  // entry after any topology renumber (tile streaming).

// Ocean cascade-1 sample-domain rotation. The mid cascade carries only ~a
// dozen Fourier modes per axis (band-passed to λ ∈ [tileSize2, tileSize1]),
// so its tile repeats as a visible axis-aligned plaid across the ocean.
// Rotating WHERE cascade 1 is sampled (domain q = R·worldXZ) turns its
// repeat lattice diagonal — decorrelated from cascade 0's wind-aligned
// pattern and from its own crest direction — which breaks the periodicity
// perception without changing the wave statistics. The spectrum's windTheta
// is compensated host-side (+kOceanCascade1RotTheta) so the waves still
// PROPAGATE along the world wind direction; only the lattice rotates.
//
// Consumers that sample cascade 1 must all agree: water_displace.comp,
// foam_world.comp (via ocean_cascade.glsl) and the CPU mirror
// DisplacedMesh::sampleHeight. Height is a scalar (rotate the sample point
// only); horizontal displacement is a vector (rotate the sampled (dx,dz)
// BACK into world by the inverse rotation).
//
// θ = atan(1/2) ≈ 26.565°; sin = 1/√5, cos = 2/√5 — kept as literals so the
// C++ and GLSL sides fold identical values.
#define kOceanCascade1RotTheta 0.4636476090008061f
#define kOceanCascade1RotSin   0.4472135954999579f
#define kOceanCascade1RotCos   0.8944271909999159f

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
        // Terrain splat shading (MaterialWithTerrainMaps). Raster G-buffer
        // only. terrainWeightTexIndex >= 0 marks a terrain material:
        //   • terrainNormalTexIndex — WORLD-space normal map (mesh UVs);
        //     replaces the interpolated vertex normal so tiles of different
        //     LOD shade identically at shared borders (mips band-limit it).
        //   • weight map (mesh UVs) selects up to four repeating band sets
        //     (albedo overlay w/ height in A + normal/roughness), world-XZ
        //     anchored, stochastic-tiled + triplanar, height-blended. Band
        //     base roughness REPLACES material roughness where bands cover.
        int32_t terrainWeightTexIndex; // -1 = not a terrain material
        int32_t terrainNormalTexIndex; // -1 = keep vertex normals
        float terrainBandStrength;     // 0..1 albedo-overlay modulation
        float terrainNormalScale;      // band tangent perturbation scale
        float terrainRoughStrength;    // 0..1 band roughness modulation
        float terrainHeightBlend;      // height-blend sharpness (0 = linear)
        // MaterialWithEnvMap::envMapIntensity (MeshStandardMaterial's
        // env_map_intensity in Python). Scales the ENVIRONMENT-lit terms of
        // this material where it is the PRIMARY surface: the reflection ray's
        // env-miss radiance, the split-sum env specular, and the diffuse
        // env-ambient gather. Direct lights, probe GI and geometry hits are
        // NOT scaled — this is an IBL knob, not an exposure. 1.0 = unscaled,
        // which is what every material that never sets it carries.
        float envMapIntensity;
        float _padTerrain;
        int32_t terrainBandAlbedoTex[4]; // -1 = band inert
        int32_t terrainBandNormalTex[4]; // -1 = no relief for that band
        float terrainBandRepeat[4];      // repeats per world metre
        float terrainBandRough[4];       // base roughness per band
    };

    // Catches silent layout drift: if any field is added/removed/reordered
    // above, the size changes and this fires. Update the GLSL `MaterialDesc`
    // mirror below to match before bumping the expected size.
    static_assert(sizeof(MaterialDesc) == 608,
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
    int   terrainWeightTexIndex;// terrain band-weight map; -1 = not terrain
    int   terrainNormalTexIndex;// world-space normal map; -1 = vertex normals
    float terrainBandStrength;  // 0..1 albedo-overlay modulation
    float terrainNormalScale;   // band tangent perturbation scale
    float terrainRoughStrength; // 0..1 band roughness modulation
    float terrainHeightBlend;   // height-blend sharpness (0 = linear)
    float envMapIntensity;      // IBL scale for this material as PRIMARY surface; 1 = unscaled
    float _padTerrain;
    ivec4 terrainBandAlbedoTex; // per-band overlay (A = height); -1 = inert
    ivec4 terrainBandNormalTex; // per-band normal/roughness; -1 = none
    vec4  terrainBandRepeat;    // repeats per world metre
    vec4  terrainBandRough;     // base roughness per band
};

#endif  // __cplusplus

// ── ShadePush ───────────────────────────────────────────────────────────────
// The ONE push-constant block for the deferred-shade pipeline family
// (deferred_shade, cluster_build, froxel_inject, froxel_integrate,
// cloud_march, cloud_shadow — all created against the same 80-byte push
// range in DeferredShade::createPipeline). Each pass reads the fields it
// needs and leaves the rest zero; the host fills the struct by NAME
// (DeferredShade.cpp), so there is no positional index to drift between
// C++ and GLSL. Every member is a 4-byte scalar, so the C++ layout, std430
// and scalar block layout agree byte-for-byte (static_assert below).
//
// flags bits: 0 = shadows, 1 = RT env-visibility (AO/GI), 2 = denoise,
// 3 = ReSTIR DI, 4 = volumetric dir-light fog, 5-6 = G-buffer MSAA code
// (0 = 1x, 1 = 2x, 2 = 4x), 7 = dispatch B runs this frame, 8 = froxel LUT
// valid this frame, 9 = shadow-dwell kill switch, 10 = solid display-
// referred background (sky store skips the pre-exposure), 11 = a ParticleField
// density volume is live, 12 = a baked splat reflection volume is live
// (PRIMARY view only — see DeferredShade::DispatchParams::splatVolume).
//
// NO default member initializers on the C++ side: `ShadePush p{};` must
// zero-fill, exactly like the positional `uint32_t pc[19] = {}` blocks it
// replaced, so passes that leave a field unset push the same bytes.

#ifdef __cplusplus
namespace threepp::vulkan {
    struct ShadePush {
        uint32_t envMipCount;      // PMREM mip levels (>= 1); shade + cloud ambient LOD
        uint32_t width;            // deferred render extent (== G-buffer extent)
        uint32_t height;
        uint32_t flags;            // bit table above
        uint32_t frame;            // frame/sample index — temporal jitter seeds
        uint32_t emissiveCount;    // # emissive triangles (0 = none)
        float emissiveTotalPower;  // emissive power-CDF total (denominator)
        float fireflyClamp;        // luminance cap for emissive NEE (large = disabled)
        float oceanFineTileSize;   // FFT fine-cascade tile (m); 0 = no ocean fine normal
        float oceanFoamTileSize;   // world-space foam tile (m); 0 = no foam sampling
        float volDensity;          // spot-beam scattering σ (1/m); 0 = beams off
        float volAniso;            // Henyey-Greenstein g for the beam phase
        float starIntensity;       // procedural sky star field; 0 = off
        float camDelta;            // camera WORLD translation this frame (m)
        float camRot;              // camera forward-direction change this frame (rad)
        float timeSec;             // wall-clock seconds (fps-independent drift)
        float sunTanHalfAngle;     // tan(sun angular RADIUS); 0 = hard 1-ray shadow
        uint32_t clusterLightCount;// # lights in the cluster buffer (0 = none)
        uint32_t shadeMode;        // 0 = dispatch A; 1 = MSAA per-sample dispatch B
        uint32_t preExpBits;       // float-bits pre-exposure baked into sceneHdr stores
    };
    static_assert(sizeof(ShadePush) == 80,
                  "ShadePush size changed - update the GLSL mirror in this file "
                  "and the 80-byte push range in DeferredShade::createPipeline.");
}

#else  // GLSL

struct ShadePush {
    uint  envMipCount;
    uint  width;
    uint  height;
    uint  flags;
    uint  frame;
    uint  emissiveCount;
    float emissiveTotalPower;
    float fireflyClamp;
    float oceanFineTileSize;
    float oceanFoamTileSize;
    float volDensity;
    float volAniso;
    float starIntensity;
    float camDelta;
    float camRot;
    float timeSec;
    float sunTanHalfAngle;
    uint  clusterLightCount;
    uint  shadeMode;
    uint  preExpBits;
};
// Every consumer declares:
//   layout(push_constant) uniform Pc { ShadePush pc; };
// so field access stays `pc.<field>`, unchanged from the old inline blocks.

#endif  // __cplusplus

#endif  // THREEPP_VULKAN_SHARED_H
