// deferred_filter_common.glsl - shared declarations + helpers for the two
// demodulated-lighting FILTER + COMPOSITE compute stages
// (deferred_gi_filter.comp, deferred_refl_filter.comp). Both read the same
// material G-buffer and read-modify-write the same sceneHdr target, so the
// CameraUbo (binding 0), the G-buffer textures (bindings 3-6), the sceneHdr out
// image (binding 7), the scene FogUbo (binding 33), the shared push-constant
// block, LUMA, and the world-reconstruction / fog-extinction / MSAA-coverage
// helpers live here, included by each stage.
//
// Not standalone: these rely on the includer's #version + extensions
// (GL_EXT_scalar_block_layout for the scalar FogUbo, GL_GOOGLE_include_directive
// for the include), exactly like the deferred_shade_NN_*.glsl split.

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 viewInverse;
    mat4 projInverse;
    vec4 jitter;
    vec4 camAux;// .x = parallel projection, .yzw = camera world forward
} cam;
#include "camera_ray.glsl"// camRayOrigin — perspective AND ortho
layout(set = 0, binding = 3) uniform sampler2D  gbufNormalTex; // xyz=n*0.5+0.5, w=roughness (<0 unlit)
layout(set = 0, binding = 4) uniform sampler2D  gbufDepthTex;  // x = NDC z [0,1]
layout(set = 0, binding = 5) uniform usampler2D gbufIdsTex;    // x = instanceId+1 (0 = sky)
layout(set = 0, binding = 6) uniform sampler2D  gbufAlbedoTex; // rgb=albedo, a=metalness
layout(set = 0, binding = 7,  rgba16f) uniform image2D outImage;            // sceneHdr (recombine: read+write)

// Scene fog — same UBO the shade pass consumes (binding 33 of the shared set).
// The GI + reflection RECOMBINES below add radiance into sceneHdr AFTER the
// shade pass fogged the base — these additions must carry the same primary-
// distance extinction or they glow through the murk. (Extinction only: the
// inscatter term was already added once by the shade pass.)
layout(set = 0, binding = 33, scalar) uniform FogUbo {
    vec3  sigmaT;
    float enabled;
    vec3  color;
    float anisotropy;
    float waterSurfaceY;
    vec3  worldUp;       // present in GpuFogUbo — declared to align the hf offsets
    // Height-fog (setHeightFog) params — mirror GpuCloudUbo's hf* so the filter
    // recombines carry the SAME hetero extinction the shade pass applies (the
    // froxel LUT/CloudUbo are NOT bound here → the closed form below stands in).
    float hfDensity;     // air-medium σ_t at baseY (0 = no air medium)
    float hfBaseY;       // air-medium base world Y
    float hfFalloff;     // air-medium exponential height scale (m)
    // Underwater murk (setUnderwaterMurk) — a separate homogeneous medium clipped
    // to below waterSurfaceY. Composes with the air extinction (see fogTransmittance).
    float murkDensity;   // murk σ_t (1/m; 0 = off)
    vec3  murkColor;     // murk tint (unused here — the recombine carries extinction only)
} fog;

layout(push_constant) uniform Pc {
    uint preExpBits;// float-bits: pre-exposure the shade stored sceneHdr with —
                    // the recombine adds bake the same factor (1.0 = legacy no-op)
    uint width; uint height; uint step;
    uint srcMode;// 0 = indirect (raw, pass 0), 1 = atrousA, 2 = atrousB
    uint dstMode;// 0 = atrousA, 1 = atrousB, 2 = recombine → sceneHdr
    uint feedback;// 1 = also write src (the 1st-pass filtered) back as temporal history
    uint channel;// RESERVED / UNUSED - the GI-vs-reflection choice is now the
                 // pipeline (deferred_gi_filter vs deferred_refl_filter), not
                 // this slot. Kept so the 10-uint/40-byte block stays byte-for-
                 // byte the shared pipeline layout (deferred_shade.comp 19-uint).
    uint msaaInfo;// bits 0..2 = G-buffer MSAA sample count (≤1 = off), bit 4 = shade dispatch B active
    uint _p9;
} pc;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 worldFromDepth(ivec2 q, float depth) {
    const vec2 uv  = (vec2(q) + 0.5) / vec2(float(pc.width), float(pc.height));
    // + jitter: the G-buffer was rasterized with a jittered projection, so the
    // depth at this texel belongs to the jittered ray — same convention as the
    // shade pass's reconstruction (keeps plane edge stops + fog on-surface).
    const vec2 ndc = vec2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0)) + cam.jitter.xy;
    const vec4 vh  = cam.projInverse * vec4(ndc, depth, 1.0);
    return (cam.viewInverse * vec4(vh.xyz / vh.w, 1.0)).xyz;
}

// Where texel q is looked at FROM — the eye under a perspective camera, this
// texel's own near-plane point under a parallel one. The à-trous plane stops
// scale their tolerance by the view leg; measuring that leg from a shared eye
// under an ortho camera (which sits wherever it had to to clear the scene
// bounds) would make the tolerance a function of that arbitrary distance.
vec3 camOriginAt(ivec2 q) {
    const vec2 uv  = (vec2(q) + 0.5) / vec2(float(pc.width), float(pc.height));
    const vec2 ndc = vec2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0)) + cam.jitter.xy;
    return camRayOrigin(ndc);
}

// Closed-form exponential-height-fog optical depth along [a,b] — KEEP IN SYNC
// with heightFogOpticalDepth in deferred_shade_60_fog_volumetrics.glsl. Ignores
// the noise modulation (a smooth mean, exactly what a recombine extinction
// wants). Zero when height fog is off.
float heightFogOpticalDepth(vec3 a, vec3 b) {
    if (fog.hfDensity <= 0.0) return 0.0;
    const float H   = max(fog.hfFalloff, 1e-3);
    const float ya  = max(a.y - fog.hfBaseY, 0.0);
    const float yb  = max(b.y - fog.hfBaseY, 0.0);
    // Clamp the leg (distance()² overflows fp32 for a sentinel/near-infinite end
    // point → Inf, then Inf·0 → NaN) and saturate the optical depth so every
    // exp(-od) sees a finite value. KEEP IN SYNC with the shade shader.
    const float len = min(distance(a, b), 1.0e7);
    // Overflow-safe DIFFERENCE form (ea−eb)/x with a Taylor fallback near x→0 —
    // ea, eb ≤ 1 (never overflow), avoids the fp32 cancellation of a huge-H profile.
    // KEEP IN SYNC with deferred_shade_60_fog_volumetrics.glsl.
    const float ea = exp(-ya / H);
    const float eb = exp(-yb / H);
    const float x  = (yb - ya) / H;
    const float f  = (abs(x) < 1e-3) ? (ea * (1.0 - 0.5 * x + x * x * (1.0 / 6.0)))
                                     : ((ea - eb) / x);
    return min(fog.hfDensity * len * f, 80.0);
}

// Fog transmittance over the camera→surface leg — the GI/reflection recombine
// must carry the SAME extinction the shade pass applied to the base, or the
// added radiance glows through the fog (the "fog missed ocean water" class of
// bug). Composes BOTH unified media multiplicatively (their optical depths add),
// mirroring the shade pass:
//   AIR medium (scene.fog / setHeightFog) → applyHeteroSurfaceFog's whole-path
//     closed-form height-fog extinction (no froxel LUT bound here — an accepted
//     approximation of LUT+tail); NOT water-clipped. The homogeneous branch is
//     vestigial (the air medium is always hetero when present).
//   MURK (setUnderwaterMurk) → applyMurk's homogeneous Beer-Lambert over the
//     BELOW-waterSurfaceY leg portion.
// Byte-identical (returns 1) when no medium is present.
vec3 fogTransmittance(vec3 a, vec3 b) {
    vec3 T = vec3(1.0);
    // Air medium (unclipped).
    if (fog.hfDensity > 0.0)
        T = exp(-vec3(heightFogOpticalDepth(a, b)));
    else if (fog.enabled > 0.5)
        T = exp(-fog.sigmaT * distance(a, b));// vestigial homogeneous path
    // Underwater murk (below waterSurfaceY only).
    if (fog.murkDensity > 0.0) {
        float d = distance(a, b);
        if (fog.waterSurfaceY < 1e29) {
            const float ya = a.y - fog.waterSurfaceY;
            const float yb = b.y - fog.waterSurfaceY;
            if (ya >= 0.0 && yb >= 0.0) d = 0.0;
            else if (!(ya < 0.0 && yb < 0.0)) {
                const float t = ya / (ya - yb);
                d *= (ya < 0.0) ? t : (1.0 - t);
            }
        }
        T *= exp(-vec3(fog.murkDensity) * d);
    }
    return T;
}

// Coverage weight for the GI/reflection RECOMBINES at MSAA complex (edge) pixels.
// indirectImage/reflectImage carry the DOMINANT sample's UNWEIGHTED values (they
// are temporal accumulators — weighting must happen here, at consumption), but at
// a complex pixel the shade pass stored only domWeight of the dominant surface:
// dispatch B added the geometry minority INLINE (with its own GI + reflection
// folded in) and the sky fraction is a background colour with neither. Adding the
// dominant GI/reflection at weight 1 therefore over-added (1-domWeight)·(GI+refl)
// at every edge pixel — a one-pixel bright rim hugging every silhouette (the
// "glowing outline", loudest in fog where the recombined GI rides on bright haze).
//   dispatch B active  → weight = dominant-cluster coverage (ids.w bits 0..3).
//   dispatch B off     → the geometry minority folded INTO the dominant surface,
//                        so only the sky fraction (ids.w bits 6..8) is excluded.
// (Known approximation: when B bails on an unlit/water minority rep the shade
// pass folds that weight back, while this uses the dominant coverage — the rare
// affected edge under-adds GI by the minority fraction instead of glowing.)
float recombineCoverage(ivec2 q) {
    const uint K = pc.msaaInfo & 0x7u;
    if (K <= 1u) return 1.0;
    const uint w = texelFetch(gbufIdsTex, q, 0).w;
    if ((w & 0x8000u) == 0u) return 1.0;// not a complex pixel
    if ((pc.msaaInfo & 0x10u) != 0u)
        return float(bitCount(w & 0xFu)) / float(K);
    return 1.0 - float((w >> 6u) & 0x7u) / float(K);
}
