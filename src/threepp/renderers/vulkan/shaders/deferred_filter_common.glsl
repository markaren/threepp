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
} cam;
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

// Fog transmittance over the camera→surface leg, clipped to y < waterSurfaceY
// (mirrors the shade pass's fogPathLength — keep in sync).
vec3 fogTransmittance(vec3 a, vec3 b) {
    if (fog.enabled < 0.5) return vec3(1.0);
    float d = distance(a, b);
    if (fog.waterSurfaceY < 1e29) {
        const float ya = a.y - fog.waterSurfaceY;
        const float yb = b.y - fog.waterSurfaceY;
        if (ya >= 0.0 && yb >= 0.0) return vec3(1.0);
        if (!(ya < 0.0 && yb < 0.0)) {
            const float t = ya / (ya - yb);
            d *= (ya < 0.0) ? t : (1.0 - t);
        }
    }
    return exp(-fog.sigmaT * d);
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
