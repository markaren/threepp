// What a ray finds at a triangle hit: the interpolated UV, the interpolated
// world normal + UV, and a bindless material texture sampled there (at the
// finest mip, or over a footprint from hitUvFootprint). Shared by the
// deferred shade's reflection/GI/shadow paths and the probe update; until
// 2026-10-10 each had its copy.
//
// Needs from the includer: geometry_fetch.glsl, its `geoms[]` buffer, the
// bindless `albedoMaps[]` array, kMaxMaterialTextures (vulkan_shared.h) and
// GL_EXT_nonuniform_qualifier.

#ifndef HIT_FETCH_GLSL
#define HIT_FETCH_GLSL

// Interpolated UV at a hit (lean — UV only, for the alpha-tested shadow test).
// Mirrors fetchHit's index + barycentric interpolation.
vec2 fetchUvAt(int instIdx, int primId, vec2 bary) {
    const GeometryDesc g = geoms[instIdx];
    if (g.uvAddress == 0ul) return vec2(0.0);
    const uvec3 idx = gfetchTri(g, primId);
    // Packed-aware (unorm16x2 on packed static geometry — see packedAttrs).
    const vec2 u0 = gfetchUv(g, idx.x);
    const vec2 u1 = gfetchUv(g, idx.y);
    const vec2 u2 = gfetchUv(g, idx.z);
    return (1.0 - bary.x - bary.y) * u0 + bary.x * u1 + bary.y * u2;
}

// Interpolated world-space normal + UV at a reflection hit. Mirrors
// closest_hit's GeometryDesc + barycentric interpolation + world transform.
void fetchHit(int instIdx, int primId, vec2 bary, mat4x3 worldToObj,
              out vec3 Nworld, out vec2 uvOut) {
    const GeometryDesc g = geoms[instIdx];
    const uvec3 idx = gfetchTri(g, primId);
    const float w = 1.0 - bary.x - bary.y;

    // gfetch* honour GeometryDesc.packedAttrs (oct-snorm16 normals / unorm16
    // uvs on packed static geometry, tightly-packed float otherwise).
    const vec3 n0 = gfetchNormal(g, idx.x);
    const vec3 n1 = gfetchNormal(g, idx.y);
    const vec3 n2 = gfetchNormal(g, idx.z);
    const vec3 nObj = normalize(w * n0 + bary.x * n1 + bary.y * n2);
    Nworld = normalize(transpose(mat3(worldToObj)) * nObj);

    uvOut = vec2(0.0);
    if (g.uvAddress != 0ul) {
        const vec2 u0 = gfetchUv(g, idx.x);
        const vec2 u1 = gfetchUv(g, idx.y);
        const vec2 u2 = gfetchUv(g, idx.z);
        uvOut = w * u0 + bary.x * u1 + bary.y * u2;
    }
}

// A material texture's colour at a hit, filtered over the ray's footprint
// (hitUvFootprint). The alpha tests below stay on the finest mip: a cutout's
// outline is not this function's to change.
vec3 hitTex(int texIndex, mat3 uvXform, vec2 uv, vec3 fallback, vec2 duvA, vec2 duvB) {
    if (texIndex < 0) return fallback;
    const int  i = clamp(texIndex, 0, int(kMaxMaterialTextures) - 1);
    const mat2 J = mat2(uvXform);
    return fallback * textureGrad(albedoMaps[nonuniformEXT(i)], (uvXform * vec3(uv, 1.0)).xy, J * duvA, J * duvB).rgb;
}

// Sample a bindless material texture at a reflection-hit UV (compute has no
// derivatives → explicit mip 0). texIndex < 0 → return `fallback` unchanged.
vec3 hitTex(int texIndex, mat3 uvXform, vec2 uv, vec3 fallback) {
    if (texIndex < 0) return fallback;
    const int i = clamp(texIndex, 0, int(kMaxMaterialTextures) - 1);
    return fallback * textureLod(albedoMaps[nonuniformEXT(i)], (uvXform * vec3(uv, 1.0)).xy, 0.0).rgb;
}

#endif// HIT_FETCH_GLSL
