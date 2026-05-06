#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable

#include "vulkan_shared.h"

// G-buffer fragment for the hybrid raster prepass. Emits world-space
// normal (with normal map applied), screen-space motion vector, and
// per-pixel IDs/flags. Depth is written automatically.
//
// Stage 1A.5+: normal mapping is done here via screen-space derivatives
// of vWorldPos + vUv. Without it, primary surfaces look flat; chit's
// non-hybrid path samples the normal map and most assets rely on it for
// surface detail (mortar lines, fabric weave, brick relief, etc.).
// Albedo / roughness / metalness sampling stays in raygen.

layout(set = 0, binding = 2, scalar) readonly buffer GbufMatBuf {
    MaterialDesc gbufMats[];
};
layout(set = 0, binding = 3) uniform sampler2D gbufAlbedoMaps[kMaxMaterialTextures];

layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) in vec4 vCurrClipUnjit;
layout(location = 2) in vec4 vPrevClip;
layout(location = 3) flat in uint vInstanceIdx;
layout(location = 4) flat in uint vFlags;
layout(location = 5) in vec2 vUv;
layout(location = 6) in vec3 vWorldPos;

// Attachment 0: world-space normal (rgba16f). Stage 2 will pack roughness
// into .w; stage 1 leaves it zero. rgba16f is necessary for ocean wave
// normals — rgba8 loses too much precision on the FFT-driven detail.
layout(location = 0) out vec4 outNormal;

// Attachment 1: motion vector in NDC delta (rg16f). raygen converts to
// pixel-space when sampling the temporal accumulator. Stationary pixels
// produce zero motion.
layout(location = 1) out vec4 outMotion;

// Attachment 2: per-pixel IDs + flags (rgba16ui).
//   .x = instanceCustomIndex + 1 (matches raygen Payload.hitInstanceId
//        convention; 0 reserved for sky/miss because the render pass
//        clears IDs to 0 before any draw)
//   .y = mesh-ID for the reproject same-mesh guard (currently == .x but
//        kept separate so future stages can decouple)
//   .z = flags (bit 0 is_water, bit 1 transmissive, bit 2 thinWalled, ...)
//   .w = reserved
layout(location = 2) out uvec4 outIds;

// Attachment 3: material UV (rg16f, .b/.a unused). raygen samples the
// bindless material texture array at this UV when hybrid mode skips the
// primary traceRayEXT (Stage 1A) — chit normally interpolates UV from
// triangle vertices; we precompute it here so raygen doesn't have to.
layout(location = 3) out vec4 outUv;

void main() {
    vec3 N = normalize(vWorldNormal);

    // Normal map perturbation. TBN derived from screen-space derivatives:
    // (dpx, dpy) = world-space partial derivatives of position
    // (duvx, duvy) = uv derivatives.
    // Tangent T = (dpx · duvy.y - dpy · duvx.y) / det. Same construction
    // as chit's per-triangle tangent (closest_hit.rchit:751-777), just
    // expressed via fragment-shader derivatives so we don't need triangle
    // vertex/UV data here.
    //
    // Skipped on water (is_water flag bit 0): the FFT cascade normal map
    // is a chit-specific input applied as part of its water BSDF, not a
    // generic surface perturbation — running it here would tile the foam
    // normal across the wave geometry and produce visible cellular noise.
    const MaterialDesc m = gbufMats[vInstanceIdx];
    const bool isWater = (vFlags & 1u) != 0u;
    if (m.normalTexIndex >= 0 && !isWater) {
        const vec3 dpx = dFdx(vWorldPos);
        const vec3 dpy = dFdy(vWorldPos);
        const vec2 duvx = dFdx(vUv);
        const vec2 duvy = dFdy(vUv);
        const float det = duvx.x * duvy.y - duvy.x * duvx.y;
        if (abs(det) > 1e-8) {
            vec3 T = (dpx * duvy.y - dpy * duvx.y) / det;
            T = T - dot(T, N) * N;// Gram-Schmidt
            const float Tlen = length(T);
            if (Tlen > 1e-6) {
                T /= Tlen;
                const vec3 B = cross(N, T);
                const int nidx = clamp(m.normalTexIndex, 0, int(kMaxMaterialTextures) - 1);
                const vec2 uvN = (m.uvTransformNormal * vec3(vUv, 1.0)).xy;
                vec3 ns = texture(gbufAlbedoMaps[nidx], uvN).rgb * 2.0 - 1.0;
                ns.xy *= m.normalScale;
                ns.z = sqrt(max(0.0, 1.0 - dot(ns.xy, ns.xy)));
                N = normalize(T * ns.x + B * ns.y + N * ns.z);
            }
        }
    }

    // Remap [-1, 1] → [0, 1] so negative components are visible in the
    // BGRA8_UNORM debug blit (which clamps negatives to 0). raygen reverses
    // this with `n * 2 - 1` when reading the attachment.
    outNormal = vec4(N * 0.5 + 0.5, 0.0);

    vec2 currNDC = vCurrClipUnjit.xy / vCurrClipUnjit.w;
    vec2 prevNDC = vPrevClip.xy      / vPrevClip.w;
    // Motion vector points from current pixel to where the surface WAS
    // last frame. Stationary surface → zero motion → reproject samples
    // self-pixel. Convention matches raygen's existing reproject expectation.
    vec2 motion = prevNDC - currNDC;
    outMotion = vec4(motion, 0.0, 0.0);

    // +1 so the renderpass's clear-to-0 means "sky/no draw", matching
    // raygen's Payload.hitInstanceId convention exactly.
    outIds = uvec4(vInstanceIdx + 1u, vInstanceIdx + 1u, vFlags, 0u);

    outUv = vec4(vUv, 0.0, 0.0);
}
