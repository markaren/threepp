#version 460

// G-buffer fragment for the hybrid raster prepass. Emits world-space
// normal, screen-space motion vector and per-pixel IDs/flags. Depth is
// written automatically into the depth attachment.
//
// Material lookup (albedo, roughness, metalness, textures) is intentionally
// deferred to PT raygen for stage 1: one bindless texture fetch on primary
// pixels has negligible cost relative to the BVH primary trace that hybrid
// has just eliminated, and it keeps this shader's descriptor set minimal.

layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) in vec4 vCurrClipUnjit;
layout(location = 2) in vec4 vPrevClip;
layout(location = 3) flat in uint vInstanceIdx;
layout(location = 4) flat in uint vFlags;
layout(location = 5) in vec2 vUv;

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
    vec3 n = normalize(vWorldNormal);
    // Remap [-1, 1] → [0, 1] so negative components are visible in the
    // BGRA8_UNORM debug blit (which clamps negatives to 0). raygen reverses
    // this with `n * 2 - 1` when reading the attachment.
    outNormal = vec4(n * 0.5 + 0.5, 0.0);

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
