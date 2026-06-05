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
//
// Raster-first (Phase 1+): albedo / roughness / metalness are now also
// sampled here and written to the G-buffer so the deferred shading pass can
// light the surface analytically. Sampling matches closest_hit.rchit exactly
// (albedo.rgb, roughness from .g, metalness from .b, per-channel uvTransforms,
// hardware sRGB decode on the albedo view) so RasterFirst and ReferencePT
// agree. Fragment-shader derivatives give correct mip selection for free —
// raygen needs the lodBias attachment because RT shaders have none, but here
// plain texture() is correct.

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 currVPjittered;
    mat4 currVPunjittered;
    mat4 prevVP;
    vec4 jitter;          // .xy = curr clip-space sub-pixel jitter
    vec4 prevJitter;      // .xy = prev clip-space sub-pixel jitter
} cam;

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

// Attachment 0: world-space normal (rgba16f). .xyz = n*0.5+0.5 encoded world
// normal, .w = linear roughness (raster-first deferred pass reads it; raygen
// samples only .xyz). rgba16f is necessary for ocean wave normals — rgba8
// loses too much precision on the FFT-driven detail.
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

// Attachment 3: material UV + LOD bias (rgba16f). .rg = UV (chit normally
// interpolates from triangle vertices; we precompute here so raygen's hybrid
// primary skips the primary traceRayEXT). .b = log2(max(|dUV/dx|, |dUV/dy|))
// — a texture-size-independent footprint. raygen adds log2(textureSize) per
// sample to drive textureLod. Without this, raygen's `texture()` calls (RT
// shaders have no implicit derivatives) snap to mip 0 and per-frame texture
// shimmer survives TAA — which non-hybrid PT averages out via accumulated
// samples but hybrid (1-2 spp/frame) cannot.
layout(location = 3) out vec4 outUv;

// Attachment 4: albedo + metalness (rgba8 unorm). .rgb = linear base colour
// (material albedo × albedo-map, sRGB-decoded on sample), .a = metalness.
// Roughness lives in outNormal.w. Together these give the deferred shading
// pass a complete PBR surface. Linear 8-bit is the standard albedo G-buffer
// format; emissive / clearcoat / sheen stay re-sampled from MaterialDesc in
// the deferred pass rather than baked here.
layout(location = 4) out vec4 outAlbedoMetal;

void main() {
    vec3 N = normalize(vWorldNormal);

    // UV derivatives — used both for the LOD bias attachment and the normal-
    // map TBN construction below. Hoisted out of the normal-map branch
    // because dFdx/dFdy must be called from non-divergent control flow.
    const vec2 duvx = dFdx(vUv);
    const vec2 duvy = dFdy(vUv);

    // Normal map perturbation. TBN derived from screen-space derivatives:
    // (dpx, dpy) = world-space partial derivatives of position
    // (duvx, duvy) = uv derivatives.
    // Tangent T = (dpx · duvy.y - dpy · duvx.y) / det. Same construction
    // as chit's per-triangle tangent (closest_hit.rchit:1050-1086), just
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

    // ── PBR material sampling — mirrors closest_hit.rchit:705-739 so the
    // deferred (RasterFirst) shade matches the path-traced (ReferencePT) shade.
    // Per-channel transformed UVs.
    const vec2 uvAlbedo     = (m.uvTransform           * vec3(vUv, 1.0)).xy;
    const vec2 uvRoughMetal = (m.uvTransformRoughMetal * vec3(vUv, 1.0)).xy;

    // Albedo: scalar PBR colour × bound albedo map (.rgb). Albedo views are
    // VK_FORMAT_*_SRGB so texture() returns linear, exactly as in chit.
    vec3 albedoSample = vec3(1.0);
    if (m.albedoTexIndex >= 0) {
        const int ai = clamp(m.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
        albedoSample = texture(gbufAlbedoMaps[ai], uvAlbedo).rgb;
    }
    const vec3 albedo = m.albedo * albedoSample;

    // glTF packs roughness in .g and metalness in .b; threepp's roughnessMap /
    // metalnessMap usually point at the same packed texture. Multiplicative —
    // matches three.js and chit.
    //
    // MeshBasicMaterial (unlit) is flagged with material roughness < 0 (see
    // VulkanRenderer::materialFromMesh). Preserve that sentinel through the
    // G-buffer — don't sample the rough/metal maps or clamp — so the deferred
    // pass can emit the base colour unlit, matching closest_hit.rchit's
    // `roughness < 0` gate. Clamping here would turn the unlit surface into a
    // glossy one that reflects the environment.
    float roughness = m.roughness;
    float metalness = m.metalness;
    if (roughness >= 0.0) {
        if (m.roughnessTexIndex >= 0) {
            const int i = clamp(m.roughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
            roughness *= texture(gbufAlbedoMaps[i], uvRoughMetal).g;
        }
        if (m.metalnessTexIndex >= 0) {
            const int i = clamp(m.metalnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
            metalness *= texture(gbufAlbedoMaps[i], uvRoughMetal).b;
        }
        roughness = clamp(roughness, 0.04, 1.0);
        metalness = clamp(metalness, 0.0,  1.0);
    }

    // Remap [-1, 1] → [0, 1] so negative components are visible in the
    // BGRA8_UNORM debug blit (which clamps negatives to 0). raygen reverses
    // this with `n * 2 - 1` when reading the attachment. .w carries linear
    // roughness for the deferred pass.
    outNormal = vec4(N * 0.5 + 0.5, roughness);
    outAlbedoMetal = vec4(albedo, metalness);

    vec2 currNDC = vCurrClipUnjit.xy / vCurrClipUnjit.w;
    vec2 prevNDC = vPrevClip.xy      / vPrevClip.w;
    // Motion vector points from current pixel to where the surface WAS last
    // frame. Tested adding a (curr_jitter - prev_jitter) delta here; sign
    // flip didn't change behavior, so the shake source isn't TAA reproject —
    // it's the PT accumulator's reproject reading a jittered chit
    // hitWorldPos. raygen now overrides hitWorldPos to a pixel-center-
    // deterministic value (see raygen primaryWorldPosHybrid override after
    // the primary traceRayEXT). Motion vec stays jitter-free.
    vec2 motion = prevNDC - currNDC;
    outMotion = vec4(motion, 0.0, 0.0);

    // +1 so the renderpass's clear-to-0 means "sky/no draw", matching
    // raygen's Payload.hitInstanceId convention exactly.
    outIds = uvec4(vInstanceIdx + 1u, vInstanceIdx + 1u, vFlags, 0u);

    // log2 of the per-pixel UV-footprint diameter (texture-size-independent).
    // raygen turns this into a per-texture LOD via `bias + log2(textureSize.x)`.
    // Floor at -16 to keep textureLod's clamp from biting the rare
    // duvx==duvy==0 case (degenerate triangle / fully orthogonal view).
    const float fp2 = max(dot(duvx, duvx), dot(duvy, duvy));
    const float lodBias = 0.5 * log2(max(fp2, 1e-32));

    outUv = vec4(vUv, lodBias, 0.0);
}
