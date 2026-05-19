#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable

#include "vulkan_shared.h"

// Phase 9: this shader fills a struct payload (radiance leaving the hit +
// sampled bounce direction + throughput multiplier). raygen handles the
// path loop; we just shade the hit and pick where the path goes next.
//
// Direct lighting: Cook-Torrance GGX for specular, energy-conserving
// Lambert for diffuse, gated by per-light shadow rays. Specular env IBL
// is sampled directly from the equirect at the reflection direction
// (matches WGPU PT and standard raster IBL).
//
// Indirect bounce: multi-lobe BSDF importance sample (sampleBsdf) — a
// stochastic split between clearcoat, base-spec (VNDF) and base-diff
// (cosine-weighted hemisphere) lobes. The spec lobe gives metals mirror
// reflections of nearby geometry; the diff lobe's BRDF·cos/pdf collapses
// to `albedo·(1-metalness)` so metals contribute nothing to diffuse GI.

struct Payload {
    vec3 radianceDiff;
    vec3 radianceSpec;
    vec3 brdfWeight;
    vec3 nextOrigin;
    vec3 nextDir;
    uint flags;
    uint seed;
    vec3 hitWorldPos;  // world-space hit point (for primary-hit reprojection)
    uint hitInstanceId;// gl_InstanceCustomIndexEXT + 1; 0 == miss/sky/pass-through
    vec3 prevWorldPos; // per-vertex prev-frame world position (skinned mesh reproject); see raygen Payload for details
    float hitRoughness;// post-clamp surface roughness; primary-hit FC cap input
    uint inFlags;      // raygen→here: bit 0 = scatter>0 (suppress emissive output;
                       // emissive NEE on the prev shade already accounted for it)
    float hitMetalness;   // raygen uses for adaptive bounce classification
    float hitTransmission;// raygen uses for adaptive bounce classification
    float bsdfPdf;        // chit→miss: pdf of the BSDF-sampled bounce direction
    float currentIor;     // medium-stack tracking; see raygen Payload for full description
    float hitSpecFrac;    // estimated [0,1] spec-vs-diff fraction at primary; see raygen
    vec4 primaryAlbedo;   // primary-surface albedo + demod-valid flag; see raygen
    vec3 hitNormal;       // world-space shading normal at this hit; see raygen Payload
};

layout(buffer_reference, scalar) readonly buffer VertexBuf { float p[]; };
layout(buffer_reference, scalar) readonly buffer NormalBuf  { float n[]; };
layout(buffer_reference, scalar) readonly buffer IndexBuf   { uint  i[]; };
layout(buffer_reference, scalar) readonly buffer UvBuf      { float u[]; };
layout(buffer_reference, scalar) readonly buffer FoamBuf    { float f[]; };

struct GeometryDesc {
    uint64_t vertexAddress;
    uint64_t normalAddress;
    uint64_t indexAddress;
    uint64_t uvAddress;// 0 == no UV attribute
    uint64_t foamAddress;// 0 == no foam attribute (per-vertex float; written by water_displace.comp for ocean meshes)
    uint64_t prevVertexAddress;// previous frame deformed positions (skinned/displaced); == vertexAddress for static
    uint     indexed;
    uint     _pad;
};

struct DirLight {
    vec3 direction;
    vec3 color;
};
struct PointLight {
    vec3  position;
    float range;  // 0 = infinite
    vec3  color;
    float decay;
};
struct SpotLight {
    vec3  position;
    float range;
    vec3  color;
    float decay;
    vec3  direction;      // toward target (emission direction)
    float cosAngleOuter;  // cos(angle)          — hard cutoff
    float cosAngleInner;  // cos(angle*(1-pen))  — full-brightness edge
};
struct RectLight {
    vec3 position;
    vec3 halfU;   // world right  * width/2
    vec3 halfV;   // world up     * height/2
    vec3 normal;  // emission direction into scene
    vec3 color;
};

// ── Specialization constants ──────────────────────────────────────────────
// Pipeline-time feature gates. Compiled into the SPIR-V at vkCreate...; the
// optimizer DCEs the unreachable branch entirely. Host picks the right
// variant (see rtVariants_) per frame based on which features are enabled —
// no runtime branch on motionFlags, no register pressure for the dead path.
// constant_id values are stable: don't renumber, or stored caches go stale.
// Note: chit's constant_id namespace is independent of raygen's — both can
// start at 0 (per-stage, per Vulkan spec).
layout(constant_id = 0) const bool kRestirDIEnabled = false;
// Scene-feature bitmask — set at first-frame pipeline build by the host
// (currentSceneFeatures() in VulkanRenderer.cpp). Bits:
//   1 = kSceneFeatHasGlass        — scene contains glass; gates caustic
//                                    photon gather (3×3×3 grid loop).
//   2 = kSceneFeatHasClearcoat    — gates clearcoat lobe setup + sampling
//                                    (ccProb / ccWeight / ccF0 / ccAlpha).
//   4 = kSceneFeatHasIridescence  — gates evalIridescence (Belcour 2017
//                                    Fourier-spectral integration).
//   8 = kSceneFeatHasSheen        — gates Charlie NDF + Neubelt visibility
//                                    sheen contribution in NEE blocks.
layout(constant_id = 1) const uint kSceneFeatures = 0u;
const uint kSceneFeatHasGlass       = 1u;
const uint kSceneFeatHasClearcoat   = 2u;
const uint kSceneFeatHasIridescence = 4u;
const uint kSceneFeatHasSheen       = 8u;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topAS;
layout(set = 0, binding = 3, scalar) readonly buffer GeomDescBuf {
    GeometryDesc geoms[];
};
layout(set = 0, binding = 4, scalar) readonly buffer MatDescBuf {
    MaterialDesc mats[];
};
layout(set = 0, binding = 5, scalar) uniform LightsUbo {
    vec3       ambient;
    uint       dirCount;
    DirLight   dirLights[8];
    uint       pointCount;
    uint       spotCount;
    uint       rectCount;
    PointLight pointLights[8];
    SpotLight  spotLights[8];
    RectLight  rectLights[4];
} lights;
layout(set = 0, binding = 6) uniform sampler2D envTex;
// Env luminance CDF (Phase A: importance-sample bright env features). Bound
// as a 1×1 dummy with envCdfTotalSum=0 when env is solid color or default.
layout(set = 0, binding = 18) uniform sampler2D envCdfTex;  // conditional, w×h
layout(set = 0, binding = 19) uniform sampler2D envMargTex; // marginal, 1×h

// Homogeneous fog (participating media). sigmaT.xyz = per-channel extinction
// (1/world-unit), enabled = 1.0 when scene.fog is set. Mirrors the WGPU PT
// uniform layout (WgpuPathTracerTypes.hpp: fog/fogColor vec4 pair).
layout(set = 0, binding = 17) uniform FogUbo {
    vec3  sigmaT;
    float enabled;
    vec3  color;
    float anisotropy;
    float waterSurfaceY;
} fog;

// Beer-Lambert fog transmittance + fogEnabled() helper live in shade_common.glsl
// (included below, once all referenced bindings are visible).

// Bindless material albedo array. Indexed by mdesc.albedoTexIndex; slot 0 is
// the host's 1×1 white default, used implicitly via -1→0 fallback below.
layout(set = 0, binding = 8) uniform sampler2D albedoMaps[kMaxMaterialTextures];

// FFT ocean fine-cascade spatial-domain height map (RG32F, .r = height in m
// after multiplying by 1/tileSize per Tessendorf convention). 1×1 dummy when
// no DisplacedMesh is in the scene; cascade-2 view when one is present.
// Sampled on thinWalled hits at world-space XZ to perturb the shading normal
// with sub-mesh-resolution wave detail. Gated by pc.oceanFineTileSize > 0.
layout(set = 0, binding = 32) uniform sampler2D oceanFineHeight;

// World-space foam accumulator (R32F, REPEAT-sampled). Written by
// foam_world.comp each frame; we sample here at the hit's world XZ to
// shade water surfaces. Wraps with `oceanFoamTileSize`. Replaces the
// per-vertex foam buffer that used to live on the BLAS — foam is now
// world-anchored, so ocean meshes can re-tessellate without dragging
// foam values with their vertex indices. Gated by pc.oceanFoamTileSize > 0.
layout(set = 0, binding = 44) uniform sampler2D oceanFoamWorld;

// Emissive-mesh NEE: each emissive triangle is packed as 4 vec4
// (v0.xyz/area, v1.xyz/cumPower, v2.xyz/power, emission.rgb/_pad).
// The host walks the scene each frame, transforms triangle vertices to
// world space, computes per-triangle area + power = lum*area, and stores
// a running CDF in v1.w (cumPower). Total power lives in pc.emissiveTotalPower.
struct EmTri {
    vec4 v0;        // xyz = pos, w = area
    vec4 v1;        // xyz = pos, w = cumPower (running CDF of power)
    vec4 v2;        // xyz = pos, w = power (per-triangle power = lum * area)
    vec4 emission;  // xyz = emissive*intensity, w = unused
};
layout(set = 0, binding = 14, scalar) readonly buffer EmissiveTriBuf {
    EmTri emissiveTris[];
};

// Caustic photon map (photon_emit.rgen deposits, we gather here).
// Grid constants + kGatherRadius come from vulkan_shared.h.

layout(set = 0, binding = 15, std430) readonly buffer PhotonCountBuf { uint photonCounts[]; };
layout(set = 0, binding = 16, scalar) readonly buffer PhotonDataBuf  { vec3 photonData[];   };

// ReSTIR DI Stage 1b — temporal-reuse infrastructure.
//   binding 9  = PrevCameraUbo (added CLOSEST_HIT stage on host); used to
//                reproject the current-frame world-space hit point into the
//                previous frame's pixel grid for reservoir lookup.
//   binding 13 = prevGbufImage (rgba32f: prev worldPos.xyz + prev instID.w);
//                used for the depth + mesh-ID validation gate.
//   bindings 28/29 = reservoir lightPos + lightType ping-pong (rgba32f).
//   bindings 30/31 = reservoir W_sum/M/W/p_hat ping-pong (rgba16f).
// 28/30 are written this frame; 29/31 hold the prior frame's writes.
layout(set = 0, binding = 9) uniform PrevCameraUboChit {
    vec4 prevCamPosX;  // .xyz = world position, .w = projScaleX = 1/(aspect·tanHalfFovY)
    vec4 prevCamFwdY;  // .xyz = forward,        .w = projScaleY = 1/tanHalfFovY
    vec4 prevCamRgt;   // .xyz = right
    vec4 prevCamUp;    // .xyz = up
} pcam;
layout(set = 0, binding = 13, rgba32f) uniform readonly image2D prevGbufImage;
// accumImage .w = per-pixel FC; chit reads at gl_LaunchIDEXT for the GI
// primary saturation gate (skip sub-trace on converged pixels).
layout(set = 0, binding =  7, rgba32f) uniform readonly image2D accumImage;
layout(set = 0, binding = 28, rgba32f) uniform writeonly image2D resPosWrite;
layout(set = 0, binding = 29, rgba32f) uniform readonly  image2D resPosRead;
layout(set = 0, binding = 30, rgba16f) uniform writeonly image2D resWWrite;
layout(set = 0, binding = 31, rgba16f) uniform readonly  image2D resWRead;
// ReSTIR GI Stage 1b — per-pixel reservoir ping-pong (3× rgba32f pairs).
//   38/39: xs.xyz   + W_sum    — chosen sample position + RIS sum
//   40/41: ns.xyz   + M        — chosen sample normal   + running candidate count
//   42/43: Lo.rgb   + W        — chosen sample radiance + finalized RIS weight
// omegaI is re-derived per-frame as `normalize(xs - currentHitPos)` so it
// doesn't need storage (saves one pair). p_hat is also not stored — Stage 1b's
// resampling re-evaluates the target function `lum(BRDF·NdotL·Lo)` at OUR pixel
// (the whole point of temporal reuse with shifted view), and we recompute it
// freshly from xs/ns/Lo each merge. f&1 ping-pong: frame N writes 38/40/42 and
// reads 39/41/43; the descriptor sets alternate slots per frame in flight.
layout(set = 0, binding = 38, rgba32f) uniform writeonly image2D giResXsWrite;
layout(set = 0, binding = 39, rgba32f) uniform readonly  image2D giResXsRead;
layout(set = 0, binding = 40, rgba32f) uniform writeonly image2D giResNsWrite;
layout(set = 0, binding = 41, rgba32f) uniform readonly  image2D giResNsRead;
layout(set = 0, binding = 42, rgba32f) uniform writeonly image2D giResLoWrite;
layout(set = 0, binding = 43, rgba32f) uniform readonly  image2D giResLoRead;
// Primary-surface albedo is no longer written here directly. Chit fills
// payload.primaryAlbedo; raygen owns the temporal-blend write to the
// ping-pong albedo accumulator (bindings 35 write / 36 read).

// Phase 11: PMREM mip count comes via the same push-constant block used by
// raygen. .x is raygen's sampleIndex (not read here); .y is envMipCount.
layout(push_constant) uniform Pc {
    uint sampleIndex;
    uint envMipCount;
    uint _pad1;
    uint _pad2;
    uint motionFlags;       // bit 2 = scene has any glass material (gates caustic gather)
    uint emissiveCount;     // # of EmTri entries
    float emissiveTotalPower;// total CDF power (last entry's cumPower)
    uint _padSpp;           // raygen spp (unused here)
    uint envCdfWidth;       // env CDF dimensions (envCdfTotalSum > 0 to enable)
    uint envCdfHeight;
    float envCdfTotalSum;   // pdf normaliser; 0 disables env importance sampling
    float fireflyClamp;     // per-NEE luminance cap; 1e30 disables (gates never fire)
    float oceanFineTileSize;// FFT fine-cascade tile size in m; 0 = no ocean fine normal
    uint _padEdgeMsaa;      // raygen-only; reserved here so foam-tile lands at slot [14]
    float oceanFoamTileSize;// world-space foam tile size in m; 0 = no foam sampling
} pc;

hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT Payload payload;
// Shadow visibility: 0 = occluded (default), 1 = clear (set by shadow_miss).
layout(location = 1) rayPayloadEXT float shadowVisibility;
// ReSTIR GI Stage 1a — sub-payload for the indirect "candidate" trace launched
// from the primary hit. Same Payload struct as the incoming `payload` so the
// recursive chit invocation can re-use the full shading path; the sub-trace
// chit gates its behaviour on `inFlags & 8u` (set here before traceRayEXT) so
// it (a) doesn't recurse into another GI sub-trace and (b) doesn't run RIS DI
// against the temporal reservoir bound for the primary pixel. Recursion depth
// 4 (primary chit → GI sub-trace chit at xs → sub-sub-trace at y → shadow ray
// at y) is allocated at pipeline creation time; the shadow_anyhit terminates
// without re-tracing.
layout(location = 2) rayPayloadEXT Payload giSubPayload;

// Shared shading helpers (BSDF math, sampling pdfs, RNG, env/fog/photon
// helpers, iridescence, sheen). Lives in shade_common.glsl so the future
// hybrid bounce-0-on-raster path in raygen.rgen can include the same source.
// Must come AFTER the binding declarations (envTex, fog, photonCounts /
// photonData) that some of the helpers reference.
#include "shade_common.glsl"

// ───── ReSTIR DI — Stage 1a (init RIS, no temporal/spatial reuse) ─────
// One reservoir per primary-shading invocation; lives in registers and is
// not persisted across frames (no SSBO yet — Stage 1b adds temporal reuse).
//
// lightType encoding:
//   < 0      : sentinel "no candidate" / env (env is NEE'd separately,
//              not via the reservoir, matching the WGPU PT)
//   0..7     : directional[lightType]      — lightPos = direction
//   8..15    : point   [lightType - 8]     — lightPos = position
//   16..23   : spot    [lightType - 16]    — lightPos = position
//   24..27   : rect    [lightType - 24]    — lightPos = sampled point
//   1000+    : emissive[lightType - 1000]  — lightPos = sampled point
//
// Target pdf p_hat = NdotL · luminance(Le).  Le here is the emitter's
// post-attenuation radiance at the receiver (point/spot include 1/d² +
// range window + cone; rect/emTri are unattenuated emitter color). No
// BRDF in the target — keeps it cheap and roughness-independent (matches
// WGPU; a BRDF-based target produced texture-pattern bias because of the
// safeAlpha clamp at low roughness).
struct Reservoir {
    vec3  lightPos;
    float lightType;
    float W_sum;
    float M;
    float W;
    float p_hat;
};

void updateReservoir(inout Reservoir r, vec3 pos, float ltype,
                     float w, float p_hat_new, inout uint seed) {
    r.W_sum += w;
    r.M     += 1.0;
    if (urand(seed) < w / max(r.W_sum, 1e-20)) {
        r.lightPos  = pos;
        r.lightType = ltype;
        r.p_hat     = p_hat_new;
    }
}

void finalizeReservoir(inout Reservoir r) {
    r.W = r.W_sum / max(r.M * r.p_hat, 1e-20);
}

// ───── ReSTIR GI — Stage 1a (single-sample reservoir at primary) ─────
// Per Ouyang 2021 (vanilla ReSTIR GI), the indirect sample at primary is
// represented as (xs, ns, Lo, omegaI) — a virtual point sample where
//   xs      = world-space position of the first indirect hit,
//   ns      = shading normal at xs,
//   Lo      = outgoing radiance at xs (one-bounce direct in Stage 1a — full
//             path-tracer continuation from xs lives in raygen, not here),
//   omegaI  = direction from the primary surface point to xs.
//
// Stage 1a generates exactly one candidate per pixel per frame via BSDF
// sampling from the primary, traces the sub-ray, and shades xs (via the
// recursive chit invocation) to obtain Lo. The reservoir struct + helpers
// are scaffolded here so Stage 1b can add temporal reuse (persistent
// reservoir ping-pong) and Stage 1c can add spatial reuse without rewriting
// the data layout. At M=1 the contribution collapses to the classic MC
// bounce-1 estimate (= bs.weight · Lo); the variance reduction kicks in at
// M>1 when neighbours/history contribute via RIS.
struct GiReservoir {
    vec3  xs;      // sample position (world space)
    vec3  ns;      // sample normal (world space)
    vec3  Lo;      // outgoing radiance at xs
    vec3  omegaI;  // direction from primary hitPos to xs
    float W_sum;   // running Σ w_i = p_hat / proposal_pdf
    float M;       // # of candidates merged
    float W;       // = W_sum / (M · p_hat_at_chosen)
    float p_hat;   // target pdf at chosen sample
};

void updateGiReservoir(inout GiReservoir r,
                      vec3 xs_new, vec3 ns_new, vec3 Lo_new, vec3 omegaI_new,
                      float w, float p_hat_new, inout uint seed) {
    r.W_sum += w;
    r.M     += 1.0;
    if (urand(seed) < w / max(r.W_sum, 1e-20)) {
        r.xs     = xs_new;
        r.ns     = ns_new;
        r.Lo     = Lo_new;
        r.omegaI = omegaI_new;
        r.p_hat  = p_hat_new;
    }
}

void finalizeGiReservoir(inout GiReservoir r) {
    r.W = r.W_sum / max(r.M * r.p_hat, 1e-20);
}

// Evaluate the GI target integrand at this pixel for an arbitrary incoming
// direction `L` and a virtual-sample radiance `Lo`. Returns
//   F_integrand = BRDF(V, L, surface) · max(N·L, 0) · Lo
// — the full bounce-1 estimator value at this surface (no 1/pdf factored in).
// Mirrors the env-NEE BSDF eval block (cc + base-spec + base-diff, sheen
// omitted as a minor approximation on indirect surfaces) so target evaluation
// at NEIGHBOUR samples (temporal reproject / spatial neighbour taps) uses
// THIS pixel's BRDF rather than the storing pixel's — that's the whole point
// of ReSTIR target re-eval. Pass `Lo = vec3(0)` to get just BRDF·NdotL.
vec3 evalGiTarget(vec3 V, vec3 L, vec3 N, vec3 F0, vec3 albedo,
                  float roughness, float metalness, float alpha,
                  float ccProb,    float ccWeight,  float ccRough,
                  float baseScale, vec3 Lo) {
    const float NdotL = dot(N, L);
    if (NdotL <= 0.0) return vec3(0.0);
    const float NdotV = max(dot(N, V), 0.0);
    const vec3  H     = normalize(V + L);
    const float NdotH = max(dot(N, H), 0.0);
    const float VdotH = max(dot(V, H), 0.0);
    const float k     = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    const vec3  F_e   = fresnelSchlick(VdotH, F0);
    const float D_e   = distGGX(NdotH, roughness);
    const float G_e   = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
    const vec3  spec  = (D_e * G_e * F_e) / max(4.0 * NdotV * NdotL, 1e-4);
    const vec3  kd    = (vec3(1.0) - F_e) * (1.0 - metalness);
    const vec3  diff  = kd * albedo / PI + kcDiff(albedo, metalness, F0, NdotV, alpha);
    vec3 brdf = (diff + spec) * baseScale;
    if (ccWeight > 0.0) {
        const float k_cc = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
        const float D_cc = distGGX(NdotH, ccRough);
        const float G_cc = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotL, k_cc);
        brdf += vec3((D_cc * G_cc) / max(4.0 * NdotV * NdotL, 1e-4) * ccWeight);
    }
    return brdf * NdotL * Lo;
}

// Reconstruct (dir, maxDist, Le_unclamped) from a stored reservoir sample.
// Used by both temporal-reuse target-pdf re-eval (Stage 1b) and the
// visibility-test shade pass. Le here is the *unclamped* emitter radiance at
// the receiver — analytic lights include attenuation; emTri / rect are raw
// emitter color. Caller applies fireflyClamp when shading. Returns a zero
// dir for sentinel "no candidate" reservoirs (typeCode < 0).
struct LightInfo { vec3 dir; float maxDist; vec3 Le; };
LightInfo evalLightInfoForReservoir(int typeCode, vec3 lightPos, vec3 hitPos) {
    LightInfo o;
    o.dir     = vec3(0.0);
    o.maxDist = 0.0;
    o.Le      = vec3(0.0);
    if (typeCode < 0) return o;
    if (typeCode >= 1000) {
        const int eTi = typeCode - 1000;
        const vec3 toL = lightPos - hitPos;
        const float dist = length(toL);
        if (dist < 1e-12) return o;
        o.dir     = toL / dist;
        o.maxDist = dist - 1e-2;
        o.Le      = emissiveTris[eTi].emission.rgb;
        return o;
    }
    if (typeCode < 8) {
        if (uint(typeCode) >= lights.dirCount) return o;
        o.dir     = normalize(lights.dirLights[typeCode].direction);
        o.maxDist = 1e30;
        o.Le      = lights.dirLights[typeCode].color;
        return o;
    }
    if (typeCode < 16) {
        const int pi = typeCode - 8;
        if (uint(pi) >= lights.pointCount) return o;
        const vec3 toL = lights.pointLights[pi].position - hitPos;
        const float dist = length(toL);
        if (dist < 1e-12) return o;
        o.dir     = toL / dist;
        o.maxDist = dist - 1e-2;
        const float decay = lights.pointLights[pi].decay;
        float atten = 1.0 / max(pow(dist, decay), 0.01);
        const float rng = lights.pointLights[pi].range;
        if (rng > 0.0) {
            const float t  = dist / rng;
            const float t4 = t * t * t * t;
            const float ww = max(1.0 - t4, 0.0);
            atten *= ww * ww;
        }
        o.Le = lights.pointLights[pi].color * atten;
        return o;
    }
    if (typeCode < 24) {
        const int si = typeCode - 16;
        if (uint(si) >= lights.spotCount) return o;
        const vec3 toL = lights.spotLights[si].position - hitPos;
        const float dist = length(toL);
        if (dist < 1e-12) return o;
        o.dir     = toL / dist;
        o.maxDist = dist - 1e-2;
        const float spotCos   = dot(-o.dir, lights.spotLights[si].direction);
        const float spotAtten = smoothstep(lights.spotLights[si].cosAngleOuter,
                                            lights.spotLights[si].cosAngleInner, spotCos);
        const float decay = lights.spotLights[si].decay;
        float atten = 1.0 / max(pow(dist, decay), 0.01);
        const float rng = lights.spotLights[si].range;
        if (rng > 0.0) {
            const float t  = dist / rng;
            const float t4 = t * t * t * t;
            const float ww = max(1.0 - t4, 0.0);
            atten *= ww * ww;
        }
        atten *= spotAtten;
        o.Le = lights.spotLights[si].color * atten;
        return o;
    }
    // typeCode in [24,28): rect light. lightPos is the *sampled point* on
    // the rect that was selected at candidate-gen time; we re-aim the shadow
    // ray at that exact point so re-evaluating Le requires no extra sampling.
    const int ri = typeCode - 24;
    if (uint(ri) >= lights.rectCount) return o;
    const vec3 toL = lightPos - hitPos;
    const float dist = length(toL);
    if (dist < 1e-12) return o;
    o.dir     = toL / dist;
    o.maxDist = dist - 1e-2;
    o.Le      = lights.rectLights[ri].color;
    return o;
}

// Env CDF sampling (envCdfSearch / sampleEnvImportance / envImportancePdf),
// the multi-lobe BSDF sampler (sampleBsdf + BsdfSample), VNDF helpers
// (cosineHemisphere / makeTBN / smithG1 / sampleVNDF_H / sampleVNDF) all
// live in shade_common.glsl — included above. Both raygen and chit use
// matching `pc` push-constant fields (envCdfWidth/Height/TotalSum), so
// the env helpers compile in either stage.

void main() {
    const float w = 1.0 - attribs.x - attribs.y;

    const GeometryDesc gdesc = geoms[gl_InstanceCustomIndexEXT];
    const MaterialDesc mdesc = mats [gl_InstanceCustomIndexEXT];

    uvec3 idx;
    if (gdesc.indexed != 0u) {
        IndexBuf ib = IndexBuf(gdesc.indexAddress);
        idx = uvec3(ib.i[gl_PrimitiveID * 3 + 0],
                    ib.i[gl_PrimitiveID * 3 + 1],
                    ib.i[gl_PrimitiveID * 3 + 2]);
    } else {
        idx = uvec3(gl_PrimitiveID * 3 + 0,
                    gl_PrimitiveID * 3 + 1,
                    gl_PrimitiveID * 3 + 2);
    }

    NormalBuf nb = NormalBuf(gdesc.normalAddress);
    const vec3 n0 = vec3(nb.n[idx.x * 3 + 0], nb.n[idx.x * 3 + 1], nb.n[idx.x * 3 + 2]);
    const vec3 n1 = vec3(nb.n[idx.y * 3 + 0], nb.n[idx.y * 3 + 1], nb.n[idx.y * 3 + 2]);
    const vec3 n2 = vec3(nb.n[idx.z * 3 + 0], nb.n[idx.z * 3 + 1], nb.n[idx.z * 3 + 2]);

    // World-space foam coverage [0..1] (FFT-Tessendorf Jacobian < 1 →
    // folding → whitewater, plus wake-formula trails and disturbance
    // splats). Sampled from a 2D texture indexed by world XZ instead of
    // interpolated from per-vertex attributes — pins foam to world
    // coordinates so it stays put as ocean meshes re-tessellate. The
    // `foamAddress` payload bit still gates whether this surface should
    // pick up foam (set on DisplacedMesh, cleared otherwise).
    float foamCoverage = 0.0;
    if (gdesc.foamAddress != 0ul && pc.oceanFoamTileSize > 0.0) {
        const vec3 hitWorld = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
        const vec2 uv = hitWorld.xz / pc.oceanFoamTileSize;
        foamCoverage = clamp(texture(oceanFoamWorld, uv).r, 0.0, 1.0);
    }

    const vec3 nObj = normalize(w * n0 + attribs.x * n1 + attribs.y * n2);
    const vec3 Nworld = normalize(transpose(mat3(gl_WorldToObjectEXT)) * nObj);

    const vec3 V = normalize(-gl_WorldRayDirectionEXT);
    // Geometric facing from triangle winding — always correct regardless of
    // shading normals. Used for the single-sided pass-through test so that
    // morphed/deformed meshes whose blended shading normals diverge from the
    // geometric normal at grazing angles don't falsely pass through.
    const bool geoFront = gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT;
    // Shading-normal facing for lighting (smooth-normal flip on double-sided).
    const bool isFront = dot(Nworld, V) >= 0.0;
    vec3 N = isFront ? Nworld : -Nworld;

    // Single-sided material hit from the wrong side: pass the ray through
    // unchanged. Uses geometric facing (not shading normal) to avoid false
    // pass-through at grazing angles. sideMode 0 = Front (back-face hit is
    // wrong side), 1 = Back (front-face hit is wrong side), 2 = Double
    // (never wrong side — falls through to shading).
    const bool wrongSideHit = (mdesc.sideMode == 0 && !geoFront) ||
                              (mdesc.sideMode == 1 &&  geoFront);
    if (wrongSideHit && mdesc.transmission <= 0.0) {
        payload.radianceDiff  = vec3(0.0);
        payload.radianceSpec  = vec3(0.0);
        payload.brdfWeight    = vec3(1.0);
        // No advance past hitT — raygen uses a 1e-4 tmin for pass-through
        // bounces so the next ray sees a host surface sitting right behind
        // the just-hit pass-through surface (decal-on-mesh case).
        payload.nextOrigin    = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        payload.nextDir       = gl_WorldRayDirectionEXT;
        payload.flags         = 4u;
        // Tag this surface so raygen can record it as the primary if nothing
        // opaque is found later (e.g. pass-through → sky). Without this, sky-
        // facing back-faces never accumulate history and stay permanently noisy.
        payload.hitWorldPos   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        payload.prevWorldPos  = payload.hitWorldPos;// pass-through — no deformation handling needed
        payload.hitInstanceId = uint(gl_InstanceCustomIndexEXT) + 1u;
        // Pass-through: no shading, so use the material's nominal roughness as
        // a coarse-but-safe FC cap input. The next visible surface upstream
        // overrides this if it lands as the primary hit.
        payload.hitRoughness    = clamp(mdesc.roughness, 0.04, 1.0);
        payload.hitMetalness    = clamp(mdesc.metalness, 0.0, 1.0);
        payload.hitTransmission = clamp(mdesc.transmission, 0.0, 1.0);
        payload.hitSpecFrac     = 0.0;// pass-through; primary surface upstream owns this
        payload.hitNormal       = vec3(0.0);// pass-through has no shading surface
        return;
    }

    // MeshBasicMaterial: unlit early-out. Sentinel `roughness < 0` set on the
    // host (VulkanRenderer.cpp materialFromMesh). Emit base color as direct
    // radiance and terminate the path — no lighting, NEE, or bounce. Mirrors
    // WGPU's `shininess == -1` unlit gate.
    if (mdesc.roughness < 0.0) {
        vec2 unlitUv = vec2(0.0);
        if (gdesc.uvAddress != 0ul) {
            UvBuf ub = UvBuf(gdesc.uvAddress);
            const vec2 uv0 = vec2(ub.u[idx.x * 2 + 0], ub.u[idx.x * 2 + 1]);
            const vec2 uv1 = vec2(ub.u[idx.y * 2 + 0], ub.u[idx.y * 2 + 1]);
            const vec2 uv2 = vec2(ub.u[idx.z * 2 + 0], ub.u[idx.z * 2 + 1]);
            unlitUv = w * uv0 + attribs.x * uv1 + attribs.y * uv2;
        }
        vec3 albedoSample = vec3(1.0);
        if (mdesc.albedoTexIndex >= 0) {
            const int idxClamped = clamp(mdesc.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
            const vec2 uvA = (mdesc.uvTransform * vec3(unlitUv, 1.0)).xy;
            albedoSample = texture(albedoMaps[idxClamped], uvA).rgb;
        }
        const vec3 hitPosUnlit = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        // Unlit: emit base color as direct radiance. Route to diff channel
        // (unlit is view-independent by definition).
        payload.radianceDiff  = mdesc.albedo * albedoSample;
        payload.radianceSpec  = vec3(0.0);
        payload.brdfWeight    = vec3(0.0);
        payload.nextOrigin    = vec3(0.0);
        payload.nextDir       = vec3(0.0);
        payload.flags         = 1u;// terminate
        payload.hitWorldPos     = hitPosUnlit;
        payload.prevWorldPos    = hitPosUnlit;// unlit terminates path, no reproject benefit
        payload.hitInstanceId   = uint(gl_InstanceCustomIndexEXT) + 1u;
        payload.hitRoughness    = 1.0;
        payload.hitMetalness    = 0.0;
        payload.hitTransmission = 0.0;
        payload.hitSpecFrac     = 0.0;// unlit: no view-dep concern
        payload.hitNormal       = N;
        return;
    }

    // UV interpolation (only when the geometry has a uv attribute). The
    // outer fallback is vec2(0) — harmless for materials without an albedo
    // texture; the slot-0 white default would just sample (0,0) anyway.
    vec2 rawUv = vec2(0.0);
    if (gdesc.uvAddress != 0ul) {
        UvBuf ub = UvBuf(gdesc.uvAddress);
        const vec2 uv0 = vec2(ub.u[idx.x * 2 + 0], ub.u[idx.x * 2 + 1]);
        const vec2 uv1 = vec2(ub.u[idx.y * 2 + 0], ub.u[idx.y * 2 + 1]);
        const vec2 uv2 = vec2(ub.u[idx.z * 2 + 0], ub.u[idx.z * 2 + 1]);
        rawUv = w * uv0 + attribs.x * uv1 + attribs.y * uv2;

        // Tangent-space normal map (glTF convention). The TBN frame is
        // derived per-pixel from triangle position + UV deltas, so we don't
        // have to upload a tangent attribute. Skipped when the UV layout
        // is degenerate (det ≈ 0) or the tangent collapses after Gram-
        // Schmidt against N. Sampled RGB is unpacked from [0,1]→[-1,1];
        // xy are scaled by mdesc.normalScale and z is re-derived to keep
        // the perturbed normal a unit vector. Mirrored UV charts (det < 0)
        // would ideally flip B's sign — not handled here; revisit if a
        // specific asset shows inverted bumps. albedoMaps is the generic
        // bindless pool (the name is slightly stale).
        if (mdesc.normalTexIndex >= 0 && gdesc.vertexAddress != 0ul) {
            VertexBuf vb = VertexBuf(gdesc.vertexAddress);
            const vec3 p0 = vec3(vb.p[idx.x * 3 + 0], vb.p[idx.x * 3 + 1], vb.p[idx.x * 3 + 2]);
            const vec3 p1 = vec3(vb.p[idx.y * 3 + 0], vb.p[idx.y * 3 + 1], vb.p[idx.y * 3 + 2]);
            const vec3 p2 = vec3(vb.p[idx.z * 3 + 0], vb.p[idx.z * 3 + 1], vb.p[idx.z * 3 + 2]);
            const vec3 e1 = p1 - p0;
            const vec3 e2 = p2 - p0;
            const vec2 duv1 = uv1 - uv0;
            const vec2 duv2 = uv2 - uv0;
            const float det = duv1.x * duv2.y - duv2.x * duv1.y;
            if (abs(det) > 1e-8) {
                const float inv = 1.0 / det;
                const vec3 Tobj = inv * (e1 * duv2.y - e2 * duv1.y);
                vec3 Tworld = mat3(gl_ObjectToWorldEXT) * Tobj;
                Tworld = Tworld - dot(Tworld, N) * N;// Gram-Schmidt vs shading N
                const float Tlen = length(Tworld);
                if (Tlen > 1e-6) {
                    const vec3 T = Tworld / Tlen;
                    const vec3 B = cross(N, T);
                    const int nidx = clamp(mdesc.normalTexIndex, 0, int(kMaxMaterialTextures) - 1);
                    vec3 ns = texture(albedoMaps[nidx], (mdesc.uvTransformNormal * vec3(rawUv, 1.0)).xy).rgb * 2.0 - 1.0;
                    ns.xy *= mdesc.normalScale;
                    ns.z = sqrt(max(0.0, 1.0 - dot(ns.xy, ns.xy)));
                    N = normalize(T * ns.x + B * ns.y + N * ns.z);
                }
            }
        }
    }

    // FFT fine-cascade normal perturbation. Adds sub-mesh-resolution wave
    // detail derived from the same Phillips/Tessendorf system that drives
    // the geometry. The vertex normals computed in water_displace.comp use
    // a coarse 4-tap finite difference (eps ≈ tileSize0/256, so ~3.9 m for
    // a 1 km tile) that captures only the macro waves; cm-scale chop in
    // cascade 2 is lost to that smoothing. Re-sampling cascade 2 here with
    // a much smaller eps recovers it as a shading-time normal perturbation
    // — animates for free (the FFT texture is rewritten per frame) and
    // matches the wave field exactly. Replaces the static procedural normal
    // map on water surfaces. Gated on thinWalled so non-ocean surfaces are
    // unaffected.
    if (mdesc.thinWalled != 0 && pc.oceanFineTileSize > 0.0) {
        const vec3  worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        const float invTile  = 1.0 / pc.oceanFineTileSize;
        const float eps      = pc.oceanFineTileSize / 32.0;
        const float depsUV   = eps * invTile;
        const vec2  uv       = worldPos.xz * invTile;
        const float hL = texture(oceanFineHeight, uv + vec2(-depsUV, 0.0)).r * invTile;
        const float hR = texture(oceanFineHeight, uv + vec2( depsUV, 0.0)).r * invTile;
        const float hD = texture(oceanFineHeight, uv + vec2(0.0, -depsUV)).r * invTile;
        const float hU = texture(oceanFineHeight, uv + vec2(0.0,  depsUV)).r * invTile;
        const vec2  grad = vec2(hR - hL, hU - hD) / (2.0 * eps);
        // Add gradient as XZ tangent-space offset to N. Ocean macro normal is
        // close to +Y so this approximates the tangent frame; tilted patches
        // get a small directional error that's invisible at typical wave
        // slopes. Strength <1 keeps the perturbation subtle — cascade 2's
        // contribution is already partly baked into the vertex Y, so re-
        // adding the full gradient double-counts the low end. 0.5 reads as
        // "sharp chop in the highlights" without making the surface noisy.
        const float strength = 0.5;
        N = normalize(N + strength * vec3(-grad.x, 0.0, -grad.y));
    }
    // Per-channel transformed UVs — applied to rawUv with each texture's own matrix.
    const vec2 uvAlbedo         = (mdesc.uvTransform            * vec3(rawUv, 1.0)).xy;
    const vec2 uvRoughMetal     = (mdesc.uvTransformRoughMetal  * vec3(rawUv, 1.0)).xy;
    const vec2 uvEmissive       = (mdesc.uvTransformEmissive    * vec3(rawUv, 1.0)).xy;
    const vec2 uvOcclusion      = (mdesc.uvTransformOcclusion   * vec3(rawUv, 1.0)).xy;
    const vec2 uvClearcoat      = (mdesc.uvTransformClearcoat   * vec3(rawUv, 1.0)).xy;
    const vec2 uvClearcoatRough = (mdesc.uvTransformClearcoatRough * vec3(rawUv, 1.0)).xy;
    const vec2 uvTransmission   = (mdesc.uvTransformTransmission * vec3(rawUv, 1.0)).xy;

    const float NdotV = max(dot(N, V), 0.0);

    // Albedo: scalar PBR colour modulated by the bound albedo map (sRGB
    // decode is hardware-side via the VK_FORMAT_R8G8B8A8_SRGB view).
    vec3 albedoSample = vec3(1.0);
    if (mdesc.albedoTexIndex >= 0) {
        const int idxClamped = clamp(mdesc.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
        albedoSample = texture(albedoMaps[idxClamped], uvAlbedo).rgb;
    }
    vec3 albedo = mdesc.albedo * albedoSample;

    // glTF packs roughness in .g and metalness in .b; threepp's metalnessMap /
    // roughnessMap typically point at the same packed texture, so the bindless
    // cache dedupes to a single slot. Multiplicative — matches three.js.
    float roughness = mdesc.roughness;
    float metalness = mdesc.metalness;
    if (mdesc.roughnessTexIndex >= 0) {
        const int i = clamp(mdesc.roughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        roughness *= texture(albedoMaps[i], uvRoughMetal).g;
    }
    if (mdesc.metalnessTexIndex >= 0) {
        const int i = clamp(mdesc.metalnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        metalness *= texture(albedoMaps[i], uvRoughMetal).b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metalness = clamp(metalness, 0.0,  1.0);

    // Foam application: folded-surface vertices (foamCoverage > 0, set by
    // water_displace.comp via the Tessendorf Jacobian) bleach the albedo
    // toward white and push roughness toward 1.0 (fully diffuse). The
    // transmission lobe also reads `transmission` later in the shader; we
    // suppress it on heavy foam so whitecaps read as opaque whitewater
    // rather than tinted glass-foam. No-op when foamCoverage = 0.
    float foamMask = 0.0;
    if (foamCoverage > 0.0) {
        // Foam shading is two-layer:
        //   - MACRO fBm (~5 m features) gates the coverage mask so the foam
        //     pool edges break up organically rather than reading as a
        //     bilinear ramp from per-vertex interpolation.
        //   - MICRO value-noise (~0.15 m features) modulates the WITHIN-foam
        //     brightness and roughness — this is what gives dense foam its
        //     bubble look instead of flat paint. Two-tone (off-white peaks,
        //     mid-gray valleys) + a small specular catch on peaks via the
        //     roughness lerp produces the look the reference image has.
        // A slow time drift (driven by sampleIndex so it works without a
        // separate time uniform) prevents dense foam from reading as a
        // static texture during long stationary camera shots.
        const vec3 hpFoam = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        const vec2 wxz = hpFoam.xz;
        const float t = float(pc.sampleIndex) * (1.0 / 60.0);   // seconds-ish
        const vec2 drift = vec2(0.42, 0.71) * t * 0.18;          // m/s scale

        // MACRO mask noise. fBm at low frequency (~5 m features) gives
        // soft, organic pool edges. Scaled to subtract a sizable chunk
        // from coverage so foam appears only where coverage clearly wins.
        const vec2 wxzMacro = wxz * 0.18 + drift;
        const float macroN  = fbm4(wxzMacro);
        foamMask = smoothstep(0.05, 0.70, foamCoverage - macroN * 0.45);

        // MICRO bubble detail. Two octaves of value noise at small scale
        // (~0.15 m). Animated with anti-phase drift so the bubble pattern
        // shifts independently of the pool boundary.
        const vec2 wxzMicro = wxz * 6.5 - drift * 0.4;
        const float micro   = vnoise21(wxzMicro) * 0.65 +
                              vnoise21(wxzMicro * 2.3) * 0.35;

        // Two-tone foam colour: peaks light (off-white with slight blue),
        // valleys mid-gray. The luminance variation alone is responsible
        // for ~80% of the "looks like foam" effect.
        const vec3 foamHi = vec3(0.97, 0.99, 1.00);
        const vec3 foamLo = vec3(0.55, 0.62, 0.66);
        const vec3 foamCol = mix(foamLo, foamHi, micro);

        albedo = mix(albedo, foamCol, foamMask);
        // Roughness: 0.45..1.0 range. Bubble peaks (high micro) get the
        // lower end → a faint specular catch. Pure 1.0 reads chalky and
        // kills the bright glints visible in real foam.
        const float foamRough = mix(1.0, 0.45, micro);
        roughness = mix(roughness, foamRough, foamMask);
    }

    vec3 F0 = mix(vec3(0.04) * mdesc.specularIntensity * mdesc.specularColor, albedo, metalness);
    // Thin-film iridescence layer (KHR_materials_iridescence). Modulates F0
    // with wavelength-dependent interference; lobe shape (GGX) is unchanged,
    // only the Fresnel base shifts per channel. Skipped when factor == 0
    // so non-iridescent materials pay nothing beyond the branch.
    // Spec-constant gate: when the scene has no iridescent material, the
    // entire evalIridescence call (Belcour 2017 spectral Fourier integration)
    // is DCEd from the SPV.
    if ((kSceneFeatures & kSceneFeatHasIridescence) != 0u && mdesc.iridescence > 0.0) {
        const vec3 irid = evalIridescence(1.0, mdesc.iridescenceIOR, NdotV,
                                          mdesc.iridescenceThicknessNm, F0);
        F0 = mix(F0, irid, mdesc.iridescence);
    }
    const float k         = (roughness + 1.0) * (roughness + 1.0) / 8.0;

    // World-space hit point, used as origin for the shadow rays. Offset
    // along the geometric normal to avoid self-intersection (acne).
    const vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    uint seed = payload.seed;
    const float pSpec = mix(0.5, 0.98, metalness);
    const float alpha = roughness * roughness;

    // === Clearcoat layer params ===
    // ccF0 fixed at 0.04 (dielectric IOR ≈ 1.5 — three.js / WGPU PT convention).
    // ccWeight = clearcoat·F_cc(N·V) both attenuates the base layer (energy
    // conservation: base receives 1−ccWeight) and weights the clearcoat GGX
    // lobe in eval/sampling. ccProb floors the cc sampling rate at 0.15·cc so
    // face-on viewing (where ccFresnel ≈ 0.04) doesn't starve the cc lobe of
    // samples. Mutually exclusive with transmission via the early-return below.
    // Out of scope for v1: separate clearcoatNormalMap — clearcoat uses the
    // shading normal (post base normal-map), matching three.js raster.
    // Spec-constant gate: clearcoat-free scene → ccScalar/ccRough collapse
    // to the no-clearcoat constants, propagating through ccWeight → ccProb
    // → baseScale below. Downstream `if (ccWeight > 0.0)` / `if (ccProb > 0.0)`
    // branches at the env-NEE and bounce sites get DCEd.
    float ccScalar;
    float ccRough;
    if ((kSceneFeatures & kSceneFeatHasClearcoat) != 0u) {
        ccScalar = mdesc.clearcoat;
        ccRough  = mdesc.clearcoatRoughness;
        if (mdesc.clearcoatTexIndex >= 0) {
            const int i = clamp(mdesc.clearcoatTexIndex, 0, int(kMaxMaterialTextures) - 1);
            ccScalar *= texture(albedoMaps[i], uvClearcoat).r;
        }
        if (mdesc.clearcoatRoughnessTexIndex >= 0) {
            const int i = clamp(mdesc.clearcoatRoughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
            ccRough *= texture(albedoMaps[i], uvClearcoatRough).g;
        }
    } else {
        ccScalar = 0.0;
        ccRough  = 1.0;
    }
    ccScalar = clamp(ccScalar, 0.0, 1.0);
    ccRough  = clamp(ccRough,  0.04, 1.0);
    const float ccF0       = 0.04;
    const float ccFresnel  = ccF0 + (1.0 - ccF0) * pow(1.0 - NdotV, 5.0);
    const float ccWeight   = ccScalar * ccFresnel;
    const float ccAlpha    = ccRough * ccRough;
    const float ccProb     = max(ccWeight, 0.15 * ccScalar);
    const float baseScale  = 1.0 - ccWeight;
    // Inverse of the base-branch probability for the stochastic 3-way split
    // below; clamped to avoid blowups when ccProb → 1.
    const float invBaseProb = 1.0 / max(1.0 - ccProb, 1e-6);

    // BLEND-mode (alphaCutoff < 0 sentinel) is now handled stochastically inside
    // the any-hit shader: when the coin says "transparent", the BVH skips the
    // candidate and continues past it. By the time we reach closest_hit on a
    // BLEND material, the texel rolled "opaque" for this sample, so we shade
    // the surface normally. The retrace approach used previously sat fireflies
    // on top of decal-on-mesh setups (LeePerrySmith head decal); the in-BVH
    // rejection mirrors WGPU PT's testTriangle() and converges cleanly.

    // === Transmission lobe ===
    // Russian-roulette gate by mdesc.transmission: with probability `transmission`
    // this hit acts as a glass interface (Schlick-Fresnel weighted reflect /
    // refract) and skips direct lighting + env NEE entirely. The path continues
    // with the chosen bounce direction. Matches the WGPU PT pattern: no
    // 1/transmission inverse-prob scaling, so `transmission` doubles as a
    // stylised reflect-vs-transmit blend factor (artist control), not a physical
    // mixing weight. Dispersion is sampled per-channel below; thin-wall
    // thickness proxy is wired through `mdesc.thickness` in the BL block.
    //
    // The outgoing payload sets bit 2 ("NEE skipped at this hit") so raygen
    // doesn't half-weight the env on the next bounce-into-miss — the MIS
    // partner (NEE shadow ray) wasn't fired here, so the env at the next miss
    // needs full weight to stay unbiased.
    //
    // glTF KHR_materials_transmission: scalar `transmission` is multiplied by
    // the transmissionMap red channel (linear texture, no sRGB decode).
    float transmission = mdesc.transmission;
    if (mdesc.transmissionTexIndex >= 0) {
        const int i = clamp(mdesc.transmissionTexIndex, 0, int(kMaxMaterialTextures) - 1);
        transmission *= texture(albedoMaps[i], uvTransmission).r;
    }
    // Foam suppresses transmission so whitecaps read as opaque whitewater
    // rather than tinted glass. Keyed off foamMask (the speckled visible
    // foam) rather than raw foamCoverage, so opacity matches the same
    // pattern the eye sees on the surface.
    transmission *= (1.0 - foamMask);
    if (transmission > 0.0 && urand(seed) < transmission) {
        const float ior_base = max(mdesc.ior, 1.0);
        const vec3  I        = gl_WorldRayDirectionEXT;

        // Sample a GGX microfacet normal so roughness scatters the refracted
        // ray. α=0 (smooth glass) degenerates to H=N (mirror). Fresnel and
        // reflect/refract all use H rather than N.
        const vec2  u2   = vec2(urand(seed), urand(seed));
        const vec3  H    = sampleVNDF_H(V, N, alpha, u2);
        const float cosH = max(dot(V, H), 0.0);

        // Thin-shell BSDF: explicit per-material opt-in via
        // MaterialWithThickness::thinWalled. Examples: FFT-displaced ocean
        // plane, sunglasses lens, leaf, single sheet of glass. Both faces
        // are treated as "entering" — refract uses eta=1/ior on both sides
        // — and the ray's medium isn't tracked through the surface (it
        // "exits" instantly at the same point, conceptually).
        const bool isThinShell = mdesc.thinWalled != 0;

        // Medium tracking — fixes camera-inside-closed-glass (e.g. windshield
        // from cabin). Geometric isFront alone would misclassify a back-face
        // hit as "exiting glass" when the ray was actually still in air. We
        // instead determine entry/exit from the ray's current medium IOR
        // carried in the payload: if we're already in this material's
        // medium → exiting; otherwise → entering. For thin shells, force
        // entering-on-both-faces and don't propagate the medium change.
        const float currentIor = max(payload.currentIor, 1.0);
        float targetIor;
        bool isEntering;
        if (isThinShell) {
            targetIor   = ior_base;
            isEntering  = true;
        } else if (abs(currentIor - ior_base) < 1e-3) {
            targetIor   = 1.0;        // exit to air (default outside medium)
            isEntering  = false;
        } else {
            targetIor   = ior_base;
            isEntering  = true;
        }

        // Schlick Fresnel at the microfacet half-vector. Uses base IOR so the
        // reflect/refract branch decision is achromatic — chromatic Fresnel
        // adds visible noise without much benefit at typical dispersion values.
        // Exiting-side path uses the transmitted-side cosine so TIR raises
        // F→1 smoothly.
        const float r0    = pow((1.0 - ior_base) / (1.0 + ior_base), 2.0);
        const float sin2H = max(0.0, 1.0 - cosH * cosH);
        const float cosSchlick = isEntering
                ? cosH
                : sqrt(max(0.0, 1.0 - ior_base * ior_base * sin2H));
        const float F = r0 + (1.0 - r0) * pow(1.0 - cosSchlick, 5.0);

        // === Reflect / refract — two strategies depending on geometry ===
        // For THIN shells (isThinShell=true: ocean plane, sunglasses lens) we
        // use a deterministic split: reflect lobe is evaluated analytically as
        // F · env(reflectDir) · visibility via a shadow ray, refract continues
        // the path with weight (1−F)·glassTint. Variance from the Fresnel pick
        // drops to zero — major denoiser-off win on smooth water.
        //
        // For CLOSED glass (isThinShell=false: spheres, goblets, vases) we use
        // the original stochastic split — pick reflect with prob F, refract
        // with prob (1−F), single ray continues. The shadow-ray approximation
        // only captures one straight-line through the glass to env, so for
        // concave geometry where a ray TIRs and bounces multiple times inside
        // the glass before exiting, multi-bounce energy is lost (manifests as
        // black bands on goblet stems, etc.). The stochastic path traverses
        // every internal bounce naturally and recovers that energy.
        //
        // Dispersion (KHR_materials_dispersion) stays stochastic per
        // wavelength regardless of mode — three-channel sampling, ×3 boost.
        float ior = ior_base;
        vec3 channelMask = vec3(1.0);
        if (mdesc.dispersion > 0.0) {
            const vec3  lambda    = vec3(0.6563, 0.5500, 0.4861);
            const float refInvSq  = 1.0 / (0.5893 * 0.5893);
            const uint  ch        = uint(urand(seed) * 3.0) % 3u;
            const float invSq     = 1.0 / (lambda[ch] * lambda[ch]);
            const float B         = (ior_base - 1.0) * mdesc.dispersion / 38.2;
            ior = ior_base + B * (invSq - refInvSq);
            channelMask = vec3(0.0);
            channelMask[ch] = 3.0;
        }
        // eta = currentIor / targetIor; for entering it reduces to (1/ior),
        // for exiting to ior. Dispersion adjusts `ior` per channel above; the
        // medium-tracking state itself uses the achromatic ior_base.
        const float eta = isEntering ? (1.0 / ior) : ior;
        const vec3 refr = refract(I, H, eta);
        const bool tir = (dot(refr, refr) < 1e-6);

        // Glass tint (used by both modes). Wraps Beer-Lambert in either the
        // thin-shell proxy (per-crossing thickness) or the closed-mesh actual-
        // distance branch.
        const float cosOut = !tir ? abs(dot(normalize(refr), H)) : 1.0;
        const float G1out  = smithG1(cosOut, alpha);
        vec3 tintBase;
        if (mdesc.ior <= 1.01) {
            const float albedoLum = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
            tintBase = mix(vec3(1.0), albedo, smoothstep(0.0, 0.1, albedoLum));
        } else {
            tintBase = albedo;
        }
        vec3 glassTint = tintBase * G1out;
        if (mdesc.attenuationDistance > 0.0) {
            if (isThinShell) {
                // Thin-shell proxy — use the user-supplied `thickness` as
                // in-medium distance. Applied at every entry crossing.
                glassTint *= pow(max(mdesc.attenuationColor, vec3(1e-6)),
                                 vec3(mdesc.thickness / mdesc.attenuationDistance));
            } else if (!isEntering) {
                // Closed-mesh actual ray distance through the medium —
                // matches the original (pre-branch) behaviour. An earlier
                // attempt added a `thickness` fallback when gl_HitTEXT < 1e-2
                // (mirrored from WGPU); it misfired on thin-walled closed
                // glass (goblet walls < 1 cm) by replacing the genuine 5 mm
                // ray distance with an unrelated asset-thickness value, and
                // over-darkened the glass into solid blue. Keep it simple.
                glassTint *= pow(max(mdesc.attenuationColor, vec3(1e-6)),
                                 vec3(gl_HitTEXT / mdesc.attenuationDistance));
            }
        }

        vec3 reflectContrib = vec3(0.0);
        vec3 wDir    = vec3(0.0);
        vec3 wOrigin = vec3(0.0);
        vec3 tWeight = vec3(0.0);
        bool terminate = false;
        bool wasReflect = false;

        if (isThinShell) {
            // ── Deterministic split (variance reduction for thin shells) ──
            const vec3 reflectDir    = reflect(I, H);
            const vec3 reflectOrigin = hitPos + N * 1e-3;
            shadowVisibility = 1.0;
            traceRayEXT(topAS,
                        gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT |
                        gl_RayFlagsNoOpaqueEXT,
                        0xff, 1, 0, 1,
                        reflectOrigin, 0.0, reflectDir, 1e30, 1);
            vec3 reflectEnv = vec3(0.0);
            if (shadowVisibility > 0.0) {
                reflectEnv = sampleEquirect(reflectDir);
            }
            reflectContrib = F * reflectEnv * shadowVisibility;

            if (!tir) {
                wDir    = normalize(refr);
                wOrigin = hitPos - N * 1e-3;
                tWeight = (1.0 - F) * glassTint * channelMask / (eta * eta);
            } else {
                // TIR — Schlick gave F=1 already, full reflection captured by
                // reflectContrib. Terminate path.
                terminate = true;
            }
        } else {
            // ── Stochastic split (original — multi-bounce intact for closed glass) ──
            if (urand(seed) < F) {
                wDir       = reflect(I, H);
                wOrigin    = hitPos + N * 1e-3;
                tWeight    = vec3(1.0);
                wasReflect = true;
            } else if (tir) {
                // TIR — fall back to mirror reflect.
                wDir       = reflect(I, H);
                wOrigin    = hitPos + N * 1e-3;
                tWeight    = vec3(1.0);
                wasReflect = true;
            } else {
                wDir    = normalize(refr);
                wOrigin = hitPos - N * 1e-3;
                tWeight = glassTint * channelMask / (eta * eta);
                // Closed-mesh refract — update the ray's current medium so
                // the next bounce knows where it is (entering glass: medium
                // becomes glass; exiting: medium becomes air). Reflect /
                // TIR-fallback paths leave currentIor untouched (ray stays
                // in the same medium).
                payload.currentIor = targetIor;
            }
        }

        vec3 emissiveOut = mdesc.emissive * mdesc.emissiveIntensity;
        if (mdesc.emissiveTexIndex >= 0) {
            const int ei = clamp(mdesc.emissiveTexIndex, 0, int(kMaxMaterialTextures) - 1);
            emissiveOut *= texture(albedoMaps[ei], uvEmissive).rgb;
        }
        const float emLum0 = dot(emissiveOut, vec3(0.2126, 0.7152, 0.0722));
        if (emLum0 > pc.fireflyClamp) emissiveOut *= pc.fireflyClamp / emLum0;
        if ((payload.inFlags & 1u) != 0u) emissiveOut = vec3(0.0);
        // For thin-shell mode reflectContrib was captured analytically; for
        // stochastic mode it stays vec3(0) and the reflect lobe's contribution
        // arrives via path traversal. raygen adds throughput·radiance to the
        // total each hit and multiplies throughput by brdfWeight.
        // Glass reflect: route to spec channel (view-dependent); emissive
        // to diff. Both will need refinement but glass already has its own
        // FC behavior via isGlassP gate in raygen.
        payload.radianceDiff  = emissiveOut;
        payload.radianceSpec  = reflectContrib;
        payload.brdfWeight    = tWeight;
        payload.nextOrigin    = wOrigin;
        payload.nextDir       = wDir;
        // flags=1 terminates path (only thin-shell + TIR right now);
        // flags=4 continues. Stochastic mode never terminates here (TIR
        // falls back to reflect) so the path always continues.
        payload.flags         = terminate ? 1u : 4u;
        payload.seed          = seed;
        // Primary-tag policy for transmission hits. With deterministic split,
        // every hit captures both reflect and refract contributions, so the
        // decision is purely material-based:
        //   ior > 1.01: real glass — tag glass as primary. Reproject anchors
        //               on the glass surface, which gives stable sky/env
        //               reflections under camera motion (the fix the
        //               vulkan-pt branch was carrying — original symptom:
        //               glass tint disappearing during movement).
        //   ior ≈ 1.0:  alpha-blend / stochastic pass-through. Refraction
        //               returns the incident direction unchanged → the ray
        //               continues to the surface behind, which IS what
        //               should anchor reproject. Skip the primary tag so
        //               the next non-zero hit wins.
        const bool tagPrimary = (ior_base > 1.01);
        if (tagPrimary) {
            payload.hitWorldPos     = hitPos;
            payload.prevWorldPos    = hitPos;// transmission front face: no deformation handling
            payload.hitInstanceId   = uint(gl_InstanceCustomIndexEXT) + 1u;
            payload.hitRoughness    = roughness;
            payload.hitMetalness    = metalness;
            payload.hitTransmission = mdesc.transmission;
            payload.hitSpecFrac     = 0.0;// glass — view-dep handled separately via isGlassP gate
            payload.hitNormal       = N;
        } else {
            payload.hitWorldPos     = vec3(0.0);
            payload.prevWorldPos    = vec3(0.0);
            payload.hitInstanceId   = 0u;
            payload.hitRoughness    = 1.0;
            payload.hitMetalness    = 0.0;
            payload.hitTransmission = 0.0;
            payload.hitSpecFrac     = 0.0;
            payload.hitNormal       = vec3(0.0);
        }
        return;
    }

    // Sum direct contribution from each scene-driven directional light.
    // Physical lights (three.js useLegacyLights = false): light.color
    // already encodes intensity, no 1/PI cancel. Each light is gated by a
    // shadow ray; closest_hit is skipped on those rays so only the shadow
    // miss handler matters. Linear HDR radiance is returned to raygen,
    // which handles the sRGB encode.
    // KHR_materials_sheen energy conservation: sheen absorbs some energy from the
    // base BRDF. Scale factor is 1 - max(sheenColor) * IBL_sheen(NdotV, roughness).
    // Spec-constant gate: sheen-free scene → hasSheen is statically false.
    // The `if (hasSheen)` branches at every NEE call site DCE to nothing.
    const bool hasSheen = (kSceneFeatures & kSceneFeatHasSheen) != 0u &&
                          mdesc.sheenRoughness > 0.0 &&
                          any(greaterThan(mdesc.sheenColor, vec3(0.0)));
    float sheenScaling = 1.0;
    if (hasSheen) {
        const float sheenMax = max(mdesc.sheenColor.r, max(mdesc.sheenColor.g, mdesc.sheenColor.b));
        sheenScaling = 1.0 - sheenMax * IBLSheenBRDF(NdotV, mdesc.sheenRoughness);
    }

    vec3 lit = vec3(0.0);

    // ReSTIR DI: at primary on opaque-ish surfaces, replace per-light shadow
    // rays with RIS over a single chosen sample (1 shadow ray total instead
    // of up to 28 + emissive). Bounces and transmissive primaries fall
    // through to classic NEE below. Env NEE remains separate (matches WGPU
    // PT — env CDF importance sampling carries its own MIS pair). Stage 1a
    // = init RIS only; 1b adds temporal reuse via a per-pixel reservoir
    // SSBO; 1c adds spatial reuse via random neighbour taps from the same
    // SSBO. Master gate is pc.motionFlags bit 4 (renderer.setRestirDIEnabled).
    // The `inFlags & 8u` exclusion suppresses RIS DI when this chit invocation
    // is a ReSTIR GI sub-trace — the reservoir at this LaunchID belongs to
    // the primary pixel (the camera ray's primary hit), and the sub-trace's
    // xs is a different surface entirely. Letting RIS run here would mix two
    // unrelated reservoir streams into the same per-pixel slot.
    // BSDF_ONLY mode (inFlags bit 5 = 32u): raygen sets this for spp samples
    // s>=1 at primary so the chit only does material setup + BSDF sample +
    // payload finalize, skipping NEE / RIS / GI / caustic gather. The result
    // is direct-lighting independent of bs.dir is computed once per pixel
    // (s=0) and the cached value is injected by raygen for s>=1; only the
    // bounce direction (and resulting indirect path) varies per spp. Saves
    // (spp-1) × (NEE shadow ray + RIS shadow ray + GI sub-trace + Stage 2
    // sub-sub-trace + photon gather) per primary-hit pixel.
    const bool bsdfOnlyMode  = (payload.inFlags & 32u) != 0u;
    // Pulled from a specialization constant at pipeline-compile time, not
    // a per-frame push-constant read. The dead branch is DCEd from the SPV.
    const bool restirOn      = kRestirDIEnabled;
    const bool useRISPrimary = restirOn
                            && ((payload.inFlags & 1u) == 0u)
                            && ((payload.inFlags & 8u) == 0u)
                            && (mdesc.transmission < 0.05)
                            && !bsdfOnlyMode;

    if (useRISPrimary) {
        Reservoir r;
        r.lightPos  = vec3(0.0);
        r.lightType = -2.0;
        r.W_sum     = 0.0;
        r.M         = 0.0;
        r.W         = 0.0;
        r.p_hat     = 0.0;

        // Joint analytic-light proposal density: pick uniformly over all
        // enabled analytic slots in the lights UBO. p_source_per_candidate
        // = 1 / analyticN, so w_i = ρ_i / p_source = ρ_i × analyticN. Without
        // this factor, M grows by `analyticN` (one per slot, including
        // intensity-zero ones) but W_sum only collects ρ from the live
        // ones — the resulting W = W_sum / (M·p_hat) is darkened by the
        // ratio of live-to-total slots. Matches WGPU's `p_source_a =
        // 1/lcount` (pt_primary_shade1.wgsl:168). Floor at 1.0 since when
        // analyticN == 0 the for-loops below don't execute anyway.
        const float analyticN = max(float(lights.dirCount + lights.pointCount
                                         + lights.spotCount + lights.rectCount), 1.0);

        // Directional candidates — delta lights.
        for (uint i = 0u; i < lights.dirCount; ++i) {
            const vec3 L = normalize(lights.dirLights[i].direction);
            const float NdotL = max(dot(N, L), 0.0);
            const vec3 Le = lights.dirLights[i].color;
            const float p_hat = NdotL * lum3(Le);
            updateReservoir(r, L, float(i), p_hat * analyticN, p_hat, seed);
        }

        // Point candidates — delta with 1/d^decay + range window in Le.
        for (uint i = 0u; i < lights.pointCount; ++i) {
            const vec3 toL_p = lights.pointLights[i].position - hitPos;
            const float dist = length(toL_p);
            if (dist < 1e-4) { r.M += 1.0; continue; }
            const vec3 L = toL_p / dist;
            const float NdotL = max(dot(N, L), 0.0);
            const float decay = lights.pointLights[i].decay;
            float atten = 1.0 / max(pow(dist, decay), 0.01);
            const float rng = lights.pointLights[i].range;
            if (rng > 0.0) {
                const float t  = dist / rng;
                const float t4 = t * t * t * t;
                const float ww = max(1.0 - t4, 0.0);
                atten *= ww * ww;
            }
            const vec3 Le = lights.pointLights[i].color * atten;
            const float p_hat = NdotL * lum3(Le);
            updateReservoir(r, lights.pointLights[i].position, float(8u + i),
                            p_hat * analyticN, p_hat, seed);
        }

        // Spot candidates — point + cone.
        for (uint i = 0u; i < lights.spotCount; ++i) {
            const vec3 toL_s = lights.spotLights[i].position - hitPos;
            const float dist = length(toL_s);
            if (dist < 1e-4) { r.M += 1.0; continue; }
            const vec3 L = toL_s / dist;
            const float NdotL = max(dot(N, L), 0.0);
            const float spotCos = dot(-L, lights.spotLights[i].direction);
            const float spotAtten = smoothstep(lights.spotLights[i].cosAngleOuter,
                                                lights.spotLights[i].cosAngleInner, spotCos);
            const float decay = lights.spotLights[i].decay;
            float atten = 1.0 / max(pow(dist, decay), 0.01);
            const float rng = lights.spotLights[i].range;
            if (rng > 0.0) {
                const float t  = dist / rng;
                const float t4 = t * t * t * t;
                const float ww = max(1.0 - t4, 0.0);
                atten *= ww * ww;
            }
            atten *= spotAtten;
            const vec3 Le = lights.spotLights[i].color * atten;
            const float p_hat = NdotL * lum3(Le);
            updateReservoir(r, lights.spotLights[i].position, float(16u + i),
                            p_hat * analyticN, p_hat, seed);
        }

        // Rect candidates — area light, sample uniform-area; combined
        // proposal = (1/analyticN) × (1/area_omega), so w = (p_hat / p_omega) × analyticN.
        for (uint i = 0u; i < lights.rectCount; ++i) {
            const float ru = urand(seed) * 2.0 - 1.0;
            const float rv = urand(seed) * 2.0 - 1.0;
            const vec3 samplePos = lights.rectLights[i].position
                                  + ru * lights.rectLights[i].halfU
                                  + rv * lights.rectLights[i].halfV;
            const vec3 toL_r = samplePos - hitPos;
            const float dist = length(toL_r);
            if (dist < 1e-4) { r.M += 1.0; continue; }
            const vec3 L = toL_r / dist;
            const float NdotL = max(dot(N, L), 0.0);
            const float cosEmitter = max(dot(-L, lights.rectLights[i].normal), 0.0);
            if (NdotL <= 0.0 || cosEmitter <= 0.0) { r.M += 1.0; continue; }
            const float area = length(cross(lights.rectLights[i].halfU,
                                             lights.rectLights[i].halfV)) * 4.0;
            const float p_omega = (dist * dist) / max(area * cosEmitter, 1e-12);
            const vec3 Le = lights.rectLights[i].color;
            const float p_hat = NdotL * lum3(Le);
            const float w_i = (p_hat / max(p_omega, 1e-20)) * analyticN;
            updateReservoir(r, samplePos, float(24u + i), w_i, p_hat, seed);
        }

        // Emissive candidates — 4 samples from per-frame power CDF.
        if (pc.emissiveCount > 0u && pc.emissiveTotalPower > 0.0) {
            // Tightest binary-search iteration count: ceil(log2(emissiveCount)).
            // The old 32-cap supported up to 2^32 entries — far beyond what any
            // real scene needs (typical scenes: 1-10000 emissive tris). findMSB
            // returns the highest set bit; max(...,1u) sidesteps findMSB(0)=-1
            // when emissiveCount==1 (1 iter is enough; the lo>=hi break inside
            // short-circuits before the first body run).
            const int emIters = findMSB(max(pc.emissiveCount - 1u, 1u)) + 1;
            for (int s = 0; s < 4; ++s) {
                const float xi = urand(seed) * pc.emissiveTotalPower;
                uint lo = 0u;
                uint hi = pc.emissiveCount - 1u;
                for (int it = 0; it < emIters; ++it) {
                    if (lo >= hi) break;
                    const uint mid = (lo + hi) >> 1u;
                    if (emissiveTris[mid].v1.w < xi) lo = mid + 1u;
                    else                              hi = mid;
                }
                const EmTri t = emissiveTris[lo];
                const float r1 = urand(seed);
                const float r2 = urand(seed);
                const float su1 = sqrt(r1);
                const float bA = 1.0 - su1;
                const float bB = su1 * (1.0 - r2);
                const float bC = su1 * r2;
                const vec3 lp = bA * t.v0.xyz + bB * t.v1.xyz + bC * t.v2.xyz;
                const vec3 toL_e = lp - hitPos;
                const float dist2 = dot(toL_e, toL_e);
                const float dist = sqrt(max(dist2, 1e-20));
                if (dist <= 1e-4) { r.M += 1.0; continue; }
                const vec3 L = toL_e / dist;
                const float NdotL = dot(N, L);
                const vec3 ge1 = t.v1.xyz - t.v0.xyz;
                const vec3 ge2 = t.v2.xyz - t.v0.xyz;
                const vec3 lnRaw = cross(ge1, ge2);
                const float lnLen = length(lnRaw);
                if (NdotL <= 0.01 || lnLen < 1e-20 || t.v0.w <= 1e-20 || t.v2.w <= 0.0) {
                    r.M += 1.0; continue;
                }
                const vec3 lN = lnRaw / lnLen;
                const float cosLight = abs(dot(-L, lN));
                if (cosLight <= 0.01) { r.M += 1.0; continue; }
                const float pickPdf = t.v2.w / pc.emissiveTotalPower;
                const float p_omega = pickPdf * dist2 / (t.v0.w * cosLight);
                const vec3 Le = t.emission.rgb;
                const float p_hat = NdotL * lum3(Le);
                const float w_i = p_hat / max(p_omega, 1e-20);
                updateReservoir(r, lp, 1000.0 + float(lo), w_i, p_hat, seed);
            }
        }

        finalizeReservoir(r);

        // ── Stage 1b: temporal reuse ──
        // Reproject world hit into prev frame's pixel grid; if mesh-ID + depth
        // gates pass, merge with prev reservoir. WGPU PT validates with
        // normal+depth via gBuf.xyz/.w; Vulkan's prevGbufImage carries
        // worldPos+meshID (same physical buffer raygen uses for accum
        // reproject), so we use mesh-ID + |Δdist|/dist instead of normal —
        // gives a comparable validity gate without a second prev-gbuf texture.
        // M-clamp: 5 during camera motion (flush stale history quickly), 20
        // when static — matches WGPU's restirParams.y values.
        {
            const ivec2 sz = imageSize(prevGbufImage);
            const vec3 toPrevCam = hitPos - pcam.prevCamPosX.xyz;
            const float prevZ    = dot(toPrevCam, pcam.prevCamFwdY.xyz);
            if (prevZ > 1e-3 && sz.x > 0 && sz.y > 0) {
                const float ndcX = dot(toPrevCam, pcam.prevCamRgt.xyz) * pcam.prevCamPosX.w / prevZ;
                const float ndcY = dot(toPrevCam, pcam.prevCamUp.xyz)  * pcam.prevCamFwdY.w / prevZ;
                const float prevU = (ndcX * 0.5 + 0.5) * float(sz.x);
                const float prevV = (0.5 - ndcY * 0.5) * float(sz.y);
                const ivec2 prevPx = ivec2(int(floor(prevU)), int(floor(prevV)));
                if (prevPx.x >= 0 && prevPx.x < sz.x &&
                    prevPx.y >= 0 && prevPx.y < sz.y) {
                    const vec4 pgb = imageLoad(prevGbufImage, prevPx);
                    const uint prevMeshId = uint(pgb.w);
                    const uint curMeshId  = uint(gl_InstanceCustomIndexEXT) + 1u;
                    if (prevMeshId != 0u && prevMeshId == curMeshId) {
                        const float curDist  = length(toPrevCam);
                        const float prevDist = length(pgb.xyz - pcam.prevCamPosX.xyz);
                        if (abs(curDist - prevDist) / max(curDist, 1e-6) < 0.1) {
                            const vec4 prevPosT = imageLoad(resPosRead, prevPx);
                            const vec4 prevWdat = imageLoad(resWRead,   prevPx);
                            const float mClamp = ((pc.motionFlags & 2u) != 0u) ? 5.0 : 20.0;
                            Reservoir rPrev;
                            rPrev.lightPos  = prevPosT.xyz;
                            rPrev.lightType = prevPosT.w;
                            rPrev.W_sum     = prevWdat.x;
                            rPrev.M         = min(prevWdat.y, mClamp);
                            rPrev.W         = prevWdat.z;
                            rPrev.p_hat     = prevWdat.w;
                            if (rPrev.W > 0.0 && rPrev.M > 0.0 && rPrev.lightType >= -0.5) {
                                const int prevType = int(rPrev.lightType);
                                const LightInfo li = evalLightInfoForReservoir(prevType, rPrev.lightPos, hitPos);
                                const float NdotL_prev = max(dot(N, li.dir), 0.0);
                                const float p_hat_prev = NdotL_prev * lum3(li.Le);
                                if (p_hat_prev > 0.0) {
                                    const float w_prev = p_hat_prev * rPrev.M * rPrev.W;
                                    r.W_sum += w_prev;
                                    r.M     += rPrev.M;
                                    if (urand(seed) < w_prev / max(r.W_sum, 1e-20)) {
                                        r.lightPos  = rPrev.lightPos;
                                        r.lightType = rPrev.lightType;
                                        r.p_hat     = p_hat_prev;
                                    }
                                    finalizeReservoir(r);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Snapshot the reservoir BEFORE spatial reuse — this is what gets
        // persisted for next frame's temporal merge, matching WGPU
        // pt_primary_shade1.wgsl:289. Persisting post-spatial would inflate
        // next-frame temporal M with spatial contributions that don't
        // generalise across reprojection, biasing temporal weights overly
        // conservative (and producing visible temporal instability). The
        // post-spatial reservoir is still used below for THIS frame's
        // visibility/shading — spatial improves the current estimate, but
        // only the pre-spatial state propagates forward.
        Reservoir rPreSpatial = r;
        // Cap the persisted W with the same firefly bound used at shading
        // time (line 1711 below). Without this, a stale high-W reservoir
        // perpetuates via temporal merge each frame — w_prev includes
        // rPrev.W as a factor, so a firefly sample picked once keeps
        // dominating that pixel's W_sum every frame, producing static
        // bright specks that never average out. Capping at persistence
        // breaks the feedback loop while leaving the current frame's
        // unclamped post-spatial r alone for shading.
        if (pc.fireflyClamp < 1e20) rPreSpatial.W = min(rPreSpatial.W, 5.0);

        // ── Stage 1c: spatial reuse ──
        // Random neighbour taps from prev-frame reservoir buffer. Validation
        // gate is mesh-ID + depth (same prevGbufImage temporal uses; we don't
        // bind a prev-normal texture, so the cone-of-normals gate WGPU has is
        // approximated by per-mesh + depth. Curved surfaces get more permissive
        // acceptance, but the recomputed p_hat at our normal keeps the merge
        // unbiased — a "wasted" tap with low p_hat barely shifts the reservoir.
        // M-cap of 4 per neighbour matches WGPU; loop terminates early once
        // total M reaches 20 (the static-frame target). spMax: 5 static, 2
        // moving (matches WGPU's restirParams cam-moving fallback).
        {
            const ivec2 sz = imageSize(prevGbufImage);
            if (sz.x > 0 && sz.y > 0) {
                const bool camMoving = (pc.motionFlags & 2u) != 0u;
                const uint spMax     = camMoving ? 2u : 5u;
                const float mTarget  = 20.0;
                const uint curMeshId = uint(gl_InstanceCustomIndexEXT) + 1u;
                const float curDistC = length(hitPos - pcam.prevCamPosX.xyz);
                for (uint sp = 0u; sp < spMax; ++sp) {
                    if (r.M >= mTarget) break;
                    const float spAngle = urand(seed) * TWO_PI;
                    const float spR     = sqrt(urand(seed)) * 20.0;
                    const ivec2 spOff   = ivec2(int(spR * cos(spAngle)),
                                                int(spR * sin(spAngle)));
                    if (spOff.x == 0 && spOff.y == 0) continue;
                    const ivec2 spPx = clamp(ivec2(gl_LaunchIDEXT.xy) + spOff,
                                             ivec2(0), ivec2(sz - ivec2(1)));
                    const vec4 spGbuf   = imageLoad(prevGbufImage, spPx);
                    const uint spMeshId = uint(spGbuf.w);
                    if (spMeshId == 0u || spMeshId != curMeshId) continue;
                    const float spDist = length(spGbuf.xyz - pcam.prevCamPosX.xyz);
                    if (abs(spDist - curDistC) / max(curDistC, 1e-3) > 0.1) continue;
                    const vec4 spPosT = imageLoad(resPosRead, spPx);
                    const vec4 spWdat = imageLoad(resWRead,   spPx);
                    Reservoir rSp;
                    rSp.lightPos  = spPosT.xyz;
                    rSp.lightType = spPosT.w;
                    rSp.W_sum     = spWdat.x;
                    rSp.M         = min(spWdat.y, 4.0);
                    rSp.W         = spWdat.z;
                    rSp.p_hat     = spWdat.w;
                    if (rSp.W <= 0.0 || rSp.M <= 0.0 || rSp.lightType < -0.5) continue;
                    const int spType = int(rSp.lightType);
                    const LightInfo li = evalLightInfoForReservoir(spType, rSp.lightPos, hitPos);
                    const float NdotL_sp = max(dot(N, li.dir), 0.0);
                    const float p_hat_sp = NdotL_sp * lum3(li.Le);
                    if (p_hat_sp > 0.0) {
                        const float w_sp = p_hat_sp * rSp.M * rSp.W;
                        r.W_sum += w_sp;
                        r.M     += rSp.M;
                        if (urand(seed) < w_sp / max(r.W_sum, 1e-20)) {
                            r.lightPos  = rSp.lightPos;
                            r.lightType = rSp.lightType;
                            r.p_hat     = p_hat_sp;
                        }
                    }
                }
                finalizeReservoir(r);
            }
        }

        // Persist this frame's PRE-SPATIAL reservoir for next frame's
        // temporal merge (rPreSpatial snapshotted above the spatial block).
        // Matches WGPU pt_primary_shade1.wgsl:368-373. NaN-guard W (W = W_sum
        // / (M * p_hat) can produce NaN when M or p_hat are zero through
        // floating-point underflow despite the max() in finalizeReservoir).
        // Done unconditionally inside the RIS branch — including W=0 cases —
        // so disocclusions get cleanly zeroed history rather than picking up
        // stale data from two frames ago. Bounce / transmissive primaries
        // skip this branch entirely; their pixel's prev reservoir lingers
        // until validation rejects it.
        const float rWpersist = isnan(rPreSpatial.W) ? 0.0 : rPreSpatial.W;
        imageStore(resPosWrite, ivec2(gl_LaunchIDEXT.xy),
                   vec4(rPreSpatial.lightPos, rPreSpatial.lightType));
        imageStore(resWWrite,   ivec2(gl_LaunchIDEXT.xy),
                   vec4(rPreSpatial.W_sum, rPreSpatial.M, rWpersist, rPreSpatial.p_hat));

        // Cap W when firefly clamping is enabled (matches WGPU's `if
        // emissiveInfo.z < 1e20 → W = min(W,5)`); pc.fireflyClamp uses 1e30
        // as the disabled sentinel. Applied AFTER the writeback so the
        // persisted W is unclamped (matches WGPU — clamp is a shading-time
        // safety net, not a structural property of the reservoir).
        if (pc.fireflyClamp < 1e20) r.W = min(r.W, 5.0);

        // ── Visibility test + shade at chosen sample ──
        if (r.W > 0.0 && r.p_hat > 0.0 && r.lightType >= -0.5) {
            const int typeCode = int(r.lightType);
            const LightInfo li = evalLightInfoForReservoir(typeCode, r.lightPos, hitPos);
            vec3 lDir       = li.dir;
            float lMaxDist  = li.maxDist;
            vec3 Le         = li.Le;
            // Per-emitter firefly clamp (analytic Le already includes
            // attenuation, so clamp on the post-attenuation luminance).
            if (typeCode >= 1000) {
                const float emLum = lum3(Le);
                if (emLum > pc.fireflyClamp) Le *= pc.fireflyClamp / emLum;
            }

            const float NdotL_c = max(dot(N, lDir), 0.0);
            if (NdotL_c > 0.0) {
                shadowVisibility = 1.0;
                traceRayEXT(topAS,
                            gl_RayFlagsTerminateOnFirstHitEXT |
                            gl_RayFlagsSkipClosestHitShaderEXT |
                            gl_RayFlagsNoOpaqueEXT,
                            0xff, 1, 0, 1,
                            hitPos + N * 1e-3, 0.0, lDir, lMaxDist, 1);
                if (shadowVisibility > 0.0) {
                    // BRDF eval at chosen direction (mirrors per-light eval below).
                    const vec3 H = normalize(V + lDir);
                    const float NdotH = max(dot(N, H), 0.0);
                    const float VdotH = max(dot(V, H), 0.0);
                    const vec3 F = fresnelSchlick(VdotH, F0);
                    const float D = distGGX(NdotH, roughness);
                    const float G = geomSmithG1(NdotV, k) * geomSmithG1(NdotL_c, k);
                    const vec3  spec_c = (D * G * F) / max(4.0 * NdotV * NdotL_c, 1e-4);
                    const vec3  kd_c   = (vec3(1.0) - F) * (1.0 - metalness);
                    const vec3  diff_c = kd_c * albedo / PI + kcDiff(albedo, metalness, F0, NdotV, alpha);
                    vec3 perChosen = (diff_c + spec_c) * baseScale * sheenScaling;
                    if (hasSheen) {
                        perChosen += mdesc.sheenColor * D_Charlie(NdotH, mdesc.sheenRoughness)
                                                      * V_Neubelt(NdotV, NdotL_c);
                    }
                    if (ccWeight > 0.0) {
                        const float k_cc = (ccRough + 1.0) * (ccRough + 1.0) / 8.0;
                        const float D_cc = distGGX(NdotH, ccRough);
                        const float G_cc = geomSmithG1(NdotV, k_cc) * geomSmithG1(NdotL_c, k_cc);
                        perChosen += vec3((D_cc * G_cc) / max(4.0 * NdotV * NdotL_c, 1e-4) * ccWeight);
                    }
                    vec3 contrib = perChosen * NdotL_c * Le * shadowVisibility * r.W;
                    if (fogEnabled() && lMaxDist < 1e20) {
                        contrib *= fogTransmittance(lMaxDist);
                    }
                    // MIS wLight for emissive-triangle samples. Pairs with the
                    // BSDF-side MIS at bounce-into-emissive (see the chit's
                    // post-emissive-eval block) so the two halves sum to 1.0
                    // (balance heuristic). Without this, DI's emissive shade
                    // contributes the full NEE estimator (BRDF·cos·Le/p_omega)
                    // while the bounce-into-emissive side ALSO contributes its
                    // share (wBSDF · Le) — net over-count by wBSDF. Analytic
                    // delta lights (typeCode < 1000) skip MIS because the BSDF
                    // sampler can't hit a measure-zero direction; no double-
                    // count possible there.
                    if (typeCode >= 1000) {
                        const int eTi = typeCode - 1000;
                        const EmTri tChosen = emissiveTris[eTi];
                        const vec3 ge1c = tChosen.v1.xyz - tChosen.v0.xyz;
                        const vec3 ge2c = tChosen.v2.xyz - tChosen.v0.xyz;
                        const vec3 lnRawC = cross(ge1c, ge2c);
                        const float lnLenC = length(lnRawC);
                        if (lnLenC > 1e-20 && tChosen.v0.w > 1e-20 && tChosen.v2.w > 0.0) {
                            const vec3  lNc      = lnRawC / lnLenC;
                            const float cosLight = max(abs(dot(-lDir, lNc)), 1e-4);
                            // dist² recovered from li.maxDist (= true dist - 1e-2).
                            const float dist     = lMaxDist + 1e-2;
                            const float dist2    = dist * dist;
                            const float pickPdf  = tChosen.v2.w / pc.emissiveTotalPower;
                            const float pdfOmega = pickPdf * dist2 / (tChosen.v0.w * cosLight);
                            const float pdfBsdfNee = brdfPdf3(V, lDir, N, roughness, metalness, ccProb, ccRough);
                            const float wLight     = pdfOmega / max(pdfOmega + pdfBsdfNee, 1e-8);
                            contrib *= wLight;
                        }
                    }
                    const float cLum = lum3(contrib);
                    if (cLum > pc.fireflyClamp) contrib *= pc.fireflyClamp / cLum;
                    lit += contrib;
                }
            }
        }
    } else if (!bsdfOnlyMode) {
    // Per-light shadow rays for analytic lights (dir/point/spot/rect) + a
    // power-CDF-picked emissive triangle. See analyticNeeOpaque() in
    // shade_common.glsl — verbatim extraction of what lived inline here.
    // Future raygen gbuf-shade path calls the same fn.
    lit += analyticNeeOpaque(V, N, hitPos,
                             F0, albedo,
                             roughness, metalness, alpha,
                             NdotV, k,
                             baseScale,
                             sheenScaling, hasSheen, mdesc.sheenColor, mdesc.sheenRoughness,
                             ccWeight, ccRough,
                             seed);
    } // end useRISPrimary else (classic NEE branch)


    // === Env NEE + MIS ===
    //   • envCdfTotalSum > 0  → importance-sample the env by luminance.
    //   • envCdfTotalSum <= 0 → fallback to BSDF-sampled env NEE with 0.5 MIS.
    // Both branches with post-multiply firefly clamp. See envNeeOpaque() in
    // shade_common.glsl for details; it's a verbatim extraction of what used
    // to live inline here. Future raygen gbuf-shade path will call the same fn.
    lit += envNeeOpaque(V, N, hitPos,
                        F0, albedo,
                        roughness, metalness, alpha,
                        NdotV, k,
                        baseScale, invBaseProb, pSpec,
                        ccProb, ccRough, ccWeight, ccAlpha,
                        seed);

    // Flat ambient irradiance — only the diffuse lobe receives it (metals
    // have no diffuse). No PI cancel under physical lights. Scaled by the
    // base-layer fraction so a 100%-clearcoat surface only shows the cc
    // contribution (which is dir-light + env NEE only — no flat ambient
    // term for clearcoat). Occlusion map (.r channel) scales ambient.
    float ao = 1.0;
    if (mdesc.occlusionTexIndex >= 0) {
        const int oi = clamp(mdesc.occlusionTexIndex, 0, int(kMaxMaterialTextures) - 1);
        ao = texture(albedoMaps[oi], uvOcclusion).r;
    }
    const vec3 ambient = albedo * (1.0 - metalness) * lights.ambient * baseScale * ao;

    // Phase 9 v2: probabilistic spec/diffuse lobe selection so polished
    // metals reflect nearby geometry, not just the env probe. p_spec mirrors
    // the WGPU PT (mix(0.5, 0.98, metalness)). The selected lobe's BRDF·cos
    // is divided by its sampling pdf and by p_spec / (1-p_spec) so the
    // estimator stays unbiased.
    //
    //   spec branch — VNDF half-vector → reflect → wi.
    //     BRDF·cos / vndfPdf collapses to F * G1(L);
    //     final weight = F * G1(L) / p_spec.
    //   diff branch — cosine-weighted hemisphere.
    //     BRDF·cos / cosPdf for Lambert collapses to diffuseAlbedo;
    //     final weight = diffuseAlbedo / (1 - p_spec).
    //
    // Spherical-cap VNDF (Dupuy 2023) guarantees above-horizon wi; the
    // valid-flag check below is retained for numerical edge cases only.
    // Same sampler the env-NEE fallback uses, so pdfs match for MIS.
    const BsdfSample bs = sampleBsdf(V, N, F0, albedo, alpha, ccAlpha,
                                     metalness, NdotV, pSpec, ccProb,
                                     ccWeight, baseScale, invBaseProb, seed);
    const vec3 bounceDir  = bs.dir;
    const vec3 brdfWeight = bs.weight;
    uint       pathFlags  = bs.valid ? 0u : 1u;// numerical edge case → terminate

    vec3 emissiveOut = mdesc.emissive * mdesc.emissiveIntensity;
    if (mdesc.emissiveTexIndex >= 0) {
        const int ei = clamp(mdesc.emissiveTexIndex, 0, int(kMaxMaterialTextures) - 1);
        emissiveOut *= texture(albedoMaps[ei], uvEmissive).rgb;
    }
    const float emLum1 = dot(emissiveOut, vec3(0.2126, 0.7152, 0.0722));
    if (emLum1 > pc.fireflyClamp) emissiveOut *= pc.fireflyClamp / emLum1;
    // BSDF-side MIS weight for emissive surfaces hit by bounce rays.
    // Replaces the older crude suppression (`emissiveOut = vec3(0)` when
    // prev shade ran NEE), which biased every bounce-into-emissive event
    // toward zero contribution regardless of how unlikely NEE was to have
    // picked that exact direction. Standard balance heuristic:
    //   wBSDF = pdfBSDF / (pdfBSDF + pdfOmega_NEE)
    // where `pdfBSDF` is the BSDF pdf at the bouncing direction (set by
    // the prev chit into payload.bsdfPdf, same field miss.rmiss already
    // uses for env MIS), and `pdfOmega_NEE` is the solid-angle pdf with
    // which prev chit's emissive NEE would have sampled THIS exact hit.
    //
    // pdfOmega derivation — note T.area cancels between the pickPdf and
    // the area-to-omega Jacobian:
    //   pickPdf      = T.emLum · T.area / totalPower
    //   areaToOmega  = dist² / (T.area · cosLight)
    //   pdfOmega     = T.emLum · dist² / (totalPower · cosLight)
    // We approximate `T.emLum` (the CDF-recorded triangle emission
    // luminance) with `emLum1` from this hit's actual material/texture
    // eval. For meshes with emission textures this differs slightly from
    // the CDF-stored emission (host stores material-level lum, not per-
    // texel) — a few % bias on emission-textured meshes; negligible on
    // typical flat-emitter geometry (light strips, area panels).
    //
    // cosLight uses the GEOMETRIC face normal (cross product of edge
    // vectors transformed to world space), matching the NEE side's
    // `abs(dot(-toL, lN))` — using N (shading / normal-mapped) here
    // would introduce a small bias whenever the shading normal deviates
    // from the face.
    //
    // Gates: `inFlags & 1u` (prev shade ran NEE), `pc.emissiveCount > 0u`
    // (scene has emissive geometry — mirror raygen's inFlags gate),
    // `pc.emissiveTotalPower > 0` (CDF denominator), `emLum1 > 0`
    // (skip the vertex-fetch + cross product when this surface is non-
    // emissive — MIS would be a no-op via 0 emission anyway).
    if ((payload.inFlags & 1u) != 0u && pc.emissiveCount > 0u &&
        pc.emissiveTotalPower > 0.0 && emLum1 > 0.0) {
        VertexBuf vbMis = VertexBuf(gdesc.vertexAddress);
        const vec3 v0_mis = vec3(vbMis.p[idx.x * 3 + 0], vbMis.p[idx.x * 3 + 1], vbMis.p[idx.x * 3 + 2]);
        const vec3 v1_mis = vec3(vbMis.p[idx.y * 3 + 0], vbMis.p[idx.y * 3 + 1], vbMis.p[idx.y * 3 + 2]);
        const vec3 v2_mis = vec3(vbMis.p[idx.z * 3 + 0], vbMis.p[idx.z * 3 + 1], vbMis.p[idx.z * 3 + 2]);
        const vec3 nFace_obj   = cross(v1_mis - v0_mis, v2_mis - v0_mis);
        const vec3 nFace_world = normalize(transpose(mat3(gl_WorldToObjectEXT)) * nFace_obj);
        const float cosLight   = max(abs(dot(nFace_world, gl_WorldRayDirectionEXT)), 1e-4);
        const float pdfOmega   = emLum1 * gl_HitTEXT * gl_HitTEXT / (pc.emissiveTotalPower * cosLight);
        const float pdfBsdfPrev = max(payload.bsdfPdf, 0.0);
        const float wBSDF       = pdfBsdfPrev / max(pdfBsdfPrev + pdfOmega, 1e-8);
        emissiveOut *= wBSDF;
    }
    // kSceneFeatHasGlass is baked into the SPV at pipeline-compile time;
    // when the host detected no glass in the scene, the entire gatherCaustics
    // call (and everything it transitively pulls in) is DCEd out.
    if ((kSceneFeatures & kSceneFeatHasGlass) != 0u && !bsdfOnlyMode)
        lit += gatherCaustics(hitPos, N, albedo, roughness);

    // ── ReSTIR GI Stage 2 — bounce-2 inside Lo (full-path-Lo, capped at 2) ──
    // When this chit is itself a GI sub-trace (inFlags bit 3 set) AND the
    // recursion stop-bit (bit 4) is NOT set, fire ONE recursive sub-sub-trace
    // in our BSDF-sampled direction and fold the result into `lit`. The
    // sub-sub-trace runs with inFlags = 8u|16u = 24u so it executes the same
    // direct-light shading at y but does NOT recurse further (one bounce
    // deep enough for >90% of indirect energy on diffuse-dominant scenes;
    // captures the GI's bounce-2 contribution unbiasedly inside Lo, which
    // means primary chit's GI consumed branch can terminate the path
    // without losing bounce-2). Recursion budget: primary chit (1) → GI
    // sub-trace at xs (2, this chit) → sub-sub-trace at y (3) → shadow ray
    // at y (4). Allocated at pipeline creation as maxPipelineRayRecursionDepth=4.
    //
    // Bounces 3+ at GI primaries are dropped (primary terminates after GI
    // consumed). For typical scenes this is small; if the bias becomes
    // visible (e.g. corner-cube cavity with strong inter-reflection), the
    // next Stage 2 commit would extend the recursion cap or restore raygen
    // continuation from y's outbound direction (needs propagating y's
    // nextOrigin/Dir up through the sub-trace's payload — bigger plumbing).
    // Stage 2 fires at the GI sub-trace's hit. Skip on hits where the bounce-2
    // contribution is either redundant (own NEE/Le dominates) or noise-prone
    // (narrow-lobe specular + delta refract). Preserves the variance-reduction
    // win on diffuse-rough surfaces (the typical case) while cutting one ray
    // per pixel in the noisy/redundant cases.
    const bool stage2Eligible = roughness > 0.30
                             && metalness < 0.7
                             && mdesc.transmission < 0.05
                             && emLum1 < 0.5;
    if ((payload.inFlags & 8u) != 0u && (payload.inFlags & 16u) == 0u
        && bs.valid && stage2Eligible) {
        giSubPayload.radianceDiff    = vec3(0.0);
        giSubPayload.radianceSpec    = vec3(0.0);
        giSubPayload.brdfWeight      = vec3(0.0);
        giSubPayload.nextOrigin      = vec3(0.0);
        giSubPayload.nextDir         = vec3(0.0);
        giSubPayload.flags           = 2u;// MIS halve env on miss
        giSubPayload.seed            = seed;
        giSubPayload.hitWorldPos     = vec3(0.0);
        giSubPayload.hitInstanceId   = 0u;
        giSubPayload.prevWorldPos    = vec3(0.0);
        giSubPayload.hitRoughness    = 1.0;
        // Same `8u` sub-trace gate + bit 4 (16u) recursion stop bit, plus
        // bit 0 to match the classic-path emissive suppression at bounce-2.
        giSubPayload.inFlags         = (8u | 16u) | ((pc.emissiveCount > 0u) ? 1u : 0u);
        giSubPayload.hitMetalness    = 0.0;
        giSubPayload.hitTransmission = 0.0;
        giSubPayload.bsdfPdf         = brdfPdf3(V, bs.dir, N, roughness, metalness, ccProb, ccRough);
        giSubPayload.currentIor      = 1.0;
        giSubPayload.hitSpecFrac     = 0.0;
        giSubPayload.primaryAlbedo   = vec4(0.0);
        giSubPayload.hitNormal       = vec3(0.0);

        traceRayEXT(topAS, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0,
                    hitPos + N * 1e-3, 0.001, bs.dir, 10000.0, 2);

        seed = giSubPayload.seed;

        // Lo at y (direct light + emission + ambient — sub-sub-trace ran
        // classic shading at y, miss already MIS-halved env if it hit sky).
        const vec3 Lo_y = giSubPayload.radianceDiff + giSubPayload.radianceSpec;
        // bs.weight here is the throughput multiplier for going from xs to y
        // via the BSDF lobe at xs (= BRDF_xs · cos / q_lobe). bs.weight · Lo_y
        // is the unbiased single-sample MC estimator for the second-bounce
        // contribution at xs. Firefly clamp the product — same `< 1e20`
        // sentinel pattern as the env NEE post-multiply clamp.
        vec3 contribY = bs.weight * Lo_y;
        if (pc.fireflyClamp < 1e20) {
            const float l = lum3(contribY);
            if (l > pc.fireflyClamp) contribY *= pc.fireflyClamp / l;
        }
        lit += contribY;
    }

    // ── ReSTIR GI Stage 1b — single-sample reservoir with temporal reuse ──
    // Stage 1a established the chit-recursive sub-trace + in-register reservoir.
    // Stage 1b adds:
    //   • persistent reservoir ping-pong (3× rgba32f images on bindings 38-43)
    //   • temporal reuse via reproject-+-resample of the prev-frame reservoir
    //   • mesh-ID + depth validation gate (no prev-normal image; same approach
    //     ReSTIR DI uses on this codebase)
    //   • explicit `evalGiTarget` for re-evaluating the target function at OUR
    //     pixel when the chosen sample came from elsewhere (temporal reproject
    //     here; spatial neighbours in Stage 1c)
    //   • visibility test from primary toward the chosen sample's xs — required
    //     because the chosen sample may have been stored at a different pixel
    //     (i.e. different camera origin) where the occluder geometry may have
    //     moved or been freshly disoccluded
    //   • W clamp on the persisted slot to break the same firefly feedback
    //     loop that ReSTIR DI ran into pre-fix
    //   • contribution form switched to `F · W` (textbook RIS): F evaluated at
    //     this pixel's BRDF in the direction of the chosen sample's xs. At M=1
    //     this is unbiased single-sample MC with `q = brdfPdf3` (marginal
    //     mixture pdf) — different per-frame variance from Stage 1a's
    //     `bs.weight·Lo` (lobe-conditional MC) but identical mean.
    //
    // The continuation hand-off (next bounce after primary) still uses
    // THIS frame's sub-trace data even when the reservoir's chosen sample is
    // from prev frame. Bounce 2+ paths are sampled independently per frame;
    // mixing this-frame's continuation with prev-frame's chosen sample is a
    // controlled bias on top of an unbiased bounce-1 estimate. Full-path Lo
    // (where Lo = Le_xs + Ldirect_xs + indirect_xs_via_continuation) is the
    // Stage-2 fix; not needed for Stage 1b's variance-reduction goal.
    //
    // Eligibility gate same as Stage 1a (see comment block in original commit
    // 657e1b15). Falls through to classic bounce-1 on miss.
    const bool restirGIOn   = (pc.motionFlags & 64u) != 0u;
    // Smooth clearcoat puts a near-delta lobe into evalGiTarget's p_hat (D_cc
    // peaks like a delta at ccRough ≈ 0). RIS then concentrates samples in
    // that spike — unbiased in expectation, but per-frame variance is huge,
    // so the surface reads as a constantly-flickering bright mirror until
    // FC builds. Fall back to classic continuation for those materials; the
    // base lobes alone don't trigger the pathology.
    const bool smoothCC     = (ccScalar > 0.05) && (ccRough < 0.20);
    const bool useGIPrimary = restirGIOn
                           && bs.valid
                           && ((payload.inFlags & 1u) == 0u)
                           && ((payload.inFlags & 8u) == 0u)
                           && (mdesc.transmission < 0.05)
                           && (roughness > 0.20)
                           && (metalness < 0.5)
                           && (emLum1 < 0.5)
                           && !smoothCC
                           && !bsdfOnlyMode;

    bool giConsumed = false;
    vec3 giContrib  = vec3(0.0);
    vec3 giNextOrigin = vec3(0.0);
    vec3 giNextDir    = vec3(0.0);
    vec3 giThroughput = vec3(0.0);// brdfWeight for raygen step 1 = bs.weight · sub.brdfWeight
    bool giTerminate  = false;    // sub-trace missed surface (env / numeric edge)

    // ── Fast-path saturation gate ──
    // For static camera + fully-converged pixels, the reservoir is already
    // M-saturated; firing a fresh sub-trace + Stage 2 + RIS + temporal +
    // spatial barely shifts the chosen sample (new candidate's weight is
    // ~1/W_sum which is huge after many frames of accumulation). Skip the
    // expensive paths and shade with the persisted reservoir directly.
    // Re-persist the same data this frame so the ping-pong stays current
    // for next frame's read. The visibility test below still catches
    // dynamic occluders changing between frames.
    bool giFastPath = false;
    GiReservoir giFast;
    giFast.xs = vec3(0.0); giFast.ns = vec3(0.0); giFast.Lo = vec3(0.0); giFast.omegaI = vec3(0.0);
    giFast.W_sum = 0.0; giFast.M = 0.0; giFast.W = 0.0; giFast.p_hat = 0.0;
    if (useGIPrimary && (pc.motionFlags & 2u) == 0u) {
        const float pixelFc = imageLoad(accumImage, ivec2(gl_LaunchIDEXT.xy)).w;
        if (pixelFc > 100.0) {
            const vec4 pX = imageLoad(giResXsRead, ivec2(gl_LaunchIDEXT.xy));
            const vec4 pN = imageLoad(giResNsRead, ivec2(gl_LaunchIDEXT.xy));
            const vec4 pL = imageLoad(giResLoRead, ivec2(gl_LaunchIDEXT.xy));
            if (pN.w >= 15.0 && pL.w > 0.0 && dot(pN.xyz, pN.xyz) > 0.01) {
                giFast.xs    = pX.xyz;
                giFast.ns    = pN.xyz;
                giFast.Lo    = pL.rgb;
                giFast.W_sum = pX.w;
                giFast.M     = pN.w;
                giFast.W     = pL.w;
                giFastPath   = true;
                // Re-persist unchanged so the ping-pong write image carries
                // forward (otherwise next frame's read would lag by one).
                imageStore(giResXsWrite, ivec2(gl_LaunchIDEXT.xy), pX);
                imageStore(giResNsWrite, ivec2(gl_LaunchIDEXT.xy), pN);
                imageStore(giResLoWrite, ivec2(gl_LaunchIDEXT.xy), pL);
            }
        }
    }

    if (useGIPrimary && !giFastPath) {
        // ── Sub-trace launch (unchanged from Stage 1a) ──
        giSubPayload.radianceDiff    = vec3(0.0);
        giSubPayload.radianceSpec    = vec3(0.0);
        giSubPayload.brdfWeight      = vec3(0.0);
        giSubPayload.nextOrigin      = vec3(0.0);
        giSubPayload.nextDir         = vec3(0.0);
        giSubPayload.flags           = 2u;// halve env on miss for MIS
        giSubPayload.seed            = seed;
        giSubPayload.hitWorldPos     = vec3(0.0);
        giSubPayload.hitInstanceId   = 0u;
        giSubPayload.prevWorldPos    = vec3(0.0);
        giSubPayload.hitRoughness    = 1.0;
        // bit 3 = "GI sub-trace gate" (suppress recursive GI + RIS DI in chit).
        // bit 0 = "prev shade ran NEE; suppress emissive at this hit". Mirrors
        // raygen's classic-path bounce loop: when we trace a bounce-1 ray, the
        // chit at that hit zeros emissiveOut, assuming the prior emissive NEE
        // covered it (crude MIS approximation, but it's what the rest of the
        // codebase does — without it, GI's Lo over-counts emissive light by
        // including bs.dir-hits-emissive contributions that don't get clipped
        // against primary's emissive NEE pick, producing visibly "extra
        // light" on emissive-heavy scenes vs the classic continuation path).
        giSubPayload.inFlags         = 8u | ((pc.emissiveCount > 0u) ? 1u : 0u);
        giSubPayload.hitMetalness    = 0.0;
        giSubPayload.hitTransmission = 0.0;
        giSubPayload.bsdfPdf         = brdfPdf3(V, bs.dir, N, roughness, metalness, ccProb, ccRough);
        giSubPayload.currentIor      = 1.0;
        giSubPayload.hitSpecFrac     = 0.0;
        giSubPayload.primaryAlbedo   = vec4(0.0);
        giSubPayload.hitNormal       = vec3(0.0);

        traceRayEXT(topAS, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0,
                    hitPos + N * 1e-3, 0.001, bs.dir, 10000.0, 2);

        seed = giSubPayload.seed;

        // ── This-frame candidate ──
        // Snapshot the FIRST sub-trace hit. xs/ns of the reservoir always
        // refer to this point so the primary's BRDF(V, bs.dir) re-eval is
        // well-defined and reservoir reuse at neighbours stays meaningful
        // (omegaI = neighbour→xs is close to bs.dir for nearby pixels).
        const vec3 xs_cand        = giSubPayload.hitWorldPos;
        const vec3 ns_cand        = giSubPayload.hitNormal;
        const float firstBsdfPdf  = giSubPayload.bsdfPdf;
        const bool subHitSurface  = giSubPayload.hitInstanceId != 0u &&
                                    (giSubPayload.flags & 1u) == 0u;
        // Glass-only trigger. Earlier version OR'd in `hitSpecFrac > 0.5`
        // which also fires the loop on smooth-metal sub-trace hits — those
        // don't have an SDS path through them (light bounces off, doesn't
        // refract through), so the extra rays + Stage 2 per iteration add
        // cost + single-sample variance for no benefit. Vessel scene with
        // many metal fittings nearly doubled GI cost via this gate.
        const bool firstWasGlass  = subHitSurface &&
                                    (giSubPayload.hitTransmission > 0.05);

        // SDS path extension. When the first sub-trace lands on a
        // transmissive/specular surface, the single-hit Lo is ~0 (glass
        // chit sets up nextOrigin/Dir for raygen continuation but doesn't
        // carry refracted radiance through itself). Continue tracing
        // through up to kMaxGlassHops more refractions to find the diffuse
        // termination of the SDS chain. The accumulated chain radiance is
        // the radiance leaving the FIRST hit in the -bs.dir direction
        // (throughput-attenuated by glass along the way), which is exactly
        // what the primary's BRDF eval at xs needs.
        //
        // Recursion budget unchanged: each hop is a separate trace from
        // the primary chit, max depth still 4 (primary → sub → Stage 2
        // sub-sub → shadow). Cost: 1 extra ray per pixel where sub-trace
        // hits glass; +1 more for nested glass (e.g. sphere's back face).
        vec3 chainRadiance = giSubPayload.radianceDiff + giSubPayload.radianceSpec;
        if (firstWasGlass) {
            vec3 chainThroughput = vec3(1.0);
            const int kMaxGlassHops = 2;
            for (int hop = 0; hop < kMaxGlassHops; ++hop) {
                chainThroughput *= giSubPayload.brdfWeight;
                const float thMax = max(max(chainThroughput.r, chainThroughput.g),
                                        chainThroughput.b);
                if (thMax < 1e-4) break;
                if ((giSubPayload.flags & 1u) != 0u) break;// TIR/thin-shell terminate
                const vec3 nextO = giSubPayload.nextOrigin;
                const vec3 nextD = giSubPayload.nextDir;
                if (length(nextD) < 0.5) break;

                // Re-fire sub-trace from the glass exit point.
                giSubPayload.radianceDiff    = vec3(0.0);
                giSubPayload.radianceSpec    = vec3(0.0);
                giSubPayload.brdfWeight      = vec3(0.0);
                giSubPayload.nextOrigin      = vec3(0.0);
                giSubPayload.nextDir         = vec3(0.0);
                giSubPayload.flags           = 2u;
                giSubPayload.seed            = seed;
                giSubPayload.hitWorldPos     = vec3(0.0);
                giSubPayload.hitInstanceId   = 0u;
                giSubPayload.prevWorldPos    = vec3(0.0);
                giSubPayload.hitRoughness    = 1.0;
                giSubPayload.inFlags         = 8u | ((pc.emissiveCount > 0u) ? 1u : 0u);
                giSubPayload.hitMetalness    = 0.0;
                giSubPayload.hitTransmission = 0.0;
                // Delta refract: glass effectively samples a single direction
                // with infinite pdf. Pass a large sentinel so the next chit's
                // bounce-into-emissive MIS gives the BSDF side full weight
                // (correct for delta sampling — NEE pdf at the matched
                // direction is negligible vs the delta).
                giSubPayload.bsdfPdf         = 1e10;
                giSubPayload.currentIor      = 1.0;
                giSubPayload.hitSpecFrac     = 0.0;
                giSubPayload.primaryAlbedo   = vec4(0.0);
                giSubPayload.hitNormal       = vec3(0.0);

                traceRayEXT(topAS, gl_RayFlagsNoneEXT, 0xff, 0, 0, 0,
                            nextO, 0.001, nextD, 10000.0, 2);
                seed = giSubPayload.seed;

                chainRadiance += chainThroughput *
                                 (giSubPayload.radianceDiff + giSubPayload.radianceSpec);

                const bool curHitSurface = (giSubPayload.hitInstanceId != 0u) &&
                                           ((giSubPayload.flags & 1u) == 0u);
                if (!curHitSurface) break;// env miss → done
                const bool curWasGlass = (giSubPayload.hitTransmission > 0.05 ||
                                          giSubPayload.hitSpecFrac     > 0.5);
                if (!curWasGlass) break;// landed on diffuse → done
            }
        }
        const vec3 Lo_cand = chainRadiance;

        // ── Initial reservoir (single candidate, M=1) ──
        // RIS target = lum(F_integrand) with F = BRDF·NdotL·Lo, proposal q =
        // brdfPdf3 (marginal mixture pdf at bs.dir). At M=1, finalize gives
        // W = 1/q. Different from Stage 1a's bs.weight·Lo form (which used
        // the lobe-conditional pdf instead of the marginal); same mean,
        // slightly different per-frame variance.
        //
        // When the sub-trace missed the geometry (env hit or numeric edge),
        // the candidate has no positional xs to store. We skip the reservoir
        // path entirely — the env contribution is added at shade time via
        // `bs.weight · Lo` (classic MC bounce-1 to env), and the reservoir
        // slot is persisted as zero so next frame's temporal merge here
        // rejects the slot via the ns unit-vector check.
        GiReservoir gi;
        gi.xs = vec3(0.0); gi.ns = vec3(0.0); gi.Lo = vec3(0.0); gi.omegaI = vec3(0.0);
        gi.W_sum = 0.0; gi.M = 0.0; gi.W = 0.0; gi.p_hat = 0.0;

        if (subHitSurface) {
            const vec3 F_cand = evalGiTarget(V, bs.dir, N, F0, albedo, roughness, metalness,
                                             alpha, ccProb, ccWeight, ccRough, baseScale, Lo_cand);
            const float p_hat_cand = max(lum3(F_cand), 0.0);
            const float q_cand     = max(firstBsdfPdf, 1e-20);
            const float w_cand     = p_hat_cand / q_cand;
            updateGiReservoir(gi, xs_cand, ns_cand, Lo_cand, bs.dir,
                              w_cand, p_hat_cand, seed);

            // ── Temporal reuse — reproject + merge prev reservoir ──
            // Reproject our world hitPos to prev-frame pixel space via the
            // prevCamera UBO (same path DI uses). Validate via mesh-ID
            // identity + |Δdist|/dist < 0.1; prev-normal isn't in the gbuf
            // schema so the cone-of-normals gate WGPU/Bitterli typically use
            // is approximated by mesh-ID. Re-evaluating the target at OUR
            // pixel keeps the merge unbiased — neighbours with very different
            // normals will simply produce low p_hat_us and contribute little
            // weight to the resampled reservoir.
            const ivec2 sz = imageSize(prevGbufImage);
            const vec3  toPrevCam = hitPos - pcam.prevCamPosX.xyz;
            const float prevZ     = dot(toPrevCam, pcam.prevCamFwdY.xyz);
            if (prevZ > 1e-3 && sz.x > 0 && sz.y > 0) {
                const float ndcX = dot(toPrevCam, pcam.prevCamRgt.xyz) * pcam.prevCamPosX.w / prevZ;
                const float ndcY = dot(toPrevCam, pcam.prevCamUp.xyz)  * pcam.prevCamFwdY.w / prevZ;
                const float prevU = (ndcX * 0.5 + 0.5) * float(sz.x);
                const float prevV = (0.5 - ndcY * 0.5) * float(sz.y);
                const ivec2 prevPx = ivec2(int(floor(prevU)), int(floor(prevV)));
                if (prevPx.x >= 0 && prevPx.x < sz.x &&
                    prevPx.y >= 0 && prevPx.y < sz.y) {
                    const vec4 pgb = imageLoad(prevGbufImage, prevPx);
                    const uint prevMeshId = uint(pgb.w);
                    const uint curMeshId  = uint(gl_InstanceCustomIndexEXT) + 1u;
                    if (prevMeshId != 0u && prevMeshId == curMeshId) {
                        const float curDist  = length(toPrevCam);
                        const float prevDist = length(pgb.xyz - pcam.prevCamPosX.xyz);
                        if (abs(curDist - prevDist) / max(curDist, 1e-6) < 0.1) {
                            // Validation passed — load prev reservoir.
                            const vec4 prevXs = imageLoad(giResXsRead, prevPx);
                            const vec4 prevNs = imageLoad(giResNsRead, prevPx);
                            const vec4 prevLo = imageLoad(giResLoRead, prevPx);
                            const vec3 prevXsP = prevXs.xyz;
                            const vec3 prevNsP = prevNs.xyz;
                            const vec3 prevLoP = prevLo.rgb;
                            const float prevW_sum = prevXs.w;
                            // M-cap: tighter on camera motion (flush stale
                            // samples fast → less smearing on view shifts);
                            // larger when static (accumulate longer history
                            // for stronger variance reduction).
                            const float mClamp = ((pc.motionFlags & 2u) != 0u) ? 5.0 : 20.0;
                            const float prevM = min(prevNs.w, mClamp);
                            const float prevW = prevLo.w;
                            // Validity: non-zero W + M, and stored ns is a
                            // unit vector (rejects images zero-cleared by host
                            // at first frame and any pixel that never wrote).
                            if (prevW > 0.0 && prevM > 0.0 && dot(prevNsP, prevNsP) > 0.01) {
                                const vec3 omegaI_prev = prevXsP - hitPos;
                                const float dist_prev  = length(omegaI_prev);
                                if (dist_prev > 1e-3) {
                                    const vec3 L_prev = omegaI_prev / dist_prev;
                                    // Re-eval F + p_hat at OUR pixel using the
                                    // stored sample's xs / Lo. This is the
                                    // textbook ReSTIR target re-eval.
                                    const vec3 F_prev = evalGiTarget(V, L_prev, N, F0, albedo,
                                                                       roughness, metalness, alpha,
                                                                       ccProb, ccWeight, ccRough,
                                                                       baseScale, prevLoP);
                                    const float p_hat_prev = max(lum3(F_prev), 0.0);
                                    if (p_hat_prev > 0.0) {
                                        // Canonical RIS temporal merge:
                                        //   w_prev = p_hat_us · M_prev · W_prev
                                        const float w_prev = p_hat_prev * prevM * prevW;
                                        gi.W_sum += w_prev;
                                        gi.M     += prevM;
                                        if (urand(seed) < w_prev / max(gi.W_sum, 1e-20)) {
                                            gi.xs     = prevXsP;
                                            gi.ns     = prevNsP;
                                            gi.Lo     = prevLoP;
                                            gi.omegaI = L_prev;
                                            gi.p_hat  = p_hat_prev;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            finalizeGiReservoir(gi);
        }

        // ── Snapshot pre-spatial state for persistence ──
        // The persisted slot is the PRE-spatial reservoir (matches DI's
        // pattern at chit:1650, and Ouyang 2021 §4.3 — feeding post-spatial
        // back into next frame's temporal merge inflates M with neighbour
        // contributions that don't generalise across reprojection, biasing
        // the temporal weights). Spatial reuse modifies `gi` AFTER this
        // snapshot; `giPreSpatial` is what we write, `gi` is what we shade.
        // We persist unconditionally (zero when subHitSurface == false) so
        // next frame's reproject at this pixel always sees a clean slot;
        // the validity gate over there (ns is a unit vector) rejects zeros.
        GiReservoir giPreSpatial = gi;
        if (pc.fireflyClamp < 1e20) giPreSpatial.W = min(giPreSpatial.W, 5.0);
        const float Wpersist = isnan(giPreSpatial.W) ? 0.0 : giPreSpatial.W;
        imageStore(giResXsWrite, ivec2(gl_LaunchIDEXT.xy), vec4(giPreSpatial.xs, giPreSpatial.W_sum));
        imageStore(giResNsWrite, ivec2(gl_LaunchIDEXT.xy), vec4(giPreSpatial.ns, giPreSpatial.M));
        imageStore(giResLoWrite, ivec2(gl_LaunchIDEXT.xy), vec4(giPreSpatial.Lo, Wpersist));

        // ── Stage 1c: spatial neighbour resampling ──
        // Disk-radius taps from the prev-frame reservoir buffer (giResXsRead
        // / giResNsRead / giResLoRead) — same images temporal uses, accepting
        // one-frame staleness rather than racing against this-frame's writes
        // mid-dispatch. Number of taps and disk radius follow the standard
        // ReSTIR PT / GI defaults (Bitterli 2020, Ouyang 2021): 5 taps when
        // static, 2 when camera moving (stale-history risk is higher under
        // motion); disk radius 20 px; per-neighbour M-cap 4 to prevent a
        // single stale-history reservoir from dominating; early-break once
        // total M reaches 20 to bound cost when many neighbours validate.
        //
        // Validation gate is mesh-ID + |Δdist|/dist < 0.1 — same as temporal.
        // The codebase doesn't carry a prev-normal image so the standard
        // cone-of-normals gate is approximated via mesh-ID. Curved surfaces
        // get more permissive acceptance, but `evalGiTarget` re-evaluates
        // at OUR pixel's normal, so a tap with very different surface
        // orientation produces low p_hat and barely shifts the reservoir.
        if (subHitSurface) {
            const ivec2 sz = imageSize(prevGbufImage);
            if (sz.x > 0 && sz.y > 0) {
                const bool camMoving = (pc.motionFlags & 2u) != 0u;
                // Reduced from 2/5 → 1/3: variance reduction plateaus past
                // M ≈ 10-15, and with M-cap-per-tap of 4, 3 taps already
                // reach ~12-16 added M. Saves ~40% of spatial cost for
                // imperceptible variance increase on the cases GI targets.
                const uint  spMax    = camMoving ? 1u : 3u;
                const float mTarget  = 20.0;
                const uint  curMeshId = uint(gl_InstanceCustomIndexEXT) + 1u;
                const float curDistC  = length(hitPos - pcam.prevCamPosX.xyz);
                for (uint sp = 0u; sp < spMax; ++sp) {
                    if (gi.M >= mTarget) break;
                    const float spAngle = urand(seed) * TWO_PI;
                    // Reduced from 20 → 12 px: tighter disk = better L2
                    // cache locality on reservoir reads, plus the wider
                    // radius pulled in cross-feature samples that the
                    // mesh-ID gate ended up rejecting anyway.
                    const float spR     = sqrt(urand(seed)) * 12.0;
                    const ivec2 spOff   = ivec2(int(spR * cos(spAngle)),
                                                int(spR * sin(spAngle)));
                    if (spOff.x == 0 && spOff.y == 0) continue;
                    const ivec2 spPx = clamp(ivec2(gl_LaunchIDEXT.xy) + spOff,
                                             ivec2(0), ivec2(sz - ivec2(1)));
                    const vec4 spGbuf  = imageLoad(prevGbufImage, spPx);
                    const uint spMeshId = uint(spGbuf.w);
                    if (spMeshId == 0u || spMeshId != curMeshId) continue;
                    const float spDist = length(spGbuf.xyz - pcam.prevCamPosX.xyz);
                    if (abs(spDist - curDistC) / max(curDistC, 1e-3) > 0.1) continue;
                    const vec4 spXs = imageLoad(giResXsRead, spPx);
                    const vec4 spNs = imageLoad(giResNsRead, spPx);
                    const vec4 spLo = imageLoad(giResLoRead, spPx);
                    const vec3 spXsP = spXs.xyz;
                    const vec3 spNsP = spNs.xyz;
                    const vec3 spLoP = spLo.rgb;
                    const float spM = min(spNs.w, 4.0);
                    const float spW = spLo.w;
                    if (spW <= 0.0 || spM <= 0.0 || dot(spNsP, spNsP) < 0.01) continue;
                    const vec3 omegaI_sp = spXsP - hitPos;
                    const float dist_sp  = length(omegaI_sp);
                    if (dist_sp < 1e-3) continue;
                    const vec3 L_sp = omegaI_sp / dist_sp;
                    // Re-evaluate target at OUR pixel (BRDF + normal) for the
                    // neighbour's stored sample. This is what makes the merge
                    // unbiased — neighbour's BRDF / normal don't apply here.
                    const vec3 F_sp = evalGiTarget(V, L_sp, N, F0, albedo,
                                                    roughness, metalness, alpha,
                                                    ccProb, ccWeight, ccRough,
                                                    baseScale, spLoP);
                    const float p_hat_sp = max(lum3(F_sp), 0.0);
                    if (p_hat_sp > 0.0) {
                        const float w_sp = p_hat_sp * spM * spW;
                        gi.W_sum += w_sp;
                        gi.M     += spM;
                        if (urand(seed) < w_sp / max(gi.W_sum, 1e-20)) {
                            gi.xs     = spXsP;
                            gi.ns     = spNsP;
                            gi.Lo     = spLoP;
                            gi.omegaI = L_sp;
                            gi.p_hat  = p_hat_sp;
                        }
                    }
                }
                finalizeGiReservoir(gi);
            }
        }

        if (pc.fireflyClamp < 1e20) gi.W = min(gi.W, 5.0);

        if (subHitSurface && gi.W > 0.0 && gi.M > 0.0) {
            // ── Visibility test for the chosen sample ──
            // When the reservoir picks a sample from prev frame, xs may now
            // be occluded from our current primary — camera moved, an opaque
            // mesh slid into the line of sight, etc. The shadow ray closes
            // that loop. For this-frame candidates, the ray is along bs.dir
            // and trivially passes (we just traced through that segment in
            // the sub-trace) but the cost is uniform either way.
            const vec3 toXs = gi.xs - hitPos;
            const float distChosen = length(toXs);
            if (distChosen > 1e-3) {
                const vec3 omegaI_chosen = toXs / distChosen;
                shadowVisibility = 1.0;
                traceRayEXT(topAS,
                            gl_RayFlagsTerminateOnFirstHitEXT |
                            gl_RayFlagsSkipClosestHitShaderEXT |
                            gl_RayFlagsNoOpaqueEXT,
                            0xff, 1, 0, 1,
                            hitPos + N * 1e-3, 0.0, omegaI_chosen,
                            distChosen - 1e-2, 1);
                const float vis = shadowVisibility;

                // ── Contribution at primary ──
                // Standard RIS estimator: F · W at the chosen sample, F
                // evaluated at OUR pixel's BRDF.
                const vec3 F_chosen = evalGiTarget(V, omegaI_chosen, N, F0, albedo,
                                                    roughness, metalness, alpha,
                                                    ccProb, ccWeight, ccRough, baseScale, gi.Lo);
                vec3 contrib_raw = F_chosen * gi.W * vis;
                if (pc.fireflyClamp < 1e20) {
                    const float giLum = lum3(contrib_raw);
                    if (giLum > pc.fireflyClamp) contrib_raw *= pc.fireflyClamp / giLum;
                }
                giContrib = contrib_raw;
            }
            // else: degenerate xs (numerical edge case) → giContrib stays 0.
        } else if (!subHitSurface) {
            // Sub-trace missed the geometry (env / numeric). Use the
            // sub-trace's radiance directly as the bounce-1 contribution —
            // miss.rmiss has already MIS-halved the env color (we set
            // giSubPayload.flags = 2u above), so bs.weight · Lo_cand is the
            // unbiased single-sample MC estimator for the env-via-bounce-1
            // half of the env MIS pair. No reservoir for env samples in
            // Stage 1b — positional xs doesn't make sense for sky/HDRI
            // (Stage 2 / full-path Lo would store them as directional).
            vec3 envContrib = bs.weight * Lo_cand;
            if (pc.fireflyClamp < 1e20) {
                const float l = lum3(envContrib);
                if (l > pc.fireflyClamp) envContrib *= pc.fireflyClamp / l;
            }
            giContrib = envContrib;
        }

        // ── Continuation hand-off ──
        // Stage 2: GI's Lo now includes bounce-2 (the sub-trace at xs ran its
        // own recursive sub-sub-trace that folded the second bounce into its
        // returned radiance). Letting raygen continue from xs into bounce-2
        // would double-count that energy, so the GI-consumed branch always
        // terminates after primary. Bounces 3+ at GI primaries are dropped;
        // the missing-multibounce bias is small on diffuse-dominant scenes
        // (≤5-10% of indirect energy past bounce-2 in most measured cases)
        // and trades cleanly against Stage 1's biased continuation hack
        // (which used this-frame's xs continuation even when the reservoir
        // chose a prev-frame sample).
        giNextOrigin = vec3(0.0);
        giNextDir    = vec3(0.0);
        giThroughput = vec3(0.0);
        giTerminate  = true;
        giConsumed   = true;
    }

    // ── Fast-path completion ──
    // Saturated/converged pixel: skip sub-trace + Stage 2 + RIS + temporal
    // + spatial. Use the persisted reservoir directly for shading. Visibility
    // test still fires to catch dynamic occluders changing between frames.
    // (Vis-fail behaviour matches the standard path on this branch — produces
    // 0 contribution; if/when vis-fail decay lands in the standard path,
    // mirror it here.)
    if (useGIPrimary && giFastPath) {
        GiReservoir gi = giFast;
        if (pc.fireflyClamp < 1e20) gi.W = min(gi.W, 5.0);
        if (gi.W > 0.0 && gi.M > 0.0) {
            const vec3 toXs = gi.xs - hitPos;
            const float distChosen = length(toXs);
            if (distChosen > 1e-3) {
                const vec3 omegaI_chosen = toXs / distChosen;
                shadowVisibility = 1.0;
                traceRayEXT(topAS,
                            gl_RayFlagsTerminateOnFirstHitEXT |
                            gl_RayFlagsSkipClosestHitShaderEXT |
                            gl_RayFlagsNoOpaqueEXT,
                            0xff, 1, 0, 1,
                            hitPos + N * 1e-3, 0.0, omegaI_chosen,
                            distChosen - 1e-2, 1);
                const float vis = shadowVisibility;
                if (vis > 0.0) {
                    const vec3 F_chosen = evalGiTarget(V, omegaI_chosen, N, F0, albedo,
                                                        roughness, metalness, alpha,
                                                        ccProb, ccWeight, ccRough, baseScale, gi.Lo);
                    vec3 contrib_raw = F_chosen * gi.W * vis;
                    if (pc.fireflyClamp < 1e20) {
                        const float giLum = lum3(contrib_raw);
                        if (giLum > pc.fireflyClamp) contrib_raw *= pc.fireflyClamp / giLum;
                    }
                    giContrib = contrib_raw;
                }
            }
        }
        giNextOrigin = vec3(0.0);
        giNextDir    = vec3(0.0);
        giThroughput = vec3(0.0);
        giTerminate  = true;
        giConsumed   = true;
    }

    // Phase 1: route ALL contributions to diff channel for now. Phase 1b
    // will split litDiff and litSpec at each NEE site for proper channel
    // separation. Keeping spec=0 for this initial split preserves current
    // behavior pixel-identical while the infrastructure lands.
    if (giConsumed) {
        // GI absorbed bounce 1's contribution; raygen's step 1 trace continues
        // from xs at bounce 2. payload.flags bit 3 (=8u) is a documentation
        // marker that raygen could use for special handling later (e.g.,
        // skipping FC adjustments for "free" GI bounces); the existing
        // bounce-loop maths is already correct without inspecting bit 3 since
        // the next-origin/dir + brdfWeight encode the post-bounce-1 state.
        payload.radianceDiff = emissiveOut + ambient + lit + giContrib;
        payload.radianceSpec = vec3(0.0);
        payload.brdfWeight   = giTerminate ? vec3(0.0) : giThroughput;
        payload.nextOrigin   = giTerminate ? vec3(0.0) : giNextOrigin;
        payload.nextDir      = giTerminate ? vec3(0.0) : giNextDir;
        payload.flags        = giTerminate ? 1u : 8u;
    } else {
        payload.radianceDiff = emissiveOut + ambient + lit;
        payload.radianceSpec = vec3(0.0);
        payload.brdfWeight   = brdfWeight;
        payload.nextOrigin   = hitPos + N * 1e-3;
        payload.nextDir      = bounceDir;
        payload.flags        = pathFlags;
    }
    // BSDF_ONLY override: zero the radiance fields so raygen's `radiance +=
    // throughput * (radianceDiff + radianceSpec)` at primary contributes
    // nothing — raygen injects the cached direct from s=0 instead. Bounce
    // fields (brdfWeight / nextOrigin / nextDir / flags) are preserved so
    // raygen's bounce loop continues with this sample's BSDF direction.
    if (bsdfOnlyMode) {
        payload.radianceDiff = vec3(0.0);
        payload.radianceSpec = vec3(0.0);
    }
    payload.seed = seed;
    // Per-vertex motion vector: interpolate this triangle's PREVIOUS-frame
    // vertex positions (via gdesc.prevVertexAddress) using the same
    // barycentric weights as the hit, then transform with the current
    // worldMatrix (gl_ObjectToWorldEXT). raygen uses motionMat · prevWorld
    // for reprojection. For static meshes prevVertexAddress == vertexAddress
    // so this resolves to the same world position as hitPos (no harm, no
    // change to existing behaviour). For skinned/displaced meshes the
    // prevVertex buffer holds frame N-1's deformed positions and we get
    // the per-vertex motion vector raygen needs to track the same surface
    // point across deformation.
    {
        VertexBuf pvb = VertexBuf(gdesc.prevVertexAddress);
        const vec3 pv0 = vec3(pvb.p[idx.x * 3 + 0], pvb.p[idx.x * 3 + 1], pvb.p[idx.x * 3 + 2]);
        const vec3 pv1 = vec3(pvb.p[idx.y * 3 + 0], pvb.p[idx.y * 3 + 1], pvb.p[idx.y * 3 + 2]);
        const vec3 pv2 = vec3(pvb.p[idx.z * 3 + 0], pvb.p[idx.z * 3 + 1], pvb.p[idx.z * 3 + 2]);
        const vec3 prevLocal = w * pv0 + attribs.x * pv1 + attribs.y * pv2;
        payload.prevWorldPos = vec3(gl_ObjectToWorldEXT * vec4(prevLocal, 1.0));
    }
    // Tag this surface for raygen's primary-hit reprojection. Pass-through and
    // transmission early-returns above leave hitInstanceId at 0 so raygen waits
    // for the first real shade event to anchor history. +1 so 0 still means
    // miss/sky after gl_InstanceCustomIndexEXT == 0.
    payload.hitWorldPos     = hitPos;
    payload.hitInstanceId   = uint(gl_InstanceCustomIndexEXT) + 1u;
    payload.hitRoughness    = roughness;// post-clamp; raygen uses for FC cap on motion
    payload.hitMetalness    = metalness;
    payload.hitTransmission = mdesc.transmission;
    payload.hitNormal       = N;// world-space shading normal for ReSTIR GI candidate (xs, ns, Lo)
    // Estimate the fraction of outgoing radiance that comes from the
    // VIEW-DEPENDENT spec lobe vs the view-INDEPENDENT diffuse lobe.
    // Drives the FC decay signal in raygen — surface-anchored reprojection
    // mixes prev-view spec with new-view spec, so a high spec fraction
    // means most of this pixel's history is "wrong-view" content that
    // should be flushed quickly on motion.
    //
    // Energy model: Fresnel at view angle gives the spec lobe's reflectance
    // weight; the complement times Lambertian-integrated albedo gives the
    // diffuse weight. Metals have F0=albedo (high spec, no diff). Dark
    // dielectrics have F0=0.04 but tiny albedo, so spec dominates anyway —
    // which is why wet asphalt / dark paint trail visibly under camera
    // motion even though they look "rough/diffuse" by roughness alone.
    {
        const vec3 F_at_V = fresnelSchlick(NdotV, F0);
        const float specW = max(F_at_V.r, max(F_at_V.g, F_at_V.b));
        const float diffW = (1.0 - specW) * (1.0 - metalness)
                            * max(albedo.r, max(albedo.g, albedo.b));
        payload.hitSpecFrac = specW / max(specW + diffW, 1e-6);
    }
    // 3-lobe pdf at the chosen bounce direction (cc + base-spec + base-diff).
    // Miss uses this for the BSDF→env MIS weight when env CDF is enabled. Must
    // match the actual sampler mixture above, otherwise MIS over/underweights
    // and adds noise on clearcoat.
    payload.bsdfPdf = brdfPdf3(V, bounceDir, N, roughness, metalness, ccProb, ccRough);

    // ── Primary-surface albedo for atrous demodulation (binding 35) ──
    // Gate: write a valid (.a=1) albedo ONLY at the primary hit on a
    // diffuse-dominant, non-emissive, non-transmissive surface. The atrous
    // shader divides radiance by max(albedo, eps) before its edge-aware
    // filter — when the surface has high-frequency texture detail (paint,
    // weathering on the boat hull) the lum-stop sees albedo×illumination
    // and rejects taps based on texture variation it can't tell apart from
    // real edges. Demodulating leaves only illumination — much smoother,
    // so the stop loosens correctly and fireflies actually get filtered.
    // For metals (F0=albedo, no diffuse lobe), glass (transmissive), and
    // emissives (emission term doesn't factor through albedo), we write
    // .a=0 so the atrous shader treats this pixel as "no demod" and
    // filters in radiance space — the safe fallback.
    const bool isPrimary       = (payload.inFlags & 1u) == 0u;
    const bool diffuseDominant = (metalness < 0.5) && (mdesc.transmission < 0.05);
    const bool notEmissive     = emLum1 < 0.5;
    // Also gate demod on low spec fraction. Surfaces with significant
    // view-dependent spec content (dark dielectrics at grazing — wet asphalt,
    // dark plastic cones at low NdotV, with specFrac > 0.3) produce
    // demod ghost trails under camera motion because the atrous kernel
    // mixes per-pixel spec lobe content spatially in illumination space
    // and the resulting smear is highly visible. For those pixels the
    // radiance-space filter is the right tool (per-pixel spec stays per
    // pixel, no spatial mixing of view-dep content). For low-specFrac
    // surfaces (textured paint, cube cells), demod still pays its way.
    const bool lowSpecFrac     = payload.hitSpecFrac < 0.3;
    if (isPrimary && diffuseDominant && notEmissive && lowSpecFrac) {
        // Clamp albedo to [0.04, 1] for stable demod — pure-black surfaces
        // would divide to infinity at the atrous step. 0.04 matches the
        // F0 floor and is below any plausible real albedo.
        const vec3 stableAlb = clamp(albedo, vec3(0.04), vec3(1.0));
        payload.primaryAlbedo = vec4(stableAlb, 1.0);
    }
    // Non-primary / non-diffuse / emissive / high-specFrac surfaces: leave
    // payload.primaryAlbedo at vec4(0.0) (set by raygen pre-init). Raygen
    // writes vec4(0) to the temporal albedo accumulator → atrous treats
    // the pixel as demod-disabled.
}
