// Splat reflection volumes — the SAMPLING half of
// plans/splat-volume-reflections.md (Part 2, "the table plumbing").
//
// SplatPass bakes each resident SplatCloud once, in cloud-local space, into an
// rgba16f 3D image (rgb = linear radiance, a = sigma_t per LOCAL metre —
// splat_bake_resolve.comp). This header is the table that binds those images
// and the ONE function that marches them: svLeg, for rays the tile rasterizer
// cannot serve.
//
// ── WHY A SECOND TABLE AND NOT FOUR MORE ROWS IN THE FIRST ───────────────────
// particle_density.glsl's table (bindings 67/68/69) is the wrong home, for
// reasons that are shape rather than taste:
//
//   • its entries are WORLD-AXIS-ALIGNED boxes; a splat volume is an OBB
//     (cloud-local content under an arbitrary model matrix) and needs a
//     world->UVW matrix per entry;
//   • its content is re-scattered EVERY FRAME by ParticleFieldPass; a splat
//     volume is static residency owned by SplatPass and freed by retireStale —
//     two lifecycles in one array means two owners for one array;
//   • its march decodes fixed-point and evaluates a blackbody ramp; ours is one
//     filtered rgba16f tap. A flavour branch inside pdEmissiveLeg's loop would
//     change the arithmetic every dust scene executes, which is exactly what
//     the "textually identical when off" doctrine forbids.
//
// So: the same PATTERN — array of samplers, std140 UBO, dummy-filled slots,
// uniform count gate — and none of the coupling.
//
// ── WHAT THE MARCH IS AND IS NOT ─────────────────────────────────────────────
// svLeg is pdEmissiveLeg's structural twin: the same union-of-intervals clip,
// the same 16 fixed steps, the same midpoint phase, no jitter, the same
// Beer-Lambert ordering, the same tau > 8 early-out, the same skip-don't-clamp
// in-box test. What it DELIBERATELY OMITS is exactly what pdEmissiveLeg omits,
// under the same house rule — ONE medium model, no second differently-wrong
// copy of the full march (see pdEmissiveLeg's header, "WHAT IS DELIBERATELY
// OMITTED"): no sun in-scatter, no ambient/env in-scatter, no per-step
// point-light glow.
//
// For splats that omission costs even less than it does for dust, because 3DGS
// colour is BAKED RADIANCE: the primary rasterizer evaluates the SH basis and
// composites it unlit, so treating the DC term as emission here is the same
// lighting model rather than an approximation of a different one. A scan
// reflected in water is its captured radiance dimmed by water Fresnel — which
// is what a photograph of that pond would show anyway, because the capture
// already contains the lighting.
//
// The view-dependent SH orders are dropped (the bake keeps the DC term only),
// which a Fresnel-weighted, wave-perturbed reflection cannot resolve anyway.
//
// ── THE PRIMARY LEG MUST NEVER CALL THIS ─────────────────────────────────────
// SplatPass already composites the real thing on the camera leg, so adding the
// volume to applyParticleFog's consumers would double-count every visible
// cloud. That is the one wrong turn the design makes easy; it is stated in the
// plan and enforced by construction — this function has its own call sites, and
// none of them is on the primary leg.

#ifndef THREEPP_SPLAT_VOLUME_GLSL
#define THREEPP_SPLAT_VOLUME_GLSL

// Volumes bound simultaneously. KEEP IN SYNC with kMaxSplatVolumes in
// SplatPass.hpp (which is itself pinned to SplatPass::kMaxClouds) and with
// kMaxSplatVolumeSlots in DeferredShade.cpp. Slots past the live count are
// bound to the renderer's 1x1x1 dummy so every descriptor is always valid.
#define kMaxSplatVolumes 8

layout(set = 0, binding = 70) uniform sampler3D splatVolTex[kMaxSplatVolumes];

// std140 (NOT scalar), for particle_density.glsl's reason: everything here is a
// mat4/vec4/uvec4, so the layout is std140-clean by construction and the header
// costs its includers no extension. Mirrored one for one by SplatVolumeUboGpu
// (SplatPass.hpp), which carries the static_assert(sizeof) drift guard —
// the ParticleDensityUboGpu precedent, copied.
layout(set = 0, binding = 71) uniform SplatVolumeUbo {
    // World point -> [0,1]^3 of that cloud's bake box. The host folds the local
    // box normalisation into inverse(model), so the per-step transform is ONE
    // mat4 multiply rather than an inverse-transform plus a rescale.
    mat4  worldToUvw[kMaxSplatVolumes];
    // World AABB of the transformed (rotated, possibly non-uniformly scaled)
    // box — CONSERVATIVE, and used only to clip the ray interval. The exact
    // membership test is the in-UVW one below, after the matrix.
    vec4  worldBoxMin[kMaxSplatVolumes];
    vec4  worldBoxMax[kMaxSplatVolumes];
    // x = sigmaScale / cbrt(|det model3x3|): sigma is stored per LOCAL metre,
    // so a scaled cloud needs it re-expressed per WORLD metre. Exact for
    // uniform scale, a stated approximation otherwise. yzw reserved.
    vec4  params[kMaxSplatVolumes];
    uvec4 counts;// x = active volume count, yzw reserved
} sv;

// 16, for pdEmissiveLeg's reason and not by coincidence: a reflected image is
// Fresnel-weighted (F ~ 0.02 face-on), perturbed by the wave normals that
// already dither the sampling across neighbouring pixels, and never the subject
// of the frame. Fixed step count, midpoint phase, NO temporal jitter — the
// determinism contract. If a banding shell ever shows, the sanctioned answer is
// a per-PIXEL hash offset (spatial dither, no frame term), not jitter.
const int kSvLegSteps = 16;

// Transmittance of the leg [ro, ro + rd*tMax]; `emis` receives the radiance
// emitted along it, already self-occluded by the medium in front of it.
// EXACTLY 1.0 / vec3(0.0) — the caller's no-op — whenever no volume is bound,
// which is what keeps every splat-free scene on the path it had before this
// function existed.
//
// Indexing splatVolTex by the loop variable is dynamically uniform (the loop
// bound is a uniform), the same pattern pdEmissiveLeg already relies on at its
// particleDensityLinTex[i] tap.
float svLeg(vec3 ro, vec3 rd, float tMax, out vec3 emis) {
    emis = vec3(0.0);
    const uint n = min(sv.counts.x, uint(kMaxSplatVolumes));
    if (n == 0u) return 1.0;

    // Union of the ray's box overlaps — ONE interval, as pdEmissiveLeg does it,
    // so two clouds that share a courtyard share the step budget.
    float t0 = tMax, t1 = 0.0;
    const vec3 inv = 1.0 / rd;// ±inf on axis-parallel components is fine below
    for (uint i = 0u; i < n; ++i) {
        const vec3  ta = (sv.worldBoxMin[i].xyz - ro) * inv;
        const vec3  tb = (sv.worldBoxMax[i].xyz - ro) * inv;
        const vec3  lo = min(ta, tb), hi = max(ta, tb);
        const float e = max(max(lo.x, lo.y), max(lo.z, 0.0));
        const float x = min(min(hi.x, hi.y), min(hi.z, tMax));
        if (x > e) { t0 = min(t0, e); t1 = max(t1, x); }
    }
    if (t1 <= t0) return 1.0;

    const float dt = (t1 - t0) / float(kSvLegSteps);
    float tau = 0.0;
    for (int s = 0; s < kSvLegSteps; ++s) {
        const vec3 x = ro + rd * (t0 + (float(s) + 0.5) * dt);
        float sig = 0.0;
        vec3  em  = vec3(0.0);
        for (uint i = 0u; i < n; ++i) {
            const vec3 uvw = (sv.worldToUvw[i] * vec4(x, 1.0)).xyz;
            // Skip, don't clamp — CLAMP_TO_EDGE smears the boundary voxels over
            // the whole world, which reads as a cloud with infinite extent. The
            // same rule pdSampleVolume documents, applied in UVW because that is
            // where an OBB's membership test lives.
            if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) continue;
            const vec4  tap = texture(splatVolTex[i], uvw);
            const float si  = tap.a * sv.params[i].x;
            sig += si;
            // Emission ∝ sigma: the pdEmissionAt physics slot with the blackbody
            // ramp replaced by the BAKED radiance. Same reason it is a product
            // there — a medium radiates in proportion to how much of it there is
            // — and it is what makes the cloud's silhouette in the mirror be the
            // splat distribution rather than a card.
            em += tap.rgb * si;
        }
        if (sig <= 0.0) continue;
        // Same order as pdEmissiveLeg and as the primary march: this step's
        // emission is attenuated by the medium IN FRONT of it, not including its
        // own slab.
        const float tr = exp(-tau);
        tau  += sig * dt;
        emis += em * (tr * dt);
        if (tau > 8.0) break;// e^-8: nothing behind this survives
    }
    return exp(-tau);
}

#endif// THREEPP_SPLAT_VOLUME_GLSL
