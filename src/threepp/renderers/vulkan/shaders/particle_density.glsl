// Shared WORLD-SPACE particle density volume — the sampling half of
// plans/particle-field.md §3.3 (phase 2).
//
// A ParticleField with DensityRepr enabled is scattered ONCE per frame into an
// r32ui 3D image covering a world box (particle_density_scatter.comp). Every
// consumer of the medium then SAMPLES that image: the froxel inject/integrate
// passes through mediumExtinction() in cloud_density.glsl, and the surface path
// through the froxel LUT's transmittance channel. The volume is world-anchored
// precisely so K views share ONE scatter — per-view froxel grids are view-
// anchored and scattering N particles into K of them would cost K·N.
//
// ── WHY r32ui AND NOT r16f/r32f ──────────────────────────────────────────────
// imageAtomicAdd on a FLOAT format needs VK_EXT_shader_atomic_float, and — the
// reason that would be rejected even where it exists — float addition is not
// associative, so the value a voxel ends up with depends on the order the GPU
// happened to schedule the atomics in. That is a per-run coin flip, and the
// sensor goldens this whole subsystem exists to serve cannot hold across one.
// INTEGER adds are associative, so an r32ui fixed-point accumulator is
// bit-reproducible run to run by construction. (It also sidesteps the storage-
// image subnormal-flush trap: nothing here ever goes through uintBitsToFloat —
// the decode is float(u) * scale, which is exact for every u < 2^24 and
// correctly rounded above it.)
//
// ── THE FIXED-POINT SCALE: Q20.12 (12 fractional bits, scale 4096) ───────────
// A voxel holds sigma_t in 1/m as a uint with 12 fractional bits:
//
//   quantum      1/4096       = 2.44e-4 /m   (the thinnest dust a single tap
//                                             can express; below it the tap
//                                             rounds to zero — plan R4's
//                                             "thin dust quantises to zero")
//   saturation   2^32 / 4096  = 1.05e6 /m    (the u32 wraps above this — R4's
//                                             "dense dust saturates")
//
// That covers everything physical by a wide margin: thin haze is sigma_t ~1e-3
// /m (3 km visibility), a heavy dust cloud ~1-10 /m, and optically solid smoke
// ~100 /m — the top of the range needs 10^6 particles of sigma 1.0 landing in
// ONE voxel, which is a volume whose resolution is grossly mismatched to its
// contents rather than a scene. 16 fractional bits were rejected for exactly
// that reason (they cap at 65536 /m, which 10^6 particles DO reach); 8 were
// rejected because their 1/256 quantum kills a 0.05-weight trilinear tap of a
// 0.05 sigma particle outright.
//
// Splatting is 8-tap trilinear and sampling is 8-tap trilinear, which is what
// turns 10^5 discrete points into a continuous medium instead of confetti.
//
// ── SAMPLING: DIRECT UINT FETCH, NOT A CONVERT PASS ──────────────────────────
// §3.3 leaves the choice open between "u32 -> r16f convert pass then texture()"
// and "direct uint fetch + unfix in the shader". Direct fetch wins on
// simplicity: no second image, no second pipeline, no second barrier, and no
// question about r16f storage/format support. The cost is that an integer
// format cannot be hardware-filtered, so the trilinear below is 8 texelFetches
// by hand — ~1.2M sampling points per frame across the two froxel passes, i.e.
// ~9.4M fetches, which is microseconds of texture bandwidth on this class of
// GPU and far below the fixed froxel cost it rides inside.

#ifndef THREEPP_PARTICLE_DENSITY_GLSL
#define THREEPP_PARTICLE_DENSITY_GLSL

// Volumes bound simultaneously. One per ParticleField with DensityRepr on;
// fields past this many are ignored (the host reports which). Unused slots are
// bound to a 1x1x1 dummy so every descriptor is always valid.
#define kMaxDensityFields 4

// KEEP IN SYNC with particle_density_scatter.comp and with
// ParticleFieldPass.hpp's kDensityFixedScale.
const float kParticleDensityScale    = 4096.0;
const float kParticleDensityInvScale = 1.0 / 4096.0;

layout(set = 0, binding = 67) uniform usampler3D particleDensityTex[kMaxDensityFields];

// The r16f mirror of each volume (particle_density_convert.comp), declared
// only where the per-pixel dust march runs (deferred_shade defines PD_LINEAR
// before its includes). The froxel pipelines do not carry binding 69, so the
// declaration must not exist there.
#ifdef PD_LINEAR
layout(set = 0, binding = 69) uniform sampler3D particleDensityLinTex[kMaxDensityFields];
#endif

// std140 (NOT scalar): this header is pulled into shaders that do not all
// enable GL_EXT_scalar_block_layout, and the layout below is std140-clean by
// construction — everything is a vec4/uvec4.
//
// ── WHY albedoAniso IS PER FIELD (F-A of plans/particle-atmosphere.md) ───────
// It used to be ONE vec4 shared by every bound volume, filled from whichever
// field happened to be enumerated first. That was a documented wart with no
// cost while the only client was dust — two dust clouds rarely differ — but it
// makes the DEFINING fire scene impossible: a fire field and a smoke field in
// one scene are a bright warm medium and a dark grey one, and one shared albedo
// cannot be both. Per-field costs 48 B of UBO and one indexed load at a site
// that already loops per volume, so there is no argument for the shared form
// once anything needs two media. `emission` is new for the same loop.
layout(set = 0, binding = 68) uniform ParticleDensityUbo {
    vec4  boxMin[kMaxDensityFields];     // xyz = world min corner, w = resolution
    vec4  boxInvSize[kMaxDensityFields]; // xyz = 1 / (2 * halfExtent), w unused
    vec4  albedoAniso[kMaxDensityFields];// rgb = single-scatter albedo, a = HG g
    // x = emissive intensity (0 = pure dust, the exact no-op), y = blackbody T
    // at the box bottom (K), z = T at the box top (K), w = ramp exponent.
    vec4  emission[kMaxDensityFields];
    // x = active volume count, y = 1 when ANY bound volume is emissive (the
    // uniform branch that keeps dust-only scenes on the pre-emission path),
    // zw reserved.
    uvec4 counts;
} pd;

// ── Analytic blackbody radiance, linear Rec.709, ~10 ALU, no LUT asset ───────
// Derived rather than tabulated (reuse-first: a 1D LUT here would be a new
// image, a new binding and a new upload for a curve that fits in two mads).
// Planck's law integrated against the CIE 1931 observer and converted to linear
// sRGB has, over 700-6500 K, a RED channel that is always the maximum — so
// normalising by it leaves only two curves to fit, and both are near-linear in
// kilokelvin:
//
//   T (K)      exact (r,g,b), max-normalised      this fit
//    1000        1.000 0.029 0.000                1.000 0.036 0.000
//    1900        1.000 0.243 0.000                1.000 0.237 0.000
//    2600        1.000 0.402 0.083                1.000 0.394 0.094
//    3000        1.000 0.484 0.155                1.000 0.484 0.171
//
// Worst component error is 0.017 over 700-3000 K, which is the whole range a
// sooty flame occupies. Above ~4000 K the green fit saturates early and the
// result reads slightly yellow rather than white (error 0.15 at 5500 K); that
// is out of scope on purpose — this is a fire ramp, not a white-point tool.
//
// MAGNITUDE is Stefan-Boltzmann, (T / 2000 K)^4: a grey body's radiance goes as
// T^4, which is what makes a flame's base blindingly brighter than its tips
// with no second authored curve. The 2000 K normalisation gives
// DensityRepr::emissiveIntensity the plain meaning "radiance of a 2000 K
// flame". T is clamped before the power so a nonsense authored temperature can
// neither produce a negative colour nor overflow to Inf.
vec3 blackbodyRGB(float T) {
    const float s = clamp(T, 500.0, 12000.0) * 0.001;// kilokelvin
    const float g = clamp(0.22394 * s - 0.18818, 0.0, 1.0);
    const float u = max(s - 2.09, 0.0);
    const float b = clamp(u * (0.17759 + 0.01179 * u), 0.0, 1.0);
    const float h  = s * 0.5;// T / 2000 K
    const float h2 = h * h;
    return vec3(1.0, g, b) * (h2 * h2);
}

// The emitted radiance ONE volume contributes at a sample point — the SINGLE
// expression the fire model has, factored out so that every consumer of it is
// the same model by construction rather than by review. `ty` is the sample's
// normalised height inside that volume's box (already in [0,1] — every caller
// reaches this only after the in-box test, which is what keeps the pow's base
// non-negative; a negative base is undefined in GLSL and has produced NaN in
// this tree before, feedback_float_pi_sin_pow_nan). `sigma` is THAT volume's
// own extinction there, because emission ∝ σ is the physics (soot radiates in
// proportion to how much soot there is) and is what makes a flame's silhouette
// be the particle distribution instead of a billboard card.
//
// Callers gate on pd.emission[i].x > 0.0 themselves: the multiply would be a
// no-op but blackbodyRGB and the pow would not.
vec3 pdEmissionAt(uint i, float ty, float sigma) {
    const float T = mix(pd.emission[i].y, pd.emission[i].z, pow(ty, pd.emission[i].w));
    return (pd.emission[i].x * sigma) * blackbodyRGB(T);
}

// One volume's contribution at a world point, trilinear. Returns 0 outside the
// box (NOT clamp-to-edge: a clamped read smears the boundary voxels across the
// whole world, which reads as infinite dust).
//
// The sampler is a function parameter so the array index is always a compile-
// time constant at the call sites below — no dynamic sampler indexing, hence
// no dependence on shaderSampledImageArrayDynamicIndexing.
float pdSampleVolume(usampler3D vol, vec4 boxMin, vec4 boxInvSize, vec3 p) {
    const vec3 t = (p - boxMin.xyz) * boxInvSize.xyz;
    if (any(lessThan(t, vec3(0.0))) || any(greaterThan(t, vec3(1.0)))) return 0.0;
    const float res = boxMin.w;
    if (res < 1.0) return 0.0;
    // Voxel CENTRES sit at (i + 0.5) / res, so the continuous grid coordinate
    // of p is t*res - 0.5 — the same convention particle_density_scatter.comp
    // splats with, which is what makes splat and sample each other's inverse.
    const vec3  g  = t * res - 0.5;
    const vec3  g0 = floor(g);
    const vec3  f  = g - g0;
    const ivec3 b  = ivec3(g0);
    const int   hi = int(res) - 1;
    float acc = 0.0;
    for (int k = 0; k < 8; ++k) {
        const ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        const ivec3 c = clamp(b + o, ivec3(0), ivec3(hi));
        const vec3  w = mix(vec3(1.0) - f, f, vec3(o));
        acc += (w.x * w.y * w.z) * float(texelFetch(vol, c, 0).x);
    }
    return acc * kParticleDensityInvScale;
}

// Total particle extinction sigma_t (1/m) at a world point — the sum over
// every bound volume. Exactly 0.0 when no field has a density representation
// this frame, which is what keeps dust-free scenes image-identical.
float particleDensity(vec3 p) {
    const uint n = pd.counts.x;
    if (n == 0u) return 0.0;
    float s = 0.0;
    if (n > 0u) s += pdSampleVolume(particleDensityTex[0], pd.boxMin[0], pd.boxInvSize[0], p);
    if (n > 1u) s += pdSampleVolume(particleDensityTex[1], pd.boxMin[1], pd.boxInvSize[1], p);
    if (n > 2u) s += pdSampleVolume(particleDensityTex[2], pd.boxMin[2], pd.boxInvSize[2], p);
    if (n > 3u) s += pdSampleVolume(particleDensityTex[3], pd.boxMin[3], pd.boxInvSize[3], p);
    return s;
}

// particleDensity(), plus the medium parameters of the mixture at p — the
// SIGMA-WEIGHTED blend of the per-field albedo and HG g. This is the form the
// froxel injector needs now that those params are per field: where a fire
// volume and a smoke volume overlap, the medium a clustered light scatters off
// is neither one of them but their mixture, weighted by how much extinction
// each contributes AT THAT POINT — which is the physically right combination
// and the only one that degrades to "the single field's own values" when the
// volumes are disjoint (the usual case).
//
// It exists as a SEPARATE function rather than as extra out-parameters on
// particleDensity() because froxel_integrate.comp and the sensor paths want
// only the scalar and must not pay for the blend, and because sampling the
// volumes a second time just to fetch the params would double the froxel pass's
// texture fetches.
//
// n == 1 short-circuits to the field's values VERBATIM: the blend would
// otherwise compute a/s * s, whose float round-trip is not the identity, and a
// single dust field is every pre-emission scene in the tree.
float particleMedium(vec3 p, out vec3 albedo, out float g) {
    albedo = vec3(1.0);
    g      = 0.0;
    const uint n = pd.counts.x;
    if (n == 0u) return 0.0;
    if (n == 1u) {
        albedo = pd.albedoAniso[0].rgb;
        g      = pd.albedoAniso[0].a;
        return pdSampleVolume(particleDensityTex[0], pd.boxMin[0], pd.boxInvSize[0], p);
    }
    float s = 0.0;
    vec3  a = vec3(0.0);
    float gw = 0.0;
    // Unrolled for the same reason pdSampleVolume takes the sampler by value:
    // the descriptor index must be a compile-time constant.
    float si;
    si = pdSampleVolume(particleDensityTex[0], pd.boxMin[0], pd.boxInvSize[0], p);
    s += si; a += pd.albedoAniso[0].rgb * si; gw += pd.albedoAniso[0].a * si;
    si = pdSampleVolume(particleDensityTex[1], pd.boxMin[1], pd.boxInvSize[1], p);
    s += si; a += pd.albedoAniso[1].rgb * si; gw += pd.albedoAniso[1].a * si;
    if (n > 2u) {
        si = pdSampleVolume(particleDensityTex[2], pd.boxMin[2], pd.boxInvSize[2], p);
        s += si; a += pd.albedoAniso[2].rgb * si; gw += pd.albedoAniso[2].a * si;
    }
    if (n > 3u) {
        si = pdSampleVolume(particleDensityTex[3], pd.boxMin[3], pd.boxInvSize[3], p);
        s += si; a += pd.albedoAniso[3].rgb * si; gw += pd.albedoAniso[3].a * si;
    }
    if (s <= 0.0) return 0.0;// no volume covers p — the outs keep their defaults
    albedo = a / s;
    g      = gw / s;
    return s;
}

#ifdef PD_LINEAR
// σ_t at p from the r16f MIRRORS (one hardware-trilinear fetch per volume where
// the integer path costs eight texelFetches), plus the σ-weighted albedo of the
// mixture — the same quantity particleMedium() returns, over the cheap
// representation, for consumers that march a whole leg rather than sampling a
// point.
//
// Only for shaders that carry binding 69, which is why it sits behind the same
// PD_LINEAR guard the declaration does: the froxel pipelines do not have it.
//
// n == 1 short-circuits for particleMedium()'s reason — the a·s/s round trip is
// not the float identity, and one field is the overwhelmingly common case.
float pdMediumLinear(vec3 p, out vec3 albedo) {
    albedo = vec3(1.0);
    const uint n = min(pd.counts.x, uint(kMaxDensityFields));
    if (n == 0u) return 0.0;
    float s = 0.0;
    vec3  a = vec3(0.0);
    for (uint i = 0u; i < n; ++i) {
        const vec3 tt = (p - pd.boxMin[i].xyz) * pd.boxInvSize[i].xyz;
        // Skip, don't clamp — CLAMP_TO_EDGE smears the boundary voxels over the
        // whole world, which reads as infinite dust.
        if (any(lessThan(tt, vec3(0.0))) || any(greaterThan(tt, vec3(1.0)))) continue;
        const float si = texture(particleDensityLinTex[i], tt).r;
        s += si;
        a += pd.albedoAniso[i].rgb * si;
    }
    if (s <= 0.0) return 0.0;
    albedo = (n == 1u) ? pd.albedoAniso[0].rgb : (a / s);
    return s;
}

// ── THE CHEAP LEG: emission + extinction only, for SECONDARY rays ────────────
// plans/particle-atmosphere.md, "water-fire reflection march".
//
// A traced reflection ray sees GEOMETRY. A flame is not geometry — it is a
// participating medium, a ParticleField density volume with F0's blackbody
// emission ramp — so a campfire beside a pond reflected in that pond showed
// only the flicker PointLight's glint on the wave facets and none of the flame
// itself. The primary leg gets the full treatment from applyParticleFog; a
// reflected leg gets THIS: the same medium, the same emission expression
// (pdEmissionAt, shared), the same skip-don't-clamp box test, the same
// Beer-Lambert compositing order — and NOTHING ELSE.
//
// WHAT IS DELIBERATELY OMITTED, and why it is an omission and not a bug:
// the sun's in-scatter (one centroid shadow ray + HG phase + cloud shadow),
// the ambient/env in-scatter, and the per-step point-light glow. Those are
// applyParticleFog's 32-step march, and the house rule this tree runs on is
// ONE medium model — a second, differently-wrong copy of the full march is the
// duplicate BSDF that rule exists to forbid. So a smoke plume reflected in
// water DIMS what is behind it (transmittance) but is not itself lit by the
// sun in the mirror: it reads as a dark column rather than a bright one. That
// is the same trade particle_light.comp's overlay fog makes, recorded there
// for the same reason. The flame, being emission-dominated, loses nothing:
// emission IS the term this returns.
//
// Fixed step count, midpoint phase, no temporal jitter — the phase-2
// determinism contract. 16 rather than the primary march's 32: a reflected
// image is Fresnel-weighted (F ≈ 0.02 face-on), perturbed by the wave normals
// that already dither the sampling across neighbouring pixels, and never the
// subject of the frame. If a banding shell ever shows, F0's sanctioned answer
// is the per-PIXEL hash offset (spatial dither, no frame term), not jitter.
const int kPdLegSteps = 16;

// Returns the leg's transmittance; `emis` receives the emitted radiance
// accumulated along it, already self-occluded by the medium in front of it.
// EXACTLY 1.0 / vec3(0.0) — the caller's no-op — whenever no bound volume is
// emissive, which is what keeps every pre-existing dust scene on the path it
// had before this function existed.
float pdEmissiveLeg(vec3 ro, vec3 rd, float tMax, out vec3 emis) {
    emis = vec3(0.0);
    const uint n = min(pd.counts.x, uint(kMaxDensityFields));
    if (n == 0u || pd.counts.y == 0u) return 1.0;

    // Union of the ray's box overlaps — one interval, as applyParticleFog
    // does it, so two plumes that share a yard share the step budget.
    float t0 = tMax, t1 = 0.0;
    const vec3 inv = 1.0 / rd;// ±inf on axis-parallel components is fine below
    for (uint i = 0u; i < n; ++i) {
        const vec3  bmin  = pd.boxMin[i].xyz;
        const vec3  bsize = 1.0 / pd.boxInvSize[i].xyz;
        const vec3  ta = (bmin - ro) * inv;
        const vec3  tb = (bmin + bsize - ro) * inv;
        const vec3  lo = min(ta, tb), hi = max(ta, tb);
        const float e = max(max(lo.x, lo.y), max(lo.z, 0.0));
        const float x = min(min(hi.x, hi.y), min(hi.z, tMax));
        if (x > e) { t0 = min(t0, e); t1 = max(t1, x); }
    }
    if (t1 <= t0) return 1.0;

    const float dt = (t1 - t0) / float(kPdLegSteps);
    float tau = 0.0;
    for (int s = 0; s < kPdLegSteps; ++s) {
        const vec3 x = ro + rd * (t0 + (float(s) + 0.5) * dt);
        float sig = 0.0;
        vec3  em  = vec3(0.0);
        for (uint i = 0u; i < n; ++i) {
            const vec3 tt = (x - pd.boxMin[i].xyz) * pd.boxInvSize[i].xyz;
            if (any(lessThan(tt, vec3(0.0))) || any(greaterThan(tt, vec3(1.0)))) continue;
            const float si = texture(particleDensityLinTex[i], tt).r;
            sig += si;
            if (pd.emission[i].x > 0.0) em += pdEmissionAt(i, tt.y, si);
        }
        if (sig <= 0.0) continue;
        // Same order as the primary march: this step's emission is attenuated
        // by the medium IN FRONT of it, not including its own slab.
        const float tr = exp(-tau);
        tau  += sig * dt;
        emis += em * (tr * dt);
        if (tau > 8.0) break;// e^-8: nothing behind this survives
    }
    return exp(-tau);
}
#endif

#endif// THREEPP_PARTICLE_DENSITY_GLSL
