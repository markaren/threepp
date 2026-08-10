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
layout(set = 0, binding = 68) uniform ParticleDensityUbo {
    vec4  boxMin[kMaxDensityFields];    // xyz = world min corner, w = resolution
    vec4  boxInvSize[kMaxDensityFields];// xyz = 1 / (2 * halfExtent), w unused
    vec4  albedoAniso;                  // rgb = medium single-scatter albedo, a = HG g
    uvec4 counts;                       // x = active volume count, yzw reserved
} pd;

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

#endif// THREEPP_PARTICLE_DENSITY_GLSL
