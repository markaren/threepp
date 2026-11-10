// Closed-form optical depth of the exponential height fog (the air medium)
// along a straight leg - the one implementation every shader that fogs a leg
// with it calls: the deferred shade (deferred_shade_60_fog_volumetrics.glsl),
// the GI/reflection recombine (deferred_filter_common.glsl), lit and unlit
// particles (particle_light.comp, particle.frag), ParticleField billboards
// (particlefield_billboard.vert) and splats (splat_common.glsl). Callers pass
// their own UBO fields; nothing here reads a binding.
//
//   a, b / yA, yB  the leg's two ends (world points, or world heights)
//   len            leg length (the height form only)
//   density        sigma_t at baseY (<= 0: no height fog)
//   falloff        scale height H of the exponential profile
//   baseY          the profile is constant (sigma0) below this height
//   waterSurfaceY  world height of the water surface; >= 1e29 means none
//
// The air medium stops at the waterline. The profile clamps to constant
// sigma0 below baseY, so with a water surface present the whole submerged
// column would carry the air medium at full base density, on top of the murk
// every caller adds for that portion of the leg. The leg is clipped to its
// above-water part instead. waterSurfaceY unset (1e30) leaves it alone.
//
// Two forms. The POINT form clips by moving the submerged end onto the
// surface with mix() and measuring what is left, so a sentinel end point
// (1e30 along the ray, the deferred shade's sky legs) still measures its
// above-water part correctly: mix() lands on the crossing and distance() of
// the remaining leg is finite. The HEIGHT form has no points, only a length,
// and clips by scaling that length with the above-water fraction; exact for
// a finite length, which is all the particle shaders ever have. A length that
// overflowed to Inf is clamped to 1e7 first and the scaled leg comes out far
// too short, so a caller with a sentinel end point uses the point form.
//
// len is clamped to 1e7: a caller's distance() to a sentinel end point
// (1e30) squares past fp32 max to Inf, and Inf * 0 is NaN. The result
// saturates at 80 (exp(-80) ~ 1.8e-35), so every exp(-od) sees a finite value.
// The integral of sigma0 e^{-max(y,base)/H} along the leg is
// sigma0 * len * (e^{-ya/H} - e^{-yb/H}) / ((yb - ya)/H); that difference
// cancels catastrophically in fp32 as x = (yb - ya)/H -> 0 (a huge H, i.e.
// near-uniform fog), where the Taylor series of (1 - e^{-x})/x is used.

#ifndef THREEPP_HEIGHT_FOG_GLSL
#define THREEPP_HEIGHT_FOG_GLSL

// The integral over a leg that is already clipped to the air.
float heightFogAirLegOpticalDepth(float yA, float yB, float len,
                                  float density, float falloff, float baseY) {
    len = min(len, 1.0e7);
    const float H  = max(falloff, 1e-3);
    const float ya = max(yA - baseY, 0.0);
    const float yb = max(yB - baseY, 0.0);
    const float ea = exp(-ya / H);
    const float eb = exp(-yb / H);
    const float x  = (yb - ya) / H;
    const float f  = (abs(x) < 1e-3) ? (ea * (1.0 - 0.5 * x + x * x * (1.0 / 6.0)))
                                     : ((ea - eb) / x);
    return min(density * len * f, 80.0);
}

// Point form: the leg [a, b]. Use it whenever the points are at hand.
float heightFogLegOpticalDepth(vec3 a, vec3 b,
                               float density, float falloff, float baseY,
                               float waterSurfaceY) {
    if (density <= 0.0) return 0.0;
    if (waterSurfaceY < 1e29) {
        const float wa = a.y - waterSurfaceY;
        const float wb = b.y - waterSurfaceY;
        if (wa < 0.0 && wb < 0.0) return 0.0;// wholly submerged
        const float tc = wa / (wa - wb);      // surface crossing, fraction from a
        if (wa < 0.0)      a = mix(a, b, tc);
        else if (wb < 0.0) b = mix(a, b, tc);
    }
    return heightFogAirLegOpticalDepth(a.y, b.y, distance(a, b), density, falloff, baseY);
}

// Height form: the leg's two heights and its FINITE length.
float heightFogLegOpticalDepth(float yA, float yB, float len,
                               float density, float falloff, float baseY,
                               float waterSurfaceY) {
    if (density <= 0.0) return 0.0;
    len = min(len, 1.0e7);
    if (waterSurfaceY < 1e29) {
        const float wa = yA - waterSurfaceY;
        const float wb = yB - waterSurfaceY;
        if (wa < 0.0 && wb < 0.0) return 0.0;// wholly submerged
        const float tc = wa / (wa - wb);      // surface crossing, fraction from A
        if (wa < 0.0) {
            yA = waterSurfaceY;
            len *= 1.0 - tc;
        } else if (wb < 0.0) {
            yB = waterSurfaceY;
            len *= tc;
        }
    }
    return heightFogAirLegOpticalDepth(yA, yB, len, density, falloff, baseY);
}

// ── Noise-displaced fog top (HeightFogSettings::topNoise) ────────────────────
// The base height a leg ending at world point p integrates with: baseY shifted
// by amp * n, n in [-1, 1] a two-octave value noise of p * invScale + offset
// (the host folds wind drift and evolution into offset). A caller passes the
// result as the baseY of the functions above, so the leg keeps its closed form
// and the top of the layer moves with the far end of the leg. The deferred
// shade and the deferred filters call it with the same arguments, since the
// recombine has to carry the extinction the shade applied.
//
// amp <= 0 returns baseY untouched. So does a sentinel end point (1e30 along
// the ray on a sky leg), where the noise argument has no fractional part left.
//
// The noise is a triangle-wave noise (the "tri noise" family): four octaves of
// folded triangle waves, each warped by a triangle-wave vector field of the
// octave below. It has no lattice and no hash, so it costs a few dozen fract()
// per sample, and the folding gives filaments where a value-noise fBm gives
// rounded blobs. `phase` advances every octave in place (the evolution).
// Range [0, ~0.57], mean about 0.29.
float heightFogTri(float x) { return abs(fract(x) - 0.5); }
vec3 heightFogTri3(vec3 p) {
    return vec3(heightFogTri(p.z + heightFogTri(p.y)),
                heightFogTri(p.z + heightFogTri(p.x)),
                heightFogTri(p.y + heightFogTri(p.x)));
}
float heightFogTriNoise(vec3 p, float phase) {
    float z = 1.4, sum = 0.0;
    vec3 bp = p;
    for (int i = 0; i < 4; ++i) {
        p += heightFogTri3(bp * 2.0) + phase;
        bp *= 1.8;
        z  *= 1.5;
        p  *= 1.2;
        sum += heightFogTri(p.z + heightFogTri(p.x + heightFogTri(p.y))) / z;
        bp += 0.14;
    }
    return sum;
}
float heightFogDisplacedBaseY(vec3 p, float baseY, float amp, float invScale, vec3 offset) {
    if (amp <= 0.0) return baseY;
    if (abs(p.x) + abs(p.y) + abs(p.z) > 1.0e6) return baseY;
    // offset.xz is the wind drift; offset.y is the evolution phase.
    const vec3  q = p * invScale + vec3(offset.x, 0.0, offset.z);
    // Two fields an octave apart, the finer one evolving a little faster so
    // the pattern changes shape as it drifts. Their sum has mean ~0.57.
    const float n = heightFogTriNoise(q, offset.y) + heightFogTriNoise(q * 2.0, offset.y * 1.2);
    return baseY + amp * clamp((n - 0.57) * 3.0, -1.0, 1.0);
}

#endif// THREEPP_HEIGHT_FOG_GLSL
