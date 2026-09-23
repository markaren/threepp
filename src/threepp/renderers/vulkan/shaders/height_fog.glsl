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

#endif// THREEPP_HEIGHT_FOG_GLSL
