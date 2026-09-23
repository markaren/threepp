// Closed-form optical depth of the exponential height fog (the air medium)
// along a straight leg — the one implementation every shader that fogs a leg
// with it calls: the deferred shade (deferred_shade_60_fog_volumetrics.glsl),
// the GI/reflection recombine (deferred_filter_common.glsl), lit and unlit
// particles (particle_light.comp, particle.frag), ParticleField billboards
// (particlefield_billboard.vert) and splats (splat_common.glsl). Callers pass
// their own UBO fields; nothing here reads a binding.
//
//   yA, yB         world heights of the leg's two ends
//   len            leg length
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
// len is clamped to 1e7: a caller's distance() to a sentinel end point
// (1e30) squares past fp32 max to Inf, and Inf * 0 is NaN. The result
// saturates at 80 (exp(-80) ~ 1.8e-35), so every exp(-od) sees a finite value.
// The integral of sigma0 e^{-max(y,base)/H} along the leg is
// sigma0 * len * (e^{-ya/H} - e^{-yb/H}) / ((yb - ya)/H); that difference
// cancels catastrophically in fp32 as x = (yb - ya)/H -> 0 (a huge H, i.e.
// near-uniform fog), where the Taylor series of (1 - e^{-x})/x is used.

#ifndef THREEPP_HEIGHT_FOG_GLSL
#define THREEPP_HEIGHT_FOG_GLSL

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

#endif// THREEPP_HEIGHT_FOG_GLSL
