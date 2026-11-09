#ifndef THREEPP_OCEAN_CHOP_NORMAL_GLSL
#define THREEPP_OCEAN_CHOP_NORMAL_GLSL

// Slope of the displaced (choppy) ocean surface, shared by the vertex normal
// (water_displace.comp) and the fine cascade's per-pixel moments
// (ocean_chop_moments.comp).
//
//   s    = (dh/dx, dh/dz) at the REST point, world units
//   jxx  = l * dDx/dx, jzz = l * dDz/dz, jxz = l * dDx/dz (l = choppiness)
//
// The surface P = (x + l*Dx, h, z + l*Dz) has the normal
// (c*sz - b*sx, J, c*sx - a*sz) with a = 1 + jxx, b = 1 + jzz, c = jxz and
// J = a*b - c*c (the cross product of its two tangents; Tessendorf 2001,
// "Simulating Ocean Water", eq. 44 for P). Divided by J that is the slope
// returned here. A fold (J <= 0) is where the choppy model's surface turns
// over, which real water does by breaking (Nouguier, Guerin and Chapron 2009,
// JGR 114). a, b and J are floored so the slope never flips sign there, and
// its length is capped at tan(75 deg): the steepest a single shading normal
// is allowed to lean, so a fold reads as a near-vertical face, not a hole.
vec2 oceanChopSlope(vec2 s, float jxx, float jzz, float jxz) {
    const float a = max(1.0 + jxx, 0.2);
    const float b = max(1.0 + jzz, 0.2);
    const float J = max(a * b - jxz * jxz, 0.04);
    vec2 sd = vec2(b * s.x - jxz * s.y, a * s.y - jxz * s.x) / J;
    const float kMaxSlope = 3.7320508;// tan(75 deg)
    const float len = length(sd);
    if (len > kMaxSlope) sd *= kMaxSlope / len;
    return sd;
}

#endif// THREEPP_OCEAN_CHOP_NORMAL_GLSL
