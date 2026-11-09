#ifndef THREEPP_OCEAN_CHOP_NORMAL_GLSL
#define THREEPP_OCEAN_CHOP_NORMAL_GLSL

// Slope of the displaced (choppy) ocean surface, shared by the vertex normal
// (water_displace.comp, OCEAN_CHOP variant) and the fine cascade's per-pixel
// moments (ocean_chop_moments.comp).
//
//   s    = (dh/dx, dh/dz) at the REST point, world units
//   jxx  = l * dDx/dx, jzz = l * dDz/dz, jxz = l * dDx/dz (l = choppiness)
//
// The surface P = (x + l*Dx, h, z + l*Dz) has the normal
// (c*sz - b*sx, J, c*sx - a*sz) with a = 1 + jxx, b = 1 + jzz, c = jxz and
// J = a*b - c*c (the cross product of its two tangents; Tessendorf 2001,
// "Simulating Ocean Water", eq. 44 for P). Divided by J that is the slope of
// the surface the geometry draws: steeper by about 1/J where the chop pinches
// a crest.
//
// Where J falls toward 0 the horizontal map is about to fold, which real
// water does by breaking (Nouguier, Guerin and Chapron 2009, "Choppy wave
// model for nonlinear gravity waves", JGR 114), and 1/J stops describing
// anything a pixel could show. So the correction is weighted by
// w = smoothstep(0.05, 0.35, J) (0 also where a or b is not positive, a fold
// along one axis): full above J = 0.35, gone below 0.05, where the slope is
// the rest slope s. The correction that was faded out is not dropped: its
// squared length is returned in .z, the slope variance a pixel there cannot
// resolve, which the per-pixel path banks as roughness. The folds themselves
// stay with the Jacobian foam. Its length is capped at tan(75 deg) so the
// banked variance stays finite at a fold.
//
// Returns (slope x, slope z, banked variance).
vec3 oceanChopSlope(vec2 s, float jxx, float jzz, float jxz) {
    const float a = 1.0 + jxx;
    const float b = 1.0 + jzz;
    const float J = a * b - jxz * jxz;
    const float w = (a > 0.0 && b > 0.0) ? smoothstep(0.05, 0.35, J) : 0.0;
    vec2 sd = vec2(b * s.x - jxz * s.y, a * s.y - jxz * s.x) / max(J, 0.05);
    const float kMaxSlope = 3.7320508;// tan(75 deg)
    const float len = length(sd);
    if (len > kMaxSlope) sd *= kMaxSlope / len;
    const vec2 d = (1.0 - w) * (sd - s);
    return vec3(s + w * (sd - s), dot(d, d));
}

#endif// THREEPP_OCEAN_CHOP_NORMAL_GLSL
