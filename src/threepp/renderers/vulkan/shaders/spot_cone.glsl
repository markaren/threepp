// spot_cone.glsl — where a ray runs inside a spot light's cone. One copy for
// the deferred shade's beam march (deferred_shade_60_fog_volumetrics.glsl)
// and the path trace's lamps in the air (path_trace.comp).
#ifndef SPOT_CONE_GLSL
#define SPOT_CONE_GLSL

// The stretch of the ray ro + t·rd inside the FORWARD nappe of a spot cone
// (apex P, unit axis A, cosine of the half angle cosHalf): [x, y], an open end
// as ±1e30, x > y when the ray misses. A cone no wider than a hemisphere is
// convex, so a line meets it in one interval. The interval only decides where
// the beam march puts its samples; the cone factor stays in the integrand. So
// every case the arithmetic cannot settle (a ray parallel to the cone's side,
// a wider cone) returns the whole line: more samples spent outside the beam,
// never light lost.
vec2 spotConeInterval(vec3 ro, vec3 rd, vec3 P, vec3 A, float cosHalf) {
    const vec2 whole = vec2(-1e30, 1e30);
    const vec2 none  = vec2(1e30, -1e30);
    if (cosHalf <= 0.0) return whole;
    const vec3  o  = ro - P;
    const float dA = dot(rd, A);
    const float oA = dot(o, A);
    const float c2 = cosHalf * cosHalf;
    // f(t) = (A·(o + t·rd))² − c2·|o + t·rd|² = a·t² + 2b·t + c, >= 0 inside
    // the double cone (rd is unit length).
    const float a = dA * dA - c2;
    const float b = dA * oA - c2 * dot(rd, o);
    const float c = oA * oA - c2 * dot(o, o);
    if (abs(a) < 1e-6) return whole;
    const float disc = b * b - a * c;
    if (disc < 0.0) return (a > 0.0) ? whole : none;
    // The stable root pair (no cancellation when one root is far away).
    const float q  = -(b + ((b >= 0.0) ? sqrt(disc) : -sqrt(disc)));
    const float r0 = q / a;
    const float r1 = (abs(q) > 1e-30) ? c / q : r0;
    const float lo = min(r0, r1), hi = max(r0, r1);
    // The ray's direction inside the cone's angle: it is in one nappe for
    // t <= lo and in the other for t >= hi; the forward one is the end it runs
    // toward along A.
    if (a > 0.0) return (dA > 0.0) ? vec2(hi, 1e30) : vec2(-1e30, lo);
    // Outside the cone's angle: inside between the roots, in one nappe.
    return (oA + 0.5 * (lo + hi) * dA > 0.0) ? vec2(lo, hi) : none;
}

#endif// SPOT_CONE_GLSL
