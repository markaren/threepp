// The geometric (face) normal of a G-buffer pixel, reconstructed from its
// depth neighbours. The G-buffer stores only the SHADING normal (vertex normal
// plus normal map), which can lean far from the triangle it shades: a
// displaced or normal-mapped wall whose shading normal points up. Anything
// that has to step OFF the surface or tell its visible side from the solid
// behind it (the probe lookup, probe_common.glsl) needs this one.
//
// The includer defines, before including (s = the MSAA sample the centre was
// reconstructed from, or -1):
//   float dnDepth(ivec2 q, int s)   NDC depth at texel q
//   uint  dnId(ivec2 q, int s)      instance id + 1 at texel q (0 = sky)
// and has the camera UBO (camera_ubo.glsl) in scope.

vec3 dnWorldPos(vec2 ndc, float depth) {
    const vec4 vh = cam.projInverse * vec4(ndc, depth, 1.0);
    return (cam.viewInverse * vec4(vh.xyz / vh.w, 1.0)).xyz;
}

// P = the centre's world position, reconstructed from `ndc` (jitter and MSAA
// sample offset included: the neighbours reuse them). Per axis the nearer of
// the two neighbours on the same instance wins, so a silhouette or a fold
// takes the side it belongs to; with neither, the shading normal `Nfallback`
// is returned. The result faces the viewer (V = towards the eye).
vec3 depthGeometricNormal(ivec2 px, int s, vec2 ndc, uint id, vec3 P, vec3 V, vec3 Nfallback) {
    const ivec2 size    = ivec2(int(pc.width), int(pc.height));
    const vec2  ndcStep = vec2(2.0 / float(pc.width), -2.0 / float(pc.height));// +1 texel in x, y
    vec3 axis[2];
    for (int a = 0; a < 2; ++a) {
        const ivec2 e  = ivec2(1 - a, a);
        vec3  best    = vec3(0.0);
        float bestLen = -1.0;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            const ivec2 q = px + e * sgn;
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size)) || dnId(q, s) != id) continue;
            const vec3  d = (dnWorldPos(ndc + vec2(e) * ndcStep * float(sgn), dnDepth(q, s)) - P) * float(sgn);
            const float l = dot(d, d);
            if (bestLen < 0.0 || l < bestLen) { best = d; bestLen = l; }
        }
        if (bestLen <= 0.0) return Nfallback;
        axis[a] = best;
    }
    const vec3  Ng = cross(axis[0], axis[1]);
    const float l2 = dot(Ng, Ng);
    if (l2 <= 1e-12 * dot(axis[0], axis[0]) * dot(axis[1], axis[1])) return Nfallback;// degenerate
    const vec3 n = Ng * inversesqrt(l2);
    return dot(n, V) < 0.0 ? -n : n;
}
