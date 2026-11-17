// The GGX visible-normal sampler, one copy for every shader that draws a
// microfacet the eye can see: the path tracer's GGX lobes (path_trace.comp)
// and the water's reflection lobe in the deferred shade (shadeWater). The
// includer defines TWO_PI.
//
// Heitz 2018 ("Sampling the GGX Distribution of Visible Normals", JCGT 7(4)):
// a GGX microfacet normal distributed by the visible normals, i.e. a facet
// drawn as often as it faces the eye. Unlike drawing from the NDF alone it
// never returns a facet turned away from Ve, and at a grazing view it keeps
// the lobe's real shape: stretched along the line of sight.
// Ve is the view direction in the tangent frame (z = N); a is GGX alpha
// (roughness^2); u is two uniforms in [0, 1). Returns the facet normal in the
// same frame.

#ifndef GGX_VNDF_GLSL
#define GGX_VNDF_GLSL

vec3 sampleGgxVndf(vec3 Ve, float a, vec2 u) {
    const vec3  Vh    = normalize(vec3(a * Ve.x, a * Ve.y, Ve.z));
    const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    const vec3  T1    = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    const vec3  T2    = cross(Vh, T1);
    const float r     = sqrt(u.x);
    const float phi   = TWO_PI * u.y;
    const float t1    = r * cos(phi);
    float       t2    = r * sin(phi);
    const float sc    = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - sc) * sqrt(max(0.0, 1.0 - t1 * t1)) + sc * t2;
    const vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    return normalize(vec3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
}

#endif// GGX_VNDF_GLSL
