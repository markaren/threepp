// Shared analytic cloud DENSITY field (Nubis/HZD-lite, procedural).
//
// The single world-space density function that couples the far-field cloud
// march (cloud_march.comp) with the near-field heterogeneous froxels (Phase C)
// — a fly-through is seamless only because both volumes sample THIS function.
// A Perlin-Worley-style fBm remapped by coverage, shaped by a height gradient
// (puffy base → wispy top) and eroded at the edges by a higher-frequency Worley
// fBm — the classic Decima/Horizon recipe, evaluated ANALYTICALLY (hash value +
// cellular noise) so there are NO baked 3D texture assets (reuse-first repo).
// Wind scrolls the field and evolveSpeed advances an independent phase so the
// clouds churn rather than merely translate.
//
// REQUIRES a `clouds` uniform block in scope (binding 58) with fields
// bottomY/topY/coverage/density/evolveSpeed/wind — declared by every includer
// (deferred_shade.comp, cloud_march.comp).

float cloudHash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
vec3 cloudHash33(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453123);
}
float cloudValueNoise(vec3 p) {
    const vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    const float n000 = cloudHash13(i + vec3(0.0, 0.0, 0.0));
    const float n100 = cloudHash13(i + vec3(1.0, 0.0, 0.0));
    const float n010 = cloudHash13(i + vec3(0.0, 1.0, 0.0));
    const float n110 = cloudHash13(i + vec3(1.0, 1.0, 0.0));
    const float n001 = cloudHash13(i + vec3(0.0, 0.0, 1.0));
    const float n101 = cloudHash13(i + vec3(1.0, 0.0, 1.0));
    const float n011 = cloudHash13(i + vec3(0.0, 1.0, 1.0));
    const float n111 = cloudHash13(i + vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}
float cloudValueFbm(vec3 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 4; ++i) { s += a * cloudValueNoise(p); p *= 2.02; a *= 0.5; }
    return s;
}
// Inverted Worley (cellular) F1 — bright cores, dark cell walls: the classic
// cauliflower erosion shape.
float cloudWorley(vec3 p) {
    const vec3 id = floor(p);
    const vec3 fp = fract(p);
    float d = 1.0;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        const vec3 g = vec3(float(dx), float(dy), float(dz));
        const vec3 o = cloudHash33(id + g);
        const vec3 r = g + o - fp;
        d = min(d, dot(r, r));
    }
    return 1.0 - clamp(sqrt(d), 0.0, 1.0);
}
float cloudWorleyFbm(vec3 p) {
    return cloudWorley(p) * 0.625 + cloudWorley(p * 2.03) * 0.25 + cloudWorley(p * 4.01) * 0.125;
}
float cloudRemap(float v, float a, float b, float c, float d) {
    return c + (clamp(v, a, b) - a) * (d - c) / max(b - a, 1e-5);
}
// Density at a world point (0 outside the shell). time drives wind + evolution.
float cloudDensity(vec3 p, float time) {
    const float thick = max(clouds.topY - clouds.bottomY, 1.0);
    const float h01   = (p.y - clouds.bottomY) / thick;
    if (h01 < 0.0 || h01 > 1.0) return 0.0;
    // Height profile: rounded, puffy toward the base, tapering to wisps up top.
    const float grad = smoothstep(0.0, 0.12, h01) * (1.0 - smoothstep(0.55, 1.0, h01));
    if (grad <= 0.0) return 0.0;
    const vec3 wind = vec3(clouds.wind.x, 0.0, clouds.wind.z) * time;
    const vec3 evo  = vec3(0.0, clouds.evolveSpeed * time * 6.0, 0.0);
    // Base shape ~ 1.6 km features.
    float base = cloudValueFbm((p + wind) * 0.00075 + evo * 0.00075);
    // Coverage carves the layer: higher coverage → more of the field survives.
    float shape = cloudRemap(base, 1.0 - clamp(clouds.coverage, 0.0, 1.0), 1.0, 0.0, 1.0);
    shape *= grad;
    if (shape <= 0.0) return 0.0;
    // Detail erosion at the edges (~120 m features), scrolling faster than base.
    const float det = cloudWorleyFbm((p + wind * 2.0) * 0.008 + evo * 0.004);
    shape = cloudRemap(shape, det * 0.45, 1.0, 0.0, 1.0);
    return clamp(shape, 0.0, 1.0) * clouds.density;
}
