// Shadow maps (VulkanRenderer::ShadowMode::ShadowMaps, ShadowMapPass): the
// receiver side. Included by deferred_shade.comp and particle_light.comp,
// both variants; set 0 is the deferred shade's set, which binds the view's
// maps at 80-82 and their UBO at 83 (1x1 dummies and a zeroed UBO when the
// view renders without maps, so every lookup answers "no map here").
//
// The maps are reverse-Z depth, as every depth target of the renderer: 1 at
// the light's near plane, 0 at its far plane, cleared to 0. The sampler
// compares GREATER_OR_EQUAL against the receiver's depth, so a tap is lit
// when the receiver is at least as near the light as the nearest caster the
// map holds, and the border (depth 0, "nothing cast") reads lit.
//
// Both lookups return 1 lit .. 0 shadowed, or -1 when no map covers the
// point (outside every cascade, a lamp without a slot, the maps not in use).
// The caller decides what -1 means: the ray-traced mode traces a ray there,
// the no-ray-tracing mode takes the light as unoccluded.
//
// The filter is three.js r186's PCF (shadowmap_pars_fragment.glsl): five
// taps on a Vogel disc of LightShadow::radius texels, turned per pixel by
// interleaved gradient noise (static per pixel; the TAA and the shadow
// channel's spatial filter smooth it), each tap a hardware 2x2 compare, and
// mix(1, shadow, LightShadow::intensity). The sun's disc adds to its radius
// (cascadeParams.z, VulkanCoreShadows.cpp's fit), a fixed kernel per cascade. LightShadow::bias keeps its
// three.js meaning (depth units of the map, negative = toward the light) and
// LightShadow::normalBias its world meaning (the receiver moved along its
// normal, metres); a light that leaves normalBias at 0 gets 1.5 texels of
// the map at the receiver.

#ifndef SHADOW_MAPS_GLSL
#define SHADOW_MAPS_GLSL

layout(set = 0, binding = 80) uniform sampler2DArrayShadow   smCascadeTex;
layout(set = 0, binding = 81) uniform sampler2DArrayShadow   smSpotTex;
layout(set = 0, binding = 82) uniform samplerCubeArrayShadow smPointTex;
// GpuShadowMapsUbo (VulkanGpuLayouts.hpp).
layout(set = 0, binding = 83, scalar) uniform ShadowMapsUbo {
    mat4  cascadeVP[kMaxShadowCascades];
    vec4  cascadeSphere[kMaxShadowCascades];// xyz centre, w radius (world)
    vec4  cascadeParams[kMaxShadowCascades];// x one texel (m), y depth units per metre, z PCF radius (texels)
    mat4  spotVP[kMaxShadowSpots];
    vec4  pointPosFar[kMaxShadowPoints];
    vec4  lampParams[kMaxShadowSpots + kMaxShadowPoints];// bias, normal offset (m), radius (texels), intensity
    vec4  lampTexel[kMaxShadowSpots + kMaxShadowPoints]; // texel angle (tan), near, far, map size
    uint  cascadeCount;
    uint  spotCount;
    uint  pointCount;
    uint  flags;// bit 0: the sun's cascades are valid
    float sunBias;
    float sunNormalBias;
    float sunIntensity;
    vec3  sunDir;
    float cascadeMapSize;
} smU;

float smIgn(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

vec2 smVogel(int i, int n, float phi) {
    const float r = sqrt((float(i) + 0.5) / float(n));
    const float t = float(i) * 2.399963229728653 + phi;
    return vec2(cos(t), sin(t)) * r;
}

float smPcf2D(sampler2DArrayShadow tex, float layer, vec2 uv, float ref, float radiusUv, float phi) {
    float s = 0.0;
    for (int i = 0; i < 5; ++i) {
        s += texture(tex, vec4(uv + smVogel(i, 5, phi) * radiusUv, layer, ref));
    }
    return s * 0.2;
}

// The sun (dir light 0). N: the surface normal for the receiver offset, or
// vec3(0) for a point in a medium (no offset; the constant bias remains; one
// tap instead of the PCF kernel).
float smSunVis(vec3 P, vec3 N, vec2 pixel) {
    if ((smU.flags & 1u) == 0u) return -1.0;
    // The cascades are nested spheres around the view's frustum slices: the
    // first that holds P (the normal offset and the filter clear of its rim)
    // is the finest. Its outer tenth hands over to the next by a static dither,
    // which the TAA resolves; no blend, so no second lookup.
    const int n  = int(min(smU.cascadeCount, uint(kMaxShadowCascades)));
    int       ci = -1;
    for (int i = 0; i < n; ++i) {
        const vec4  s = smU.cascadeSphere[i];
        const float r = s.w - (2.0 + smU.cascadeParams[i].z) * smU.cascadeParams[i].x;
        const vec3  d = P - s.xyz;
        if (dot(d, d) >= r * r) continue;
        ci = i;
        if (i + 1 < n) {
            const vec4  s1 = smU.cascadeSphere[i + 1];
            const float r1 = s1.w - (2.0 + smU.cascadeParams[i + 1].z) * smU.cascadeParams[i + 1].x;
            const vec3  d1 = P - s1.xyz;
            if (dot(d1, d1) < r1 * r1 &&
                smIgn(pixel + vec2(37.0, 11.0)) < smoothstep(0.9 * r, r, length(d))) {
                ci = i + 1;
            }
        }
        break;
    }
    if (ci < 0) return -1.0;
    const float texel   = smU.cascadeParams[ci].x;
    const float nb      = smU.sunNormalBias > 0.0 ? smU.sunNormalBias : 1.5 * texel;
    const bool  surface = dot(N, N) > 0.25;
    const vec3  Pb      = surface ? P + N * nb : P;
    const vec4  c       = smU.cascadeVP[ci] * vec4(Pb, 1.0);
    const vec2  uv      = vec2(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    // One texel toward the sun, in the map's depth units, then three.js's bias.
    const float ref     = c.z + texel * smU.cascadeParams[ci].y - smU.sunBias;
    // A point in a medium (the fog and shaft marches, a particle) takes one
    // hardware 2x2 tap: those march many jittered steps and blur by
    // themselves, and five taps a step cost the fog more than its rays did.
    const float s = surface ? smPcf2D(smCascadeTex, float(ci), uv, ref,
                                      max(smU.cascadeParams[ci].z, 0.0) / max(smU.cascadeMapSize, 1.0),
                                      smIgn(pixel) * 6.283185307)
                            : texture(smCascadeTex, vec4(uv, float(ci), ref));
    return mix(1.0, s, smU.sunIntensity);
}

// A point or spot light through its slot (GpuPointLight / GpuSpotLight /
// GpuClusterLight::shadowSlot, by the light's type). The cone and the range
// stay the caller's.
float smLampVis(float slot, bool spot, vec3 P, vec3 N, vec3 lightPos, vec2 pixel) {
    if (slot < 0.0) return -1.0;
    const int k = int(slot + 0.5);
    if (k >= int(spot ? smU.spotCount : smU.pointCount)) return -1.0;
    const int   pi    = spot ? k : kMaxShadowSpots + k;
    const vec4  prm   = smU.lampParams[pi];
    const vec4  tx    = smU.lampTexel[pi];
    const float dist  = length(P - lightPos);
    const float texel = tx.x * dist;// one texel of the map at P (m)
    const vec3  Pb    = P + N * (prm.y > 0.0 ? prm.y : 1.5 * texel);
    const float phi   = smIgn(pixel) * 6.283185307;
    if (spot) {
        const vec4 c = smU.spotVP[k] * vec4(Pb, 1.0);
        if (c.w <= 1e-6) return 1.0;// behind the spot: its cone sends nothing there
        // The constant bias: the receiver's depth taken one texel nearer the light.
        const vec3  toL = lightPos - Pb;
        const vec4  cb  = smU.spotVP[k] * vec4(Pb + toL * (texel / max(length(toL), 1e-4)), 1.0);
        const float ref = cb.z / cb.w - prm.x;
        const vec2  uv  = vec2(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
        return mix(1.0, smPcf2D(smSpotTex, float(k), uv, ref, prm.z / max(tx.w, 1.0), phi), prm.w);
    }
    // getPointShadow (three.js r186) on the cube array: the direction picks the
    // face in hardware, the depth is the largest |component| (the face's view
    // depth) turned into that face's reverse-Z perspective depth, and the taps
    // turn about the direction in a tangent frame.
    const vec3  v    = Pb - lightPos;
    const vec3  av   = abs(v);
    const float vz   = max(max(av.x, av.y), av.z) - texel;
    const float near = tx.y, far = tx.z;
    if (vz > far || vz < near) return 1.0;
    const float dp   = (far * (vz - near)) / (vz * (far - near));
    const float ref  = (1.0 - dp) - prm.x;
    const vec3  bd   = normalize(v);
    const vec3  ad   = abs(bd);
    const vec3  t    = normalize(cross(bd, ad.x > ad.z ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    const vec3  b    = cross(bd, t);
    const float ts   = prm.z / max(tx.w, 1.0);
    float s = 0.0;
    for (int i = 0; i < 5; ++i) {
        const vec2 o = smVogel(i, 5, phi);
        s += texture(smPointTex, vec4(bd + (t * o.x + b * o.y) * ts, float(k)), ref);
    }
    return mix(1.0, s * 0.2, prm.w);
}

#endif// SHADOW_MAPS_GLSL
