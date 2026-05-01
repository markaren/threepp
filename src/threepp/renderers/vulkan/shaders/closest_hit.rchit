#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Phase 9: this shader fills a struct payload (radiance leaving the hit +
// sampled bounce direction + throughput multiplier). raygen handles the
// path loop; we just shade the hit and pick where the path goes next.
//
// Direct lighting: Cook-Torrance GGX for specular, energy-conserving
// Lambert for diffuse, gated by per-light shadow rays. Specular env IBL
// is sampled directly from the equirect at the reflection direction
// (matches WGPU PT and standard raster IBL).
//
// Indirect bounce: cosine-weighted hemisphere sample. BRDF·cos/pdf for
// Lambert collapses to `albedo·(1-metalness)`, so metals get throughput=0
// and naturally do not contribute to diffuse GI. Specular indirect
// (mirror reflections of nearby geometry) is a follow-up — for now metals
// reflect the env map only.

struct Payload {
    vec3 radiance;
    vec3 brdfWeight;
    vec3 nextOrigin;
    vec3 nextDir;
    uint flags;
    uint seed;
};

layout(buffer_reference, scalar) readonly buffer NormalBuf { float n[]; };
layout(buffer_reference, scalar) readonly buffer IndexBuf  { uint  i[]; };
layout(buffer_reference, scalar) readonly buffer UvBuf     { float u[]; };

struct GeometryDesc {
    uint64_t normalAddress;
    uint64_t indexAddress;
    uint64_t uvAddress;// 0 == no UV attribute
    uint     indexed;
    uint     _pad;
};

struct MaterialDesc {
    vec3  albedo;
    float roughness;
    float metalness;
    vec3  emissive;
    float emissiveIntensity;
    int   albedoTexIndex;// -1 == no albedo texture
    float alphaCutoff;   // <= 0 == disabled (used by the any-hit shader)
};

const uint kMaxMaterialTextures = 64;

struct DirLight {
    vec3 direction;// world-space, toward the light
    vec3 color;    // already includes intensity
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topAS;
layout(set = 0, binding = 3, scalar) readonly buffer GeomDescBuf {
    GeometryDesc geoms[];
};
layout(set = 0, binding = 4, scalar) readonly buffer MatDescBuf {
    MaterialDesc mats[];
};
layout(set = 0, binding = 5, scalar) uniform LightsUbo {
    vec3     ambient;// already includes intensity
    uint     dirCount;
    DirLight dirLights[8];
} lights;
layout(set = 0, binding = 6) uniform sampler2D envTex;
// Bindless material albedo array. Indexed by mdesc.albedoTexIndex; slot 0 is
// the host's 1×1 white default, used implicitly via -1→0 fallback below.
layout(set = 0, binding = 8) uniform sampler2D albedoMaps[kMaxMaterialTextures];

// Phase 11: PMREM mip count comes via the same push-constant block used by
// raygen. .x is raygen's sampleIndex (not read here); .y is envMipCount.
layout(push_constant) uniform Pc {
    uint sampleIndex;
    uint envMipCount;
    uint _pad1;
    uint _pad2;
} pc;

hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT Payload payload;
// Shadow visibility: 0 = occluded (default), 1 = clear (set by shadow_miss).
layout(location = 1) rayPayloadEXT float shadowVisibility;

const float PI = 3.14159265358979;
const float TWO_PI = 6.28318530717958;

float distGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float geomSmithG1(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// PCG XSH-RR. Cheap, well-mixed, single-uint state — perfect for shaders.
uint pcgNext(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float urand(inout uint state) {
    return float(pcgNext(state)) / 4294967296.0;
}

// Tangent-space cosine-weighted hemisphere sample (Z up).
vec3 cosineHemisphere(vec2 u) {
    const float r = sqrt(u.x);
    const float phi = TWO_PI * u.y;
    return vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

// Build an arbitrary orthonormal basis around N.
mat3 makeTBN(vec3 N) {
    const vec3 up = abs(N.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    const vec3 T = normalize(cross(up, N));
    const vec3 B = cross(N, T);
    return mat3(T, B, N);
}

// Analytic Smith G1 for GGX (alpha = roughness²). Matches the form used
// by VNDF importance sampling — must NOT be swapped with the Schlick-Smith
// `geomSmithG1` above, which is a different approximation.
float smithG1(float NdotX, float alpha) {
    const float a2 = alpha * alpha;
    return 2.0 * NdotX /
           max(NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX), 1e-6);
}

// VNDF-sampled half-vector → reflected wi. Mirrors WgpuPathTracerShaders_Rt's
// sampleVNDF so polished metals reflect nearby geometry the same way both
// path tracers do. May produce below-horizon wi for grazing rays + high
// roughness; the caller checks `dot(N, wi) > 0` and aborts the bounce.
vec3 sampleVNDF(vec3 wo, vec3 n, float alpha, vec2 u) {
    const vec3 nt  = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0)
                                     : vec3(1.0, 0.0, 0.0);
    const vec3 t1  = normalize(cross(nt, n));
    const vec3 t2  = cross(n, t1);
    const vec3 woLocal = vec3(dot(wo, t1), dot(wo, t2), dot(wo, n));
    const vec3 woStr   = normalize(vec3(alpha * woLocal.x,
                                        alpha * woLocal.y,
                                        woLocal.z));
    const float lensq = woStr.x * woStr.x + woStr.y * woStr.y;
    const vec3 T1 = lensq > 1e-7
            ? vec3(-woStr.y, woStr.x, 0.0) / sqrt(lensq)
            : vec3(1.0, 0.0, 0.0);
    const vec3 T2 = cross(woStr, T1);

    const float r   = sqrt(u.x);
    const float phi = TWO_PI * u.y;
    const float t1s = r * cos(phi);
    const float s   = 0.5 * (1.0 + woStr.z);
    const float t2s = mix(sqrt(max(0.0, 1.0 - t1s * t1s)),
                          r * sin(phi), s);
    const vec3 nhLocal = t1s * T1 + t2s * T2 +
                         sqrt(max(0.0, 1.0 - t1s * t1s - t2s * t2s)) * woStr;
    const vec3 hLocal = normalize(vec3(alpha * nhLocal.x,
                                       alpha * nhLocal.y,
                                       max(1e-6, nhLocal.z)));
    const vec3 hm = hLocal.x * t1 + hLocal.y * t2 + hLocal.z * n;
    return reflect(-wo, hm);
}

void main() {
    const float w = 1.0 - attribs.x - attribs.y;

    const GeometryDesc gdesc = geoms[gl_InstanceCustomIndexEXT];
    const MaterialDesc mdesc = mats [gl_InstanceCustomIndexEXT];

    uvec3 idx;
    if (gdesc.indexed != 0u) {
        IndexBuf ib = IndexBuf(gdesc.indexAddress);
        idx = uvec3(ib.i[gl_PrimitiveID * 3 + 0],
                    ib.i[gl_PrimitiveID * 3 + 1],
                    ib.i[gl_PrimitiveID * 3 + 2]);
    } else {
        idx = uvec3(gl_PrimitiveID * 3 + 0,
                    gl_PrimitiveID * 3 + 1,
                    gl_PrimitiveID * 3 + 2);
    }

    NormalBuf nb = NormalBuf(gdesc.normalAddress);
    const vec3 n0 = vec3(nb.n[idx.x * 3 + 0], nb.n[idx.x * 3 + 1], nb.n[idx.x * 3 + 2]);
    const vec3 n1 = vec3(nb.n[idx.y * 3 + 0], nb.n[idx.y * 3 + 1], nb.n[idx.y * 3 + 2]);
    const vec3 n2 = vec3(nb.n[idx.z * 3 + 0], nb.n[idx.z * 3 + 1], nb.n[idx.z * 3 + 2]);

    const vec3 nObj = normalize(w * n0 + attribs.x * n1 + attribs.y * n2);
    vec3 N = normalize(transpose(mat3(gl_WorldToObjectEXT)) * nObj);

    const vec3 V = normalize(-gl_WorldRayDirectionEXT);
    if (dot(N, V) < 0.0) N = -N;// double-sided shading for thin meshes / cull-disabled
    const float NdotV = max(dot(N, V), 0.0);

    // UV interpolation (only when the geometry has a uv attribute). The
    // outer fallback is vec2(0) — harmless for materials without an albedo
    // texture; the slot-0 white default would just sample (0,0) anyway.
    vec2 uv = vec2(0.0);
    if (gdesc.uvAddress != 0ul) {
        UvBuf ub = UvBuf(gdesc.uvAddress);
        const vec2 uv0 = vec2(ub.u[idx.x * 2 + 0], ub.u[idx.x * 2 + 1]);
        const vec2 uv1 = vec2(ub.u[idx.y * 2 + 0], ub.u[idx.y * 2 + 1]);
        const vec2 uv2 = vec2(ub.u[idx.z * 2 + 0], ub.u[idx.z * 2 + 1]);
        uv = w * uv0 + attribs.x * uv1 + attribs.y * uv2;
    }

    // Albedo: scalar PBR colour modulated by the bound albedo map (sRGB
    // decode is hardware-side via the VK_FORMAT_R8G8B8A8_SRGB view).
    vec3 albedoSample = vec3(1.0);
    if (mdesc.albedoTexIndex >= 0) {
        const int idxClamped = clamp(mdesc.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
        albedoSample = texture(albedoMaps[idxClamped], uv).rgb;
    }
    const vec3  albedo    = mdesc.albedo * albedoSample;
    const float roughness = clamp(mdesc.roughness, 0.04, 1.0);
    const float metalness = clamp(mdesc.metalness, 0.0,  1.0);
    const vec3  F0        = mix(vec3(0.04), albedo, metalness);
    const float k         = (roughness + 1.0) * (roughness + 1.0) / 8.0;

    // World-space hit point, used as origin for the shadow rays. Offset
    // along the geometric normal to avoid self-intersection (acne).
    const vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Sum direct contribution from each scene-driven directional light.
    // Physical lights (three.js useLegacyLights = false): light.color
    // already encodes intensity, no 1/PI cancel. Each light is gated by a
    // shadow ray; closest_hit is skipped on those rays so only the shadow
    // miss handler matters. Linear HDR radiance is returned to raygen,
    // which handles the sRGB encode.
    vec3 lit = vec3(0.0);
    for (uint i = 0u; i < lights.dirCount; ++i) {
        const vec3 L = normalize(lights.dirLights[i].direction);
        const float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        // Shadow ray: any non-alpha-tested hit blocks the light. Without
        // OpaqueEXT the any-hit shader fires and lets cutout foliage etc.
        // pass light through their transparent texels.
        shadowVisibility = 0.0;
        traceRayEXT(topAS,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT,
                    0xff, 0, 0, 1,// missIndex = 1 (shadow miss)
                    hitPos + N * 1e-3, 0.0, L, 1e4, 1);
        if (shadowVisibility <= 0.0) continue;

        const vec3  H     = normalize(V + L);
        const float NdotH = max(dot(N, H), 0.0);
        const float VdotH = max(dot(V, H), 0.0);

        const vec3  F        = fresnelSchlick(VdotH, F0);
        const float D        = distGGX(NdotH, roughness);
        const float G        = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
        const vec3  specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
        const vec3  kd       = (vec3(1.0) - F) * (1.0 - metalness);
        const vec3  diffuse  = kd * albedo / PI;

        lit += (diffuse + specular) * NdotL * lights.dirLights[i].color;
    }

    // Flat ambient irradiance — only the diffuse lobe receives it (metals
    // have no diffuse). No PI cancel under physical lights.
    const vec3 ambient = albedo * (1.0 - metalness) * lights.ambient;

    // Phase 11: spec IBL via PMREM. We sample the GGX-prefiltered env at a
    // roughness-derived LOD so the contribution naturally fades from a
    // mirror at roughness 0 to fully blurred at roughness 1 — no (1-r)²
    // fade hack. Mip 0 is the source mirror; α=roughness² scales linearly
    // in mip index up to the last (most blurred) mip. As before, we sample
    // unconditionally — visibility-testing env reflections would black out
    // whenever R intersects scene geometry, which is wrong (and breaks
    // metals that have no diffuse to fall back on).
    const vec3 R = reflect(-V, N);
    const float u = 0.5 + atan(R.z, R.x) / TWO_PI;
    const float vv = 0.5 + asin(clamp(R.y, -1.0, 1.0)) / PI;
    const float lodMax  = max(0.0, float(pc.envMipCount) - 1.0);
    const float envLod  = roughness * lodMax;
    const vec3  envProbe = textureLod(envTex, vec2(u, vv), envLod).rgb;
    const vec3  F_macro = fresnelSchlick(NdotV, F0);
    const vec3  envSpec = envProbe * F_macro;

    // Phase 9 v2: probabilistic spec/diffuse lobe selection so polished
    // metals reflect nearby geometry, not just the env probe. p_spec mirrors
    // the WGPU PT (mix(0.5, 0.98, metalness)). The selected lobe's BRDF·cos
    // is divided by its sampling pdf and by p_spec / (1-p_spec) so the
    // estimator stays unbiased.
    //
    //   spec branch — VNDF half-vector → reflect → wi.
    //     BRDF·cos / vndfPdf collapses to F * G1(L);
    //     final weight = F * G1(L) / p_spec.
    //   diff branch — cosine-weighted hemisphere.
    //     BRDF·cos / cosPdf for Lambert collapses to diffuseAlbedo;
    //     final weight = diffuseAlbedo / (1 - p_spec).
    //
    // VNDF samples can fall below the horizon for grazing rays at high
    // roughness; we terminate the path quietly when that happens (acceptable
    // bias for v1; spherical-cap upgrade is a follow-up).
    uint seed = payload.seed;
    const float pSpec = mix(0.5, 0.98, metalness);
    const float xi    = urand(seed);
    const float alpha = roughness * roughness;
    vec3 bounceDir;
    vec3 brdfWeight;
    uint pathFlags = 0u;
    if (xi < pSpec) {
        const vec2 u2 = vec2(urand(seed), urand(seed));
        bounceDir = sampleVNDF(V, N, alpha, u2);
        const float NdotL = dot(N, bounceDir);
        if (NdotL <= 0.0) {
            brdfWeight = vec3(0.0);
            pathFlags |= 1u;// kill path — VNDF sample dipped below horizon
        } else {
            const vec3  H_b   = normalize(V + bounceDir);
            const float VdotH = max(0.0, dot(V, H_b));
            const vec3  F_b   = fresnelSchlick(VdotH, F0);
            const float G1L   = smithG1(NdotL, alpha);
            brdfWeight = F_b * G1L / pSpec;
        }
    } else {
        const vec2 u2 = vec2(urand(seed), urand(seed));
        const vec3 localDir = cosineHemisphere(u2);
        const mat3 tbn = makeTBN(N);
        bounceDir = normalize(tbn * localDir);
        brdfWeight = albedo * (1.0 - metalness) / (1.0 - pSpec);
    }

    const vec3 emissiveOut = mdesc.emissive * mdesc.emissiveIntensity;
    payload.radiance   = emissiveOut + ambient + lit + envSpec;
    payload.brdfWeight = brdfWeight;
    payload.nextOrigin = hitPos + N * 1e-3;
    payload.nextDir    = bounceDir;
    payload.flags      = pathFlags;
    payload.seed       = seed;
}
