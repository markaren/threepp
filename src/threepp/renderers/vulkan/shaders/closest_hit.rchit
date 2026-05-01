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

struct GeometryDesc {
    uint64_t normalAddress;
    uint64_t indexAddress;
    uint     indexed;
    uint     _pad;
};

struct MaterialDesc {
    vec3  albedo;
    float roughness;
    float metalness;
    float _pad0;
    float _pad1;
    float _pad2;
};

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

    const vec3  albedo    = mdesc.albedo;
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

        // Shadow ray: any opaque hit within the scene blocks the light.
        // Default visibility = 0; shadow_miss flips to 1 if the ray escapes.
        shadowVisibility = 0.0;
        traceRayEXT(topAS,
                    gl_RayFlagsOpaqueEXT |
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

    // Phase 7: spec IBL by sampling the equirect env map at the reflection
    // direction directly — no occlusion trace. Visibility-tested env
    // reflection would zero out wherever R hits another scene object, and
    // metals (no diffuse, no ambient) would go pure black. Standard raster
    // / WGPU PT IBL samples unconditionally; we match that here. Without a
    // roughness-prefiltered env we fade the contribution by (1-roughness)^2
    // as a cheap stand-in: perfect mirror at roughness 0, ~zero at
    // roughness 1.
    const vec3 R = reflect(-V, N);
    const float u = 0.5 + atan(R.z, R.x) / TWO_PI;
    const float vv = 0.5 + asin(clamp(R.y, -1.0, 1.0)) / PI;
    const vec3 envProbe = texture(envTex, vec2(u, vv)).rgb;
    const float oneMinusR = 1.0 - roughness;
    const float specEnvFade = oneMinusR * oneMinusR;
    const vec3  F_macro = fresnelSchlick(NdotV, F0);
    const vec3  envSpec = envProbe * F_macro * specEnvFade;

    // Phase 9: sample a cosine-weighted hemisphere direction for the next
    // bounce. BRDF·cos/pdf for Lambert simplifies to `diffuseAlbedo`, which
    // is naturally zero for metals — they exit the bounce loop and rely on
    // direct + envSpec only. (Spec-lobe importance sampling for metal
    // indirect is a v2 follow-up.)
    uint seed = payload.seed;
    const vec2 u01 = vec2(urand(seed), urand(seed));
    const vec3 localDir = cosineHemisphere(u01);
    const mat3 tbn = makeTBN(N);
    const vec3 bounceDir = normalize(tbn * localDir);
    const vec3 diffuseAlbedo = albedo * (1.0 - metalness);

    payload.radiance   = ambient + lit + envSpec;
    payload.brdfWeight = diffuseAlbedo;
    payload.nextOrigin = hitPos + N * 1e-3;
    payload.nextDir    = bounceDir;
    payload.flags      = 0u;
    payload.seed       = seed;
}
