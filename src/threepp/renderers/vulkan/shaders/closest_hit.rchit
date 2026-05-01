#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Phase 5b: per-mesh material driving Cook-Torrance (GGX D, Smith G, Schlick F)
// + energy-conserving Lambert diffuse against a single directional light.
// Material data (albedo / roughness / metalness) lives in a `MaterialDesc`
// SSBO indexed by gl_InstanceCustomIndexEXT, parallel to the GeometryDesc
// table from Phase 5a. Lights are still a hard-coded sun; Phase 6 will
// thread real Light objects through.

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

layout(set = 0, binding = 3, scalar) readonly buffer GeomDescBuf {
    GeometryDesc geoms[];
};
layout(set = 0, binding = 4, scalar) readonly buffer MatDescBuf {
    MaterialDesc mats[];
};

hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT vec3 payload;

const float PI = 3.14159265358979;

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
    const vec3 L = normalize(vec3(0.4, 1.0, 0.3));
    const vec3 H = normalize(V + L);

    const float NdotL = max(dot(N, L), 0.0);
    const float NdotV = max(dot(N, V), 0.0);
    const float NdotH = max(dot(N, H), 0.0);
    const float VdotH = max(dot(V, H), 0.0);

    const vec3  albedo    = mdesc.albedo;
    const float roughness = clamp(mdesc.roughness, 0.04, 1.0);
    const float metalness = clamp(mdesc.metalness, 0.0,  1.0);

    const vec3 F0  = mix(vec3(0.04), albedo, metalness);
    const vec3 F   = fresnelSchlick(VdotH, F0);
    const float D  = distGGX(NdotH, roughness);
    const float k  = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    const float G  = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);

    const vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
    const vec3 kd       = (vec3(1.0) - F) * (1.0 - metalness);
    const vec3 diffuse  = kd * albedo / PI;

    // Physical lights (matches three.js useLegacyLights = false): light
    // color is irradiance, no 1/PI cancel. Closest_hit returns linear HDR
    // radiance; raygen handles sRGB encode for display.
    const vec3 lightColor = vec3(3.0);// sun-ish irradiance
    const vec3 lit        = (diffuse + specular) * NdotL * lightColor;
    const vec3 ambient    = albedo * 0.03;

    payload = ambient + lit;
}
