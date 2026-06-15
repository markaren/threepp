#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable

#include "vulkan_shared.h"

// Photon closest-hit: fills PhotonPayload with surface info so photon_emit.rgen
// can decide whether to refract, reflect, deposit, or discard the photon path.

layout(buffer_reference, scalar) readonly buffer VertexBuf { float p[]; };
layout(buffer_reference, scalar) readonly buffer IndexBuf  { uint  i[]; };

struct GeometryDesc {
    uint64_t vertexAddress;
    uint64_t normalAddress;
    uint64_t indexAddress;
    uint64_t uvAddress;
    uint64_t foamAddress;// unused here, kept for layout match with closest_hit.rchit
    uint64_t prevVertexAddress;// unused here, kept for layout match
    uint64_t colorAddress;// unused here, kept for layout match with closest_hit.rchit
    uint     indexed;
    uint     _pad;
};

// MaterialDesc comes from vulkan_shared.h. This shader only reads albedo,
// roughness, metalness, transmission, and ior; the rest of the struct is
// declared so mats[i] uses the correct stride without a hand-padded mirror.

layout(set = 0, binding = 3, scalar) readonly buffer GeomDescBuf { GeometryDesc geoms[]; };
layout(set = 0, binding = 4, scalar) readonly buffer MatDescBuf  { MaterialDesc mats[];  };

struct PhotonPayload {
    vec3  pos;
    vec3  normal;
    float roughness;
    float metalness;
    float transmission;
    float ior;
    uint  hitValid;
};

layout(location = 2) rayPayloadInEXT PhotonPayload pp;
hitAttributeEXT vec2 bary;

void main() {
    const GeometryDesc g = geoms[gl_InstanceCustomIndexEXT];
    uvec3 idx;
    if (g.indexed != 0u) {
        IndexBuf ib = IndexBuf(g.indexAddress);
        idx = uvec3(ib.i[gl_PrimitiveID * 3],
                    ib.i[gl_PrimitiveID * 3 + 1],
                    ib.i[gl_PrimitiveID * 3 + 2]);
    } else {
        idx = uvec3(gl_PrimitiveID * 3,
                    gl_PrimitiveID * 3 + 1,
                    gl_PrimitiveID * 3 + 2);
    }
    VertexBuf vb = VertexBuf(g.vertexAddress);
    const vec3 v0 = vec3(vb.p[idx.x * 3], vb.p[idx.x * 3 + 1], vb.p[idx.x * 3 + 2]);
    const vec3 v1 = vec3(vb.p[idx.y * 3], vb.p[idx.y * 3 + 1], vb.p[idx.y * 3 + 2]);
    const vec3 v2 = vec3(vb.p[idx.z * 3], vb.p[idx.z * 3 + 1], vb.p[idx.z * 3 + 2]);
    const float w = 1.0 - bary.x - bary.y;
    pp.pos = (gl_ObjectToWorldEXT * vec4(w * v0 + bary.x * v1 + bary.y * v2, 1.0)).xyz;

    const vec3 localN = normalize(cross(v1 - v0, v2 - v0));
    // Raw geometric outward normal — NOT flipped toward the ray.
    // photon_emit.rgen detects entry vs. exit via dot(dir, N) and sets
    // eta = 1/ior (entering) or ior (exiting) accordingly.
    pp.normal = normalize((gl_ObjectToWorldEXT * vec4(localN, 0.0)).xyz);

    const MaterialDesc m = mats[gl_InstanceCustomIndexEXT];
    pp.roughness    = m.roughness;
    pp.metalness    = m.metalness;
    pp.transmission = m.transmission;
    pp.ior          = m.ior;
    pp.hitValid     = 1u;
}
