#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout    : require
#extension GL_EXT_buffer_reference       : require
#extension GL_EXT_buffer_reference2      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Shadow-map caster (ShadowMapPass): gbuffer_indirect.vert reduced to the
// position and UV pull. The same DrawInfo records and indirect commands as
// the G-buffer (built by buildShadowDrawData), the light's clip matrix in the
// push constants' model slot (the indirect path never read it there).

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer FloatBuf { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer UintBuf  { uint  v[]; };

struct DrawInfo {
    mat4     model;
    uint64_t posAddr;
    uint64_t nrmAddr;
    uint64_t uvAddr;
    uint64_t prevPosAddr;
    uint64_t indexAddr;
    uint64_t colorAddr;
    uint     instanceCustomIndex;
    uint     flags;
    uint     indexed;
    float    polygonOffset;
    uint     stableId;
    uint     packedAttrs;
};
layout(set = 0, binding = 4, scalar) readonly buffer DrawInfoBuf {
    DrawInfo draws[];
};

layout(push_constant) uniform Pc {
    mat4  lightVP;// world -> the map's clip space, reverse-Z
    uvec4 info;   // x bit 0: a point light's cube face (its matrix carries the y flip)
} pc;

layout(location = 0) out vec2 vUv;
layout(location = 1) flat out uint vInstanceIdx;

#include "vertex_fetch.glsl"

void main() {
    const DrawInfo d = draws[gl_InstanceIndex];
    uint vid;
    if (d.indexed == 0u) {
        vid = uint(gl_VertexIndex);
    } else if ((d.packedAttrs & kPackedIndex16) != 0u) {
        const uint k = uint(gl_VertexIndex);
        vid = (UintBuf(d.indexAddr).v[k >> 1u] >> ((k & 1u) << 4u)) & 0xFFFFu;
    } else {
        vid = UintBuf(d.indexAddr).v[uint(gl_VertexIndex)];
    }
    vUv          = (d.uvAddr != 0ul) ? fetchUv(d.uvAddr, d.packedAttrs, vid) : vec2(0.0);
    vInstanceIdx = d.instanceCustomIndex;
    gl_Position  = pc.lightVP * (d.model * vec4(fetchVec3(d.posAddr, vid), 1.0));
    // The raster's y flip, so a 2D map's rows run as its v coordinate does
    // (shadow_maps.glsl: v = 0.5 - 0.5 * clip.y).
    if ((pc.info.x & 1u) == 0u) gl_Position.y = -gl_Position.y;
}
