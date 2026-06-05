#version 460
#extension GL_EXT_scalar_block_layout    : require
#extension GL_EXT_buffer_reference       : require
#extension GL_EXT_buffer_reference2      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Indirect-drawing variant of gbuffer.vert. Replaces fixed-function vertex
// input bindings with bindless pulls from per-mesh buffer device addresses,
// so the whole G-buffer pass can ship as 1–3 vkCmdDrawIndirect calls
// (one per cull mode) instead of N vkCmdDrawIndexed calls. That cuts the
// GPU command-processor overhead from ~15 µs/draw × N down to a near-fixed
// cost regardless of mesh count — see the perf notes around
// recordRasterGbufPass.
//
// Per-draw data (model matrix + buffer addresses + flags) is read from
// the binding-4 DrawInfo SSBO. The global DrawInfo index for each draw
// is encoded into VkDrawIndirectCommand.firstInstance, surfaced here as
// gl_InstanceIndex (since each draw runs exactly one instance, gl_-
// InstanceIndex == firstInstance throughout the draw). gl_VertexIndex
// runs 0..vertexCount-1; for indexed meshes we fetch the real vertex ID
// from the bindless index buffer manually so the pipeline can declare
// zero vertex input bindings.

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 currVPjittered;
    mat4 currVPunjittered;
    mat4 prevVP;
    vec4 jitter;
    vec4 prevJitter;
} cam;

layout(set = 0, binding = 1, scalar) readonly buffer MotionMatBuf {
    mat4 motionMat[];
};

// Bindless attribute buffers. `scalar` layout is required so vec3 reads
// pack tightly (no GLSL std140/std430 padding-to-vec4 surprises).
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer FloatBuf { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer UintBuf  { uint  v[]; };

struct DrawInfo {
    mat4     model;
    uint64_t posAddr;
    uint64_t nrmAddr;
    uint64_t uvAddr;
    uint64_t prevPosAddr;
    uint64_t indexAddr;        // 0 → non-indexed (gl_VertexIndex IS the vertex ID)
    uint     instanceCustomIndex;
    uint     flags;
    uint     indexed;          // 0 / 1
    float    polygonOffset;    // clip-z depth bias (reverse-Z: + = toward near = on top of coplanar geom)
};
layout(set = 0, binding = 4, scalar) readonly buffer DrawInfoBuf {
    DrawInfo draws[];
};

layout(location = 0) out vec3 vWorldNormal;
layout(location = 1) out vec4 vCurrClipUnjit;
layout(location = 2) out vec4 vPrevClip;
layout(location = 3) flat out uint vInstanceIdx;
layout(location = 4) flat out uint vFlags;
layout(location = 5) out vec2 vUv;
layout(location = 6) out vec3 vWorldPos;

vec3 fetchVec3(uint64_t addr, uint i) {
    FloatBuf b = FloatBuf(addr);
    return vec3(b.v[i * 3u + 0u], b.v[i * 3u + 1u], b.v[i * 3u + 2u]);
}

vec2 fetchVec2(uint64_t addr, uint i) {
    FloatBuf b = FloatBuf(addr);
    return vec2(b.v[i * 2u + 0u], b.v[i * 2u + 1u]);
}

void main() {
    const DrawInfo d = draws[gl_InstanceIndex];

    // Indexed meshes: gl_VertexIndex is the index-buffer cursor (0..indexCount-1);
    // resolve to the real vertex ID by reading from the bindless index buffer.
    // Non-indexed: gl_VertexIndex IS the vertex ID directly.
    const uint vid = (d.indexed != 0u)
            ? UintBuf(d.indexAddr).v[uint(gl_VertexIndex)]
            : uint(gl_VertexIndex);

    const vec3 inPos    = fetchVec3(d.posAddr, vid);
    const vec3 inNormal = (d.nrmAddr != 0ul) ? fetchVec3(d.nrmAddr, vid) : vec3(0.0, 1.0, 0.0);
    const vec2 inUv     = (d.uvAddr  != 0ul) ? fetchVec2(d.uvAddr,  vid) : vec2(0.0);
    // prevPos: for static meshes the host sets prevPosAddr == posAddr so this
    // collapses to motionMat-only motion; skinned / displaced meshes have
    // their own prev-pose buffer captured at the end of the previous frame.
    const vec3 inPrevPos = fetchVec3(d.prevPosAddr != 0ul ? d.prevPosAddr : d.posAddr, vid);

    const vec4 worldPos     = d.model * vec4(inPos,     1.0);
    const vec4 prevWorldPos = motionMat[d.instanceCustomIndex] * d.model * vec4(inPrevPos, 1.0);

    vWorldNormal = mat3(d.model) * inNormal;

    vCurrClipUnjit = cam.currVPunjittered * worldPos;
    vPrevClip      = cam.prevVP           * prevWorldPos;
    vInstanceIdx   = d.instanceCustomIndex;
    vFlags         = d.flags;
    vUv            = inUv;
    vWorldPos      = worldPos.xyz;

    gl_Position    = cam.currVPjittered * worldPos;
    // Per-mesh polygon offset (decals): bias clip-z so the fragment's NDC depth
    // shifts toward NEAR (reverse-Z) → wins the depth test against coplanar
    // geometry and renders on top, no z-fighting. 0 for normal meshes.
    gl_Position.z += d.polygonOffset * gl_Position.w;
    gl_Position.y  = -gl_Position.y;
}
