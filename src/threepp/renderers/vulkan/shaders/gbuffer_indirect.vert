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
    uint64_t colorAddr;        // 0 → no per-vertex color (material.vertexColors off / geometry has no "color")
    uint     instanceCustomIndex;
    uint     flags;            // bits 0..7 render flags | bits 8..15 semantic class id
    uint     indexed;          // 0 / 1
    float    polygonOffset;    // clip-z depth bias (reverse-Z: + = toward near = on top of coplanar geom)
    uint     stableId;         // stable per-object instance id (host-assigned; -> outIds.y)
    uint     packedAttrs;      // BlasRecord::packedMask — bit 0: nrmAddr is oct-snorm16x2
                               // (1 uint/vertex), bit 1: uvAddr is unorm16x2, bit 2:
                               // colorAddr is unorm8x4. 0 → all tightly-packed float.
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
layout(location = 7) out vec3 vColor;// per-vertex color (vec3(1) when no "color" / vertexColors off)
layout(location = 8) flat out uint vStableId;// stable per-object id -> outIds.y
// Per-particle identity -> outIds.w. Zero here: only particlefield_gbuf.vert
// draws particles. Declared because gbuffer.frag is shared between the two.
layout(location = 9) flat out uint vParticleId;

vec3 fetchVec3(uint64_t addr, uint i) {
    FloatBuf b = FloatBuf(addr);
    return vec3(b.v[i * 3u + 0u], b.v[i * 3u + 1u], b.v[i * 3u + 2u]);
}

vec2 fetchVec2(uint64_t addr, uint i) {
    FloatBuf b = FloatBuf(addr);
    return vec2(b.v[i * 2u + 0u], b.v[i * 2u + 1u]);
}

// Octahedral decode — MUST mirror probe_common.glsl's octDecode (and the CPU
// encoder in VulkanCoreGeometry.cpp) including the signNotZero convention.
vec3 octDecodeN(vec2 e) {
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        const vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

// Packed-aware fetches (DrawInfo.packedAttrs bits, see BlasRecord::packedMask).
vec3 fetchNormal(uint64_t addr, uint packedAttrs, uint i) {
    if ((packedAttrs & 1u) != 0u) {
        return octDecodeN(unpackSnorm2x16(UintBuf(addr).v[i]));
    }
    return fetchVec3(addr, i);
}

vec2 fetchUv(uint64_t addr, uint packedAttrs, uint i) {
    if ((packedAttrs & 2u) != 0u) {
        return unpackUnorm2x16(UintBuf(addr).v[i]);
    }
    return fetchVec2(addr, i);
}

vec3 fetchColor(uint64_t addr, uint packedAttrs, uint i) {
    if ((packedAttrs & 4u) != 0u) {
        return unpackUnorm4x8(UintBuf(addr).v[i]).rgb;
    }
    return fetchVec3(addr, i);
}

void main() {
    const DrawInfo d = draws[gl_InstanceIndex];

    // Indexed meshes: gl_VertexIndex is the index-buffer cursor (0..indexCount-1);
    // resolve to the real vertex ID by reading from the bindless index buffer.
    // Non-indexed: gl_VertexIndex IS the vertex ID directly. packedAttrs bit 3:
    // the index buffer is uint16, two indices per word.
    uint vid;
    if (d.indexed == 0u) {
        vid = uint(gl_VertexIndex);
    } else if ((d.packedAttrs & 8u) != 0u) {
        const uint k = uint(gl_VertexIndex);
        vid = (UintBuf(d.indexAddr).v[k >> 1u] >> ((k & 1u) << 4u)) & 0xFFFFu;
    } else {
        vid = UintBuf(d.indexAddr).v[uint(gl_VertexIndex)];
    }

    const vec3 inPos    = fetchVec3(d.posAddr, vid);// positions are always float
    const vec3 inNormal = (d.nrmAddr != 0ul) ? fetchNormal(d.nrmAddr, d.packedAttrs, vid) : vec3(0.0, 1.0, 0.0);
    const vec2 inUv     = (d.uvAddr  != 0ul) ? fetchUv(d.uvAddr, d.packedAttrs, vid) : vec2(0.0);
    // prevPos: for static meshes the host sets prevPosAddr == posAddr so this
    // collapses to motionMat-only motion; skinned / displaced meshes have
    // their own prev-pose buffer captured at the end of the previous frame.
    const vec3 inPrevPos = fetchVec3(d.prevPosAddr != 0ul ? d.prevPosAddr : d.posAddr, vid);

    const vec4 worldPos     = d.model * vec4(inPos,     1.0);
    const vec4 prevWorldPos = motionMat[d.instanceCustomIndex] * d.model * vec4(inPrevPos, 1.0);

    // Cofactor = det·(inverse-transpose): correct normals under non-uniform
    // scale, normalize-safe, no per-vertex inverse (see gbuffer.vert).
    const mat3 nm = mat3(d.model);
    vWorldNormal = mat3(cross(nm[1], nm[2]), cross(nm[2], nm[0]), cross(nm[0], nm[1])) * inNormal;

    vCurrClipUnjit = cam.currVPunjittered * worldPos;
    vPrevClip      = cam.prevVP           * prevWorldPos;
    vInstanceIdx   = d.instanceCustomIndex;
    vFlags         = d.flags;// render flags in bits 0..7 | class id in bits 8..15
    vStableId      = d.stableId;
    vParticleId    = 0u;
    vUv            = inUv;
    vWorldPos      = worldPos.xyz;
    // Per-vertex color (material.vertexColors). gbuffer.frag multiplies albedo
    // by this; white when the mesh has no "color" attribute so the multiply is
    // a no-op. Linear working space — matches the material albedo.
    vColor         = (d.colorAddr != 0ul) ? fetchColor(d.colorAddr, d.packedAttrs, vid) : vec3(1.0);

    gl_Position    = cam.currVPjittered * worldPos;
    // Per-mesh polygon offset (decals): bias clip-z so the fragment's NDC depth
    // shifts toward NEAR (reverse-Z) → wins the depth test against coplanar
    // geometry and renders on top, no z-fighting. 0 for normal meshes.
    gl_Position.z += d.polygonOffset * gl_Position.w;
    gl_Position.y  = -gl_Position.y;
}
