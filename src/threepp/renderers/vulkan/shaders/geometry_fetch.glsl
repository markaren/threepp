// Reading a mesh through its GeometryDesc: the buffer references for the
// vertex streams and the fetches that honour the packed-attribute flags
// (oct-snorm16 normals, unorm16 uvs, uint16 indices). Shared by the
// deferred shade and the probe update; until 2026-10-10 each had its copy.
//
// Needs from the includer: GL_EXT_buffer_reference, the 64-bit integer
// types and GL_EXT_scalar_block_layout. The includer declares its own
// `GeomDescBuf { GeometryDesc geoms[]; }` at whatever binding it uses.

#ifndef GEOMETRY_FETCH_GLSL
#define GEOMETRY_FETCH_GLSL

#include "geometry_desc.glsl"
#include "vulkan_shared.h"// kGeomFlag*, the GeometryDesc.flags bits

layout(buffer_reference, scalar) readonly buffer NormalBuf { float n[]; };
layout(buffer_reference, scalar) readonly buffer IndexBuf  { uint  i[]; };
layout(buffer_reference, scalar) readonly buffer UvBuf     { float u[]; };
layout(buffer_reference, scalar) readonly buffer PackedAttrBuf { uint w[]; };
layout(buffer_reference, scalar) readonly buffer PosBuf    { float p[]; };// GeometryDesc.vertexAddress: float xyz, the buffer the BLAS is built over

// Octahedral decode — mirrors probe_common.glsl octDecode and the CPU encoder
// in VulkanCoreGeometry.cpp (signNotZero convention).
vec3 gfetchOctDecode(vec2 e) {
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        const vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

// Per-vertex fetches that honour the packed bits of GeometryDesc.flags.
vec3 gfetchNormal(in GeometryDesc g, uint vid) {
    if ((g.flags & kGeomFlagPackedNormal) != 0u) {
        return gfetchOctDecode(unpackSnorm2x16(PackedAttrBuf(g.normalAddress).w[vid]));
    }
    NormalBuf nb = NormalBuf(g.normalAddress);
    return vec3(nb.n[vid * 3 + 0], nb.n[vid * 3 + 1], nb.n[vid * 3 + 2]);
}

vec2 gfetchUv(in GeometryDesc g, uint vid) {
    if ((g.flags & kGeomFlagPackedUv) != 0u) {
        return unpackUnorm2x16(PackedAttrBuf(g.uvAddress).w[vid]);
    }
    UvBuf ub = UvBuf(g.uvAddress);
    return vec2(ub.u[vid * 2 + 0], ub.u[vid * 2 + 1]);
}

// Triangle indices, honouring kGeomFlagPackedIndex16 (uint16 indices, two per word).
uvec3 gfetchTri(in GeometryDesc g, int primId) {
    if (g.indexed == 0u)
        return uvec3(primId * 3 + 0, primId * 3 + 1, primId * 3 + 2);
    const uint base = uint(primId) * 3u;
    if ((g.flags & kGeomFlagPackedIndex16) != 0u) {
        PackedAttrBuf ib = PackedAttrBuf(g.indexAddress);
        return uvec3((ib.w[(base + 0u) >> 1u] >> (((base + 0u) & 1u) << 4u)) & 0xFFFFu,
                     (ib.w[(base + 1u) >> 1u] >> (((base + 1u) & 1u) << 4u)) & 0xFFFFu,
                     (ib.w[(base + 2u) >> 1u] >> (((base + 2u) & 1u) << 4u)) & 0xFFFFu);
    }
    IndexBuf ib = IndexBuf(g.indexAddress);
    return uvec3(ib.i[base + 0u], ib.i[base + 1u], ib.i[base + 2u]);
}

#endif// GEOMETRY_FETCH_GLSL
