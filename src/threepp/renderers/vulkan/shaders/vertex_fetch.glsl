// Vertex attributes pulled by device address for the bindless G-buffer
// vertex stages (gbuffer_indirect.vert, particlefield_gbuf.vert), honouring
// DrawInfo.packedAttrs: bit 0 oct-snorm16 normals, bit 1 unorm16 uvs, bit 2
// unorm8 colours. Until 2026-10-10 both stages carried their own copies.
//
// Needs from the includer: the FloatBuf / UintBuf buffer references and
// the 64-bit integer types.

#ifndef VERTEX_FETCH_GLSL
#define VERTEX_FETCH_GLSL

vec3 fetchVec3(uint64_t addr, uint i) {
    FloatBuf b = FloatBuf(addr);
    return vec3(b.v[i * 3u + 0u], b.v[i * 3u + 1u], b.v[i * 3u + 2u]);
}

vec2 fetchVec2(uint64_t addr, uint i) {
    FloatBuf b = FloatBuf(addr);
    return vec2(b.v[i * 2u + 0u], b.v[i * 2u + 1u]);
}

// Octahedral decode — MUST mirror probe_common.glsl's octDecode, gfetchOctDecode
// (geometry_fetch.glsl) and the CPU encoder in VulkanCoreGeometry.cpp, including
// the signNotZero convention.
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

#endif// VERTEX_FETCH_GLSL
