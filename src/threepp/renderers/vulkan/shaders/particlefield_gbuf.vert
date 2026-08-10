#version 460
#extension GL_EXT_scalar_block_layout    : require
#extension GL_EXT_buffer_reference       : require
#extension GL_EXT_buffer_reference2      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// ParticleField G-buffer vertex stage — the mesh representation of
// threepp::ParticleField (plans/particle-field.md §3.1).
//
// A SIBLING of gbuffer_indirect.vert, not a modification of it. That file
// documents an invariant this one deliberately does not share: "each draw
// runs exactly one instance, gl_InstanceIndex == firstInstance". A field is
// ONE indirect draw whose instanceCount is the GPU-side live count, so here
//
//     gl_InstanceIndex == the PARTICLE index      (firstInstance is 0)
//     the DrawInfo index comes from a push constant
//
// and the two shaders must therefore stay separate pipelines over the same
// render pass. Everything else — the bindless vertex pull, the DrawInfo
// contents, the cofactor normal, the motion-vector composition, the reverse-Z
// polygon offset — is copied verbatim so a particle shades identically to the
// InstancedMesh it replaces.
//
// NO EXPANSION PASS. The per-particle world transform is composed INLINE from
// positions[i] and the (write-once) orientation quaternion, and the PREVIOUS
// transform is composed the same way from prevPositions[i] with the SAME
// quaternion — which is what makes correct motion vectors possible without a
// materialised mat4 per particle. plans/particle-field.md §3.1 leaves that as
// a measurement; it comes out in favour of inline, so particle_expand.comp
// does not exist.

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

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer FloatBuf { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer UintBuf  { uint  v[]; };
// ParticlePos / physx::PxVec4 / GLSL vec4 under scalar layout are the same
// 16 bytes — the identity the whole buffer contract is built on.
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer PosBuf { vec4 v[]; };
// Orientations: quaternion as snorm16x4, 8 B, written once at slot claim.
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer OriBuf { uvec2 v[]; };

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

// Per-field. The addresses ride here rather than in the FieldDesc SSBO so the
// pass needs NO descriptor of its own: a descriptor write per field per frame
// is exactly the VUID-03047 exposure this phase is supposed to avoid, and a
// push constant costs nothing and cannot be stale.
layout(push_constant, scalar) uniform PfPush {
    uint64_t posAddr;      // ParticlePos[capacity], this frame's ring slot
    uint64_t prevPosAddr;  // previous frame's positions (== posAddr → no motion)
    uint64_t oriAddr;      // 0 → identity orientation for every particle
    uint     drawIdx;      // index into draws[] — the field's ONE DrawInfo
    uint     wSemantic;    // 0 = InvMass (proxy carries the size), 1 = Radius
    float    invUniformRadius;// 1 / Config::uniformRadius (wSemantic == 1 only)
    float    _pad;
} pf;

layout(location = 0) out vec3 vWorldNormal;
layout(location = 1) out vec4 vCurrClipUnjit;
layout(location = 2) out vec4 vPrevClip;
layout(location = 3) flat out uint vInstanceIdx;
layout(location = 4) flat out uint vFlags;
layout(location = 5) out vec2 vUv;
layout(location = 6) out vec3 vWorldPos;
layout(location = 7) out vec3 vColor;
layout(location = 8) flat out uint vStableId;
// Per-PARTICLE identity → outIds.w. Every other vertex stage writes 0 there
// (the channel was reserved); this is the only producer.
layout(location = 9) flat out uint vParticleId;

vec3 fetchVec3(uint64_t addr, uint i) {
    FloatBuf b = FloatBuf(addr);
    return vec3(b.v[i * 3u + 0u], b.v[i * 3u + 1u], b.v[i * 3u + 2u]);
}

vec2 fetchVec2(uint64_t addr, uint i) {
    FloatBuf b = FloatBuf(addr);
    return vec2(b.v[i * 2u + 0u], b.v[i * 2u + 1u]);
}

// MUST mirror gbuffer_indirect.vert's octDecodeN (and probe_common.glsl, and
// the CPU encoder) including the signNotZero convention.
vec3 octDecodeN(vec2 e) {
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        const vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

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

mat3 quatToMat3(vec4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x * x2, xy = x * y2, xz = x * z2;
    const float yy = y * y2, yz = y * z2, zz = z * z2;
    const float wx = w * x2, wy = w * y2, wz = w * z2;
    // Column-major, matching Matrix4::makeRotationFromQuaternion — the
    // orientation is authored on the host as a quaternion and the InstancedMesh
    // path it replaces baked the same rotation into its 3x3.
    return mat3(vec3(1.0 - (yy + zz), xy + wz,         xz - wy),
                vec3(xy - wz,         1.0 - (xx + zz), yz + wx),
                vec3(xz + wy,         yz - wx,         1.0 - (xx + yy)));
}

void main() {
    const DrawInfo d = draws[pf.drawIdx];
    const uint pi = uint(gl_InstanceIndex);// firstInstance is 0 for this draw

    const vec4 P = PosBuf(pf.posAddr).v[pi];
    // Dead slot. NOT written as (P.w < 0.0): a NaN fails every comparison, so
    // the negated form catches garbage as dead too, which matters because a
    // slot past the sim's high-water mark holds whatever the allocator left.
    const bool dead = !(P.w >= 0.0);

    // Previous position for the motion vector. A slot that was dead last frame
    // and is alive now has no previous world point at all — reuse the current
    // one so it reprojects to itself (zero motion) instead of streaking in
    // from a stale location, the same choice prevVertex makes for a deformer
    // whose topology changed.
    vec4 Pp = PosBuf(pf.prevPosAddr).v[pi];
    if (!(Pp.w >= 0.0)) Pp = P;

    // Scale. Under WSemantic::InvMass the proxy geometry is authored at the
    // particle's world size and w is PhysX's inverse mass, which says nothing
    // about size — so the scale is 1. Under WSemantic::Radius the proxy is
    // authored at Config::uniformRadius and each particle scales relative to
    // it. Documented on ParticleField::Config::uniformRadius.
    const float s = (pf.wSemantic == 1u) ? (P.w * pf.invUniformRadius) : 1.0;

    mat3 R = mat3(1.0);
    if (pf.oriAddr != 0ul) {
        const uvec2 packed = OriBuf(pf.oriAddr).v[pi];
        R = quatToMat3(vec4(unpackSnorm2x16(packed.x), unpackSnorm2x16(packed.y)));
    }

    // ── Vertex pull, identical to gbuffer_indirect.vert ─────────────────────
    uint vid;
    if (d.indexed == 0u) {
        vid = uint(gl_VertexIndex);
    } else if ((d.packedAttrs & 8u) != 0u) {
        const uint k = uint(gl_VertexIndex);
        vid = (UintBuf(d.indexAddr).v[k >> 1u] >> ((k & 1u) << 4u)) & 0xFFFFu;
    } else {
        vid = UintBuf(d.indexAddr).v[uint(gl_VertexIndex)];
    }

    const vec3 inPos    = fetchVec3(d.posAddr, vid);
    const vec3 inNormal = (d.nrmAddr != 0ul) ? fetchNormal(d.nrmAddr, d.packedAttrs, vid) : vec3(0.0, 1.0, 0.0);
    const vec2 inUv     = (d.uvAddr  != 0ul) ? fetchUv(d.uvAddr, d.packedAttrs, vid) : vec2(0.0);

    // Dead → every vertex of the proxy collapses onto the particle centre, so
    // the triangle has exactly zero area and covers no sample. Same idiom, and
    // the same cost (none), as GrainField zeroing its instance 3x3.
    const vec3 localOff = dead ? vec3(0.0) : (R * (inPos * s));

    const vec4 worldPos     = d.model * vec4(P.xyz  + localOff, 1.0);
    const vec4 prevWorldPos = motionMat[d.instanceCustomIndex] *
                              d.model * vec4(Pp.xyz + localOff, 1.0);

    const mat3 nm = mat3(d.model) * R;
    vWorldNormal = mat3(cross(nm[1], nm[2]), cross(nm[2], nm[0]), cross(nm[0], nm[1])) * inNormal;

    vCurrClipUnjit = cam.currVPunjittered * worldPos;
    vPrevClip      = cam.prevVP           * prevWorldPos;
    vInstanceIdx   = d.instanceCustomIndex;
    vFlags         = d.flags;
    vStableId      = d.stableId;
    vParticleId    = pi & 0xFFFFu;// outIds is rgba16ui — 16 bits is the channel
    vUv            = inUv;
    vWorldPos      = worldPos.xyz;
    vColor         = (d.colorAddr != 0ul) ? fetchColor(d.colorAddr, d.packedAttrs, vid) : vec3(1.0);

    gl_Position    = cam.currVPjittered * worldPos;
    gl_Position.z += d.polygonOffset * gl_Position.w;
    gl_Position.y  = -gl_Position.y;
}
