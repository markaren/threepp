#version 460
#extension GL_EXT_scalar_block_layout : require

// Raster G-buffer prepass. Produces depth, world-space normal,
// screen-space motion vector and per-pixel instance/flags for
// deferred_shade.comp to shade analytically, using exact raster primary
// visibility instead of a traced primary ray. AA happens in raster (TAA)
// — that's what eliminates the moving-object shake under continuous
// camera motion.
//
// Vertex / normal inputs are bound from the same device buffers that BLAS
// reads (VERTEX_BUFFER_BIT was added at allocation; see VulkanRenderer.cpp).
// No upload duplication; the raster prepass and ray-query shadow/reflection
// rays warm the same cache lines.

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 currVPjittered;  // for gl_Position; primary AA jitter applied here
    mat4 currVPunjittered;// for motion-vec; must match prev's projection family
    mat4 prevVP;          // previous-frame view-proj, unjittered
    vec4 jitter;          // .xy = jitter offset in clip-space (sub-texel), .zw = 1/resolution
    vec4 prevJitter;      // .xy = previous frame's jitter offset. NOTE: the motion vec
                          // stays JITTER-FREE (gbuffer.frag writes clean prevNDC−currNDC;
                          // a (prev−curr) delta was tested there and rejected). Kept for
                          // the deferred shade's hybrid reproject tap correction; .z
                          // smuggles the normal-map Toksvig toggle.
} cam;

// motionMat[i] = prev_world_i * inverse(current_world_i). Apply to a
// current-frame world-space point to get its previous-frame world position.
// Bound here under the raster prepass's own descriptor set, independent of
// other pipelines' layouts.
layout(set = 0, binding = 1, scalar) readonly buffer MotionMatBuf {
    mat4 motionMat[];
};

layout(push_constant) uniform PC {
    mat4 model;
    uint instanceCustomIndex;
    uint flags;            // bit 0 is_water, bit 1 transmissive, bit 2 thinWalled
    uint _pad0;
    uint _pad1;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;// passthrough for deferred_shade.comp texture sampling
// Previous-frame local-space vertex position. For SkinnedMesh + DisplacedMesh
// the host maintains a separate prev-pose buffer; for static meshes the host
// binds rec->vertex here so inPrevPos == inPos and motion reduces to the
// rigid-body case via motionMat[i] alone (same as before this change).
layout(location = 3) in vec3 inPrevPos;

layout(location = 0) out vec3 vWorldNormal;
layout(location = 1) out vec4 vCurrClipUnjit;// motion-vec source — must not include jitter
layout(location = 2) out vec4 vPrevClip;
layout(location = 3) flat out uint vInstanceIdx;
layout(location = 4) flat out uint vFlags;
layout(location = 5) out vec2 vUv;
layout(location = 6) out vec3 vWorldPos;// for fragment-shader TBN via dFdx/dFdy
layout(location = 7) out vec3 vColor;// per-vertex color, white here. NOT a parity hole:
                                     // this fixed-input pipeline is created but never
                                     // bound for draws — every mesh rasterizes through
                                     // gbuffer_indirect.vert, which fetches the real
                                     // "color" attribute (DrawInfo.colorAddr). This
                                     // shader only keeps the shared gbuffer.frag
                                     // interface satisfied.
layout(location = 8) flat out uint vStableId;// stable per-object id — matches the shared
                                             // gbuffer.frag interface. This fixed path is not
                                             // used for the ids pass (indirect draws are), so
                                             // it just mirrors the visible index as before.
// Per-particle identity -> outIds.w, written by particlefield_gbuf.vert alone.
// Declared here because gbuffer.frag is SHARED: a fragment input with no
// matching vertex output is an interface mismatch, so this path publishes the
// same "no particle" zero the channel always carried.
layout(location = 9) flat out uint vParticleId;

void main() {
    vec4 worldPos     = pc.model * vec4(inPos, 1.0);
    // prev_world = (motionMat * curr_model) * prev_local_pos = prev_model * inPrevPos.
    //   Static meshes:    inPrevPos == inPos     → equivalent to motionMat * worldPos.
    //   Skinned/displaced: inPrevPos = prev pose → captures the per-vertex
    //                                              deformation that motionMat alone
    //                                              (identity for these meshes since
    //                                              the rigid world matrix doesn't
    //                                              change) would miss.
    vec4 prevWorldPos = motionMat[pc.instanceCustomIndex] * pc.model * vec4(inPrevPos, 1.0);

    // Normals need the inverse-transpose of the model's linear part under
    // non-uniform scale. Cofactor form: cof(M) = det(M)·M⁻ᵀ — the det scale
    // washes out in the frag's normalize, and three cross products beat a
    // per-vertex inverse(). Matches the ray-query side's transpose(worldToObj).
    const mat3 m = mat3(pc.model);
    vWorldNormal = mat3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1])) * inNormal;

    vCurrClipUnjit = cam.currVPunjittered * worldPos;
    vPrevClip      = cam.prevVP           * prevWorldPos;
    vInstanceIdx   = pc.instanceCustomIndex;
    vFlags         = pc.flags;
    vUv            = inUv;
    vWorldPos      = worldPos.xyz;
    vColor         = vec3(1.0);// interface filler — this pipeline never draws (see decl)
    vStableId      = pc.instanceCustomIndex + 1u;// fixed path: mirror .x (unused for ids)
    vParticleId    = 0u;

    // threepp's projection matrix follows the GL convention (Y up in NDC).
    // Vulkan NDC has Y pointing down, so we negate Y at the gl_Position
    // boundary. vCurrClipUnjit / vPrevClip are kept in GL convention so
    // motion vectors stay self-consistent (taa_resolve.comp Y-flips on read).
    gl_Position    = cam.currVPjittered * worldPos;
    gl_Position.y  = -gl_Position.y;
}
