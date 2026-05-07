#version 460
#extension GL_EXT_scalar_block_layout : require

// Hybrid raster G-buffer prepass. Produces depth, world-space normal,
// screen-space motion vector and per-pixel instance/flags so the PT raygen
// can skip its primary BVH trace and continue from bounce 1 using exact
// raster primary visibility. AA happens in raster (TAA), not as Monte
// Carlo on the PT primary — that's what eliminates the moving-object
// shake under continuous camera motion.
//
// Vertex / normal inputs are bound from the same device buffers that BLAS
// reads (VERTEX_BUFFER_BIT was added at allocation; see VulkanRenderer.cpp).
// No upload duplication; raster prepass and PT shadow rays warm the same
// cache lines.

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 currVPjittered;  // for gl_Position; primary AA jitter applied here
    mat4 currVPunjittered;// for motion-vec; must match prev's projection family
    mat4 prevVP;          // previous-frame view-proj, unjittered
    vec4 jitter;          // .xy = jitter offset in clip-space (sub-texel), .zw = 1/resolution
} cam;

// motionMat[i] = prev_world_i * inverse(current_world_i). Apply to a
// current-frame world-space point to get its previous-frame world position.
// Same physical buffer as raygen.rgen's binding 10 — bound here under
// raster's own descriptor set so the layouts stay independent.
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
layout(location = 2) in vec2 inUv;// passthrough for raygen texture sampling
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

    // mat3(model) is correct for normals only under uniform/no-shear scale.
    // threepp's scene graphs are typically uniform-scaled; non-uniform
    // scaled meshes would need transpose(inverse(mat3(model))) here.
    vWorldNormal = mat3(pc.model) * inNormal;

    vCurrClipUnjit = cam.currVPunjittered * worldPos;
    vPrevClip      = cam.prevVP           * prevWorldPos;
    vInstanceIdx   = pc.instanceCustomIndex;
    vFlags         = pc.flags;
    vUv            = inUv;
    vWorldPos      = worldPos.xyz;

    // threepp's projection matrix follows the GL convention (Y up in NDC).
    // Vulkan NDC has Y pointing down, so we negate Y at the gl_Position
    // boundary. vCurrClipUnjit / vPrevClip are kept in GL convention so
    // motion vectors stay self-consistent (raygen will Y-flip on read).
    gl_Position    = cam.currVPjittered * worldPos;
    gl_Position.y  = -gl_Position.y;
}
