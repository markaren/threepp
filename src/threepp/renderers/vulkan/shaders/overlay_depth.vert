#version 460

// Hybrid raster overlay's UNJITTERED depth prepass. Same scene-geometry
// submit as gbuffer.vert but writes only depth, using cam.currVPunjittered
// instead of cam.currVPjittered. This produces a depth attachment whose
// per-pixel z values exactly match what overlay.vert computes when shading
// wireframe / line overlays — so the overlay pass's depth test compares
// like-with-like and doesn't shimmer between frames.
//
// The G-buffer's own depth attachment is kept jittered (TAA + chit primary
// rely on it being sampled at the same offsets the color attachments were
// rasterized with). This prepass is independent.
//
// Vertex input is just position at location 0. Push constants share the
// raster pipeline's PC layout; only `model` is consumed.

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 currVPjittered;
    mat4 currVPunjittered;
    mat4 prevVP;
    vec4 jitter;
    vec4 prevJitter;
} cam;

layout(push_constant) uniform PC {
    mat4 model;
    uint instanceCustomIndex;
    uint flags;
    uint _pad0;
    uint _pad1;
} pc;

layout(location = 0) in vec3 inPos;

void main() {
    vec4 worldPos = pc.model * vec4(inPos, 1.0);
    vec4 clip     = cam.currVPunjittered * worldPos;
    // Vulkan NDC y-flip — same correction gbuffer.vert applies.
    clip.y        = -clip.y;
    gl_Position   = clip;
}
