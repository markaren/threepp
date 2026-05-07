#version 460

// Hybrid-mode raster overlay (wireframe + line) vertex shader. The pipeline
// runs after the PT denoise + TAA composite and writes onto the swapchain
// directly, depth-tested against the G-buffer depth attachment so overlays
// are correctly occluded by path-traced surfaces. Sub-pixel-stable: the MVP
// uses the *unjittered* camera matrices to match the post-TAA raster
// position, otherwise wireframes shimmer one pixel per frame.
//
// Push-constant layout matches overlay.frag's; CPU-side computes
// mvp = projUnjittered · viewInverse · model and packs it.

layout(location = 0) in vec3 inPos;

layout(push_constant) uniform Pc {
    mat4 mvp;     // viewProj_unjit · model
    vec4 color;   // overlay tint (xyz = linear RGB, w = unused)
} pc;

void main() {
    vec4 clip = pc.mvp * vec4(inPos, 1.0);
    // threepp's projection matrix follows the GL convention (Y up in NDC).
    // Vulkan NDC has Y pointing down, so we negate Y at the gl_Position
    // boundary — same correction gbuffer.vert applies. Without this the
    // overlay is vertically mirrored, which combined with depth-test
    // failures from the inverted positions can look like the geometry is
    // "rotating with the camera" (the user's reported symptom).
    clip.y = -clip.y;
    gl_Position = clip;
}
