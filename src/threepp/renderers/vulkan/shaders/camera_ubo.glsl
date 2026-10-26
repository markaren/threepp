// The shade-side camera block, set 0 binding 0: what every compute and ray
// pass that reconstructs world positions from the G-buffer reads. The host
// fills it in updateCameraUbo (VulkanCoreUploads.cpp): two mat4 back to back,
// then jitter, then camAux; std140, 160 bytes. This is the ONE declaration.
// A shader that needs the camera includes this file instead of re-declaring
// a subset of it, so a member added here reaches every reader at once.
// (Until 2026-10-10 eleven shaders carried their own copies of this block,
// some without camAux, each correct only because it was a prefix of the
// real layout.)
//
// The raster passes' camera (jittered and unjittered VP, previous VP) is a
// different block with a different layout: raster_camera_ubo.glsl.

#ifndef CAMERA_UBO_GLSL
#define CAMERA_UBO_GLSL

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 viewInverse;
    mat4 projInverse;// clip -> view, reverse-Z (matches the depth the raster wrote)
    vec4 jitter;     // .xy = clip-space sub-pixel offset matching the raster's
                     // Halton sequence, .zw = 1 / resolution
    vec4 camAux;     // .x = 1 for a parallel (orthographic) projection, .yzw =
                     // the camera's world forward; camera_ray.glsl builds every
                     // primary ray from these, froxelWorldRay reads .x
} cam;

// View-space distance from reversed-Z NDC depth — only the projection
// inverse's z/w rows matter (xy jitter never feeds them). The DoF passes
// (dof_coc.comp, dof_combine.comp) linearize their depth reads with it.
float viewDepth(float zNdc) {
    const float vz = cam.projInverse[2][2] * zNdc + cam.projInverse[3][2];
    const float vw = cam.projInverse[2][3] * zNdc + cam.projInverse[3][3];
    return -(vz / vw);
}

#endif// CAMERA_UBO_GLSL
