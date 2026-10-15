// The raster-side camera block: what the G-buffer vertex and fragment
// stages, the overlay depth pass, the particle-field G-buffer pass and the
// occlusion cull read. Mirrors RasterCameraData in VulkanViewContext.hpp,
// filled by uploadRasterCameraUbo: three mat4, then two vec4; std140, 224
// bytes. This is the ONE declaration (until 2026-10-10 six shaders carried
// their own copies).
//
// Set, binding and instance name default to set 0, binding 0, `cam`. A
// shader that binds it elsewhere defines RASTER_CAM_SET, RASTER_CAM_BINDING
// or RASTER_CAM_NAME before the include (occl_cull.comp: binding 5, `rcam`).
//
// The shade passes' camera (viewInverse, projInverse) is a different block
// with a different layout: camera_ubo.glsl.

#ifndef RASTER_CAMERA_UBO_GLSL
#define RASTER_CAMERA_UBO_GLSL

#ifndef RASTER_CAM_SET
#define RASTER_CAM_SET 0
#endif
#ifndef RASTER_CAM_BINDING
#define RASTER_CAM_BINDING 0
#endif
#ifndef RASTER_CAM_NAME
#define RASTER_CAM_NAME cam
#endif

layout(set = RASTER_CAM_SET, binding = RASTER_CAM_BINDING) uniform RasterCameraUbo {
    mat4 currVPjittered;  // drives gl_Position; the primary AA jitter is applied here
    mat4 currVPunjittered;// for the motion vector; must match prevVP's projection family
    mat4 prevVP;          // previous frame's view-projection, unjittered
    vec4 jitter;          // .xy = jitter offset in clip space (sub-texel), .zw = 1 / resolution
    vec4 prevJitter;      // .xy = previous frame's jitter offset. The motion vector stays
                          // JITTER-FREE (gbuffer.frag writes a clean prevNDC - currNDC;
                          // a (prev - curr) delta was tested there and rejected); .xy is
                          // kept for the deferred shade's hybrid reproject tap correction.
                          // .z = G-buffer switches as a small whole number: bit 0
                          // the normal-map Toksvig toggle, bit 1 the stochastic
                          // tiling's exact lattice hash. .w = frame counter
                          // (alpha-hash decorrelation).
} RASTER_CAM_NAME;

#endif// RASTER_CAMERA_UBO_GLSL
