#version 460

// Overlay vertex shader for Points objects. Same vertex+color attribs as
// overlay_color.vert, but also writes gl_PointSize so the rasteriser draws
// each vertex as a square sprite. The host packs PointsMaterial::size into
// the push constant's color.w slot for this pipeline (line/wireframe
// variants treat color.w as opacity; the point pipeline overrides that
// meaning).
//
// PointsMaterial::size means one of two things, and which one is the
// material's `sizeAttenuation` flag — the SAME contract as the GL backend's
// points_vert.glsl, which this mirrors:
//
//   sizeAttenuation == true   size is in WORLD units, and a point covers
//                             `size * (0.5*viewportHeight) / -viewZ` pixels,
//                             so it shrinks with distance like geometry does.
//   sizeAttenuation == false  size is already in PIXELS and is constant.
//
// Reading color.w as pixels unconditionally — which this shader used to do —
// silently collapses every attenuated cloud to 1px specks, because a
// world-space size is a small fraction (0.06 m for the editor's sensor
// overlay) and max(1.0, 0.06) is 1.0. Same cloud, same count, same place, but
// a tenth the ink on Vulkan as on GL.
//
// params.x carries the attenuation scale (0.5*viewportHeight) already resolved
// by the host, or 0 when the size is in pixels — the host is where both the
// viewport height and the "is this projection perspective" answer live, and an
// ortho camera never attenuates (again matching points_vert.glsl, which gates
// on isPerspectiveMatrix).

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color; // .rgb = material color tint, .w = point size (world units or px)
    vec4 params;// .x = attenuation scale, 0 = color.w is already in pixels
} pc;

layout(location = 0) out vec3 vColor;

void main() {
    vec4 clip = pc.mvp * vec4(inPos, 1.0);
    clip.y    = -clip.y;
    gl_Position = clip;

    // clip.w == -viewZ for a perspective projection, which is exactly the
    // divisor points_vert.glsl uses; the depth-remap the host folds into the
    // MVP touches row 2 only, so w survives it. Guard the divide: a point
    // exactly on the eye plane would otherwise produce inf.
    float px = pc.params.x > 0.0
                     ? pc.color.w * pc.params.x / max(clip.w, 1e-4)
                     : pc.color.w;

    // Upper clamp is the spec floor for VkPhysicalDeviceLimits::pointSizeRange
    // (guaranteed to reach 64), not a look choice — a point a few centimetres
    // from the near plane can ask for thousands of pixels, and exceeding the
    // device range is undefined rather than merely large.
    gl_PointSize = clamp(px, 1.0, 64.0);
    vColor       = inColor;
}
