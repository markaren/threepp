#version 460

// Overlay vertex shader for Points objects. Same vertex+color attribs as
// overlay_color.vert, but also writes gl_PointSize so the rasteriser draws
// each vertex as a square sprite of `pc.color.w` pixels. The host packs
// PointsMaterial::size into the push constant's color.w slot for this
// pipeline (line/wireframe variants treat color.w as opacity; the point
// pipeline overrides that meaning).

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color;  // .rgb = material color tint, .w = point size in pixels
} pc;

layout(location = 0) out vec3 vColor;

void main() {
    vec4 clip = pc.mvp * vec4(inPos, 1.0);
    clip.y    = -clip.y;
    gl_Position  = clip;
    gl_PointSize = max(1.0, pc.color.w);
    vColor       = inColor;
}
