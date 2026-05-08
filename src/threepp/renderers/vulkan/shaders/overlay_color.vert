#version 460

// Overlay vertex shader with per-vertex color input. Used by the colored
// Line / LineSegments pipelines (and any future colored mesh overlay) when
// the geometry has a "color" attribute AND the material has
// vertexColors == true. Final fragment color is pc.color * inColor —
// matches three.js semantics where vertexColors modulates the material
// color rather than replacing it.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec3 vColor;

void main() {
    vec4 clip = pc.mvp * vec4(inPos, 1.0);
    clip.y    = -clip.y;
    gl_Position = clip;
    vColor    = inColor;
}
