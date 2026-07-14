#version 460

// Pair to overlay_color.vert. Modulates the material color (push constant)
// by the per-vertex color, which is the three.js convention when
// material.vertexColors == true. Alpha follows pc.color.w (material opacity).

layout(location = 0) in vec3 vColor;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMask;// coverage for overlay_aa (see overlay.frag)

void main() {
    outColor = vec4(pc.color.rgb * vColor, pc.color.w);
    outMask  = vec4(1.0);
}
