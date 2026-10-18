#version 460
#extension GL_GOOGLE_include_directive : enable

// Pair to overlay_color.vert. Modulates the material color (push constant)
// by the per-vertex color, which is the three.js convention when
// material.vertexColors == true. Alpha follows pc.color.w (material opacity).
//
// Like every overlay shader, the output lands in a display-referred UNORM
// swapchain (the rendered frame is sRGB-encoded by post_composite), so the
// linear product must go through the same linear->sRGB OETF as overlay.frag —
// without it, vertex-colored lines render darker than the identical material
// on the GL backend (which encodes via linearToOutputTexel).

layout(location = 0) in vec3 vColor;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec4 outColor;

#include "overlay_display.glsl"

void main() {
    outColor = vec4(odLinearToSRGB(pc.color.rgb * vColor), pc.color.w);
}
