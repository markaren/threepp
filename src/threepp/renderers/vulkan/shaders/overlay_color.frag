#version 460

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
layout(location = 1) out vec4 outMask;// coverage for overlay_aa (see overlay.frag)

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

void main() {
    outColor = vec4(linearToSRGB(pc.color.rgb * vColor), pc.color.w);
    outMask  = vec4(1.0);
}
