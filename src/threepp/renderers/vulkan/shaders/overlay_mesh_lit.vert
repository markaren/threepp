#version 460

// Lit split-screen pane mesh vertex shader. A secondary scissored render()
// (the editor's camera preview, multiple_scenes' second pane) used to draw
// meshes through the flat overlay shader — silhouettes over whatever the
// swapchain already held. This pair gives that pane real solid shading:
// depth-tested triangles lit by the scene's sun + ambient (see the .frag).
//
// Push-constant layout is 128 bytes exactly (the guaranteed floor):
// mvp (64) + the world normal-matrix columns as three vec4 (48) + color (16).
// The light data would not fit, so it lives in a small UBO (set 0, binding 0)
// that is constant for the whole pane anyway.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform Pc {
    mat4 mvp;  // proj(z-remapped) · view · model
    vec4 nrm0; // world normal matrix, column 0 (w unused)
    vec4 nrm1;
    vec4 nrm2;
    vec4 color;// material base color (linear), w = opacity
} pc;

layout(location = 0) out vec3 vNormal;

void main() {
    vec4 clip = pc.mvp * vec4(inPos, 1.0);
    // GL-convention projection (Y up in NDC) → Vulkan (Y down): negate at the
    // gl_Position boundary, same as overlay.vert / gbuffer.vert.
    clip.y = -clip.y;
    gl_Position = clip;
    vNormal = mat3(pc.nrm0.xyz, pc.nrm1.xyz, pc.nrm2.xyz) * inNormal;
}
