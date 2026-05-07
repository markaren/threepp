#version 460

// Pair to overlay.vert. Emits a constant tint — that's all the overlay
// needs. sRGB encoding follows the surrounding render-target convention
// (post-tonemap intermediate writes linear; UNORM swapchain expects sRGB
// pre-encoded). The host stores the user's color as already-linear and
// passes it through; if the surface format is UNORM_SRGB the GPU's
// hardware sRGB write-out converts on store.

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = pc.color;
}
