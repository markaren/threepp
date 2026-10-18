#version 460
#extension GL_GOOGLE_include_directive : enable

// Pair to overlay.vert. Emits a constant tint for HUD/overlay geometry
// (ortho-HUD filled meshes + lines, and the world-space wireframe/line
// overlay). threepp material colors are linear; the swapchain is
// VK_FORMAT_B8G8R8A8_UNORM (no hardware sRGB write-out), and the
// rendered image is sRGB-encoded in post_composite.comp before it lands here.
// So we must apply the same linear->sRGB OETF, else overlay geometry is
// drawn darker than the rest of the frame (dark fills collapse to black).

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec4 outColor;

#include "overlay_display.glsl"

void main() {
    outColor = vec4(odLinearToSRGB(pc.color.rgb), pc.color.a);
}
