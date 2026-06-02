#version 460

// Pair to overlay.vert. Emits a constant tint for HUD/overlay geometry
// (ortho-HUD filled meshes + lines, and the world-space wireframe/line
// overlay). threepp material colors are linear; the swapchain is
// VK_FORMAT_B8G8R8A8_UNORM (no hardware sRGB write-out), and the
// path-traced image is sRGB-encoded in denoise.comp before it lands here.
// So we must apply the same linear->sRGB OETF, else overlay geometry is
// drawn darker than the rest of the frame (dark fills collapse to black).

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec4 outColor;

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

void main() {
    outColor = vec4(linearToSRGB(pc.color.rgb), pc.color.a);
}
