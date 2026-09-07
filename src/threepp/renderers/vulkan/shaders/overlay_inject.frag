#version 460

// Scene-inject for the MSAA hybrid overlay pass.
//
// The overlay rasterizes into a multisampled color target that is resolved
// onto the swapchain at vkCmdEndRendering. The swapchain already holds the
// TAA-resolved (and post-composited) scene, so before ANY overlay geometry is
// drawn this fullscreen triangle seeds every sample of every pixel with that
// scene image — copied to a 1-sample scratch beforehand, because a pass may
// not sample its own render target. Without it the MSAA target would start
// undefined and every alpha-blended overlay would composite against garbage,
// and untouched pixels would resolve to noise.
//
// texelFetch (not a filtered sample) so the swapchain copy round-trips
// bit-exactly: UNORM8 -> float -> UNORM8 is lossless, so pixels no overlay
// primitive touches resolve back to the identical value they started with.
// Colour handling is deliberately pass-through: the scratch is display-
// referred (sRGB-encoded by post_composite into a UNORM swapchain) and the
// overlay shaders encode to the same space, so the resolve averages in
// display-referred space exactly like a GL default framebuffer does.

layout(set = 0, binding = 0) uniform sampler2D srcTex;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(texelFetch(srcTex, ivec2(gl_FragCoord.xy), 0).rgb, 1.0);
}
