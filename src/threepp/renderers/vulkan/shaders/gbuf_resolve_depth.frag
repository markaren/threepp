#version 460

// MSAA depth resolve — companion to gbuf_resolve.comp. Compute shaders
// cannot target a depth-aspect image (imageStore has no depth-aspect
// storage class), so the dominant sample's depth is written here instead,
// through a tiny fullscreen-triangle rasterization pass with depthWriteEnable
// and gl_FragDepth. Reads the dominant-sample index gbuf_resolve.comp
// already packed into the just-resolved ids.w (bits 4..5) — so this pass
// re-derives the SAME winning sample rather than re-running its own vote,
// guaranteeing the two resolves never disagree about which sample won.
//
// depthMS is the raw MSAA depth attachment (VK_FORMAT_D32_SFLOAT with
// samples>1); idsResolved is the ALREADY-WRITTEN single-sample ids image
// from gbuf_resolve.comp (this pass runs strictly after it, ordered by a
// compute->fragment barrier in recordCommandBuffer).

layout(set = 0, binding = 0) uniform sampler2DMS depthMS;
layout(set = 0, binding = 1) uniform usampler2D  idsResolved;

void main() {
    const ivec2 px = ivec2(gl_FragCoord.xy);
    const uint  w  = texelFetch(idsResolved, px, 0).w;
    const int   dominantIdx = int((w >> 4u) & 0x3u);
    gl_FragDepth = texelFetch(depthMS, px, dominantIdx).x;
}
