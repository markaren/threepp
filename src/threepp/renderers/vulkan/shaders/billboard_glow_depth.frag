#version 460

// Half-extent depth reduction for the billboard glow source
// (BillboardGlowPass, plans/particle-atmosphere.md F4 item 1).
//
// The glow source target is half the display extent and the overlay pass's
// depth is full extent; a render area must fit its attachments, so the glow
// draw cannot depth-test against the overlay's buffer directly. This
// fullscreen pass reduces that buffer into the half-extent depth attachment
// the glow draw then tests against, which is what stops an occluded spark
// from contributing a halo through a wall.
//
// Compute cannot target a depth-aspect image (imageStore has no depth-aspect
// storage class), so the reduction is a rasterization writing gl_FragDepth —
// the same reason gbuf_resolve_depth.frag exists.
//
// THE REDUCTION IS A MIN, and under reverse-Z that means the FARTHEST of the
// four full-extent texels a half-extent texel covers. The glow test is
// GREATER_OR_EQUAL, so the farthest occluder is the permissive choice: a
// spark within one half-extent texel of a silhouette stays visible rather
// than being quantised away by a depth buffer that is coarser than the
// geometry that filled it.

layout(push_constant) uniform Push {
    uvec2 srcExtent;// the full-extent overlay depth's dimensions
} pc;

#ifdef DEPTH_MS
layout(set = 0, binding = 0) uniform sampler2DMS overlayDepth;
#else
layout(set = 0, binding = 0) uniform sampler2D overlayDepth;
#endif

void main() {
    const ivec2 base = ivec2(gl_FragCoord.xy) * 2;
    const ivec2 lim  = ivec2(pc.srcExtent) - ivec2(1);

    // Reverse-Z: 1 is the near plane, so start above every stored value.
    float d = 1.0;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            // clamp: an odd display extent leaves the last half-texel with
            // only one full-extent column/row under it.
            const ivec2 p = min(base + ivec2(dx, dy), lim);
#ifdef DEPTH_MS
            const int n = textureSamples(overlayDepth);
            for (int s = 0; s < n; ++s) d = min(d, texelFetch(overlayDepth, p, s).x);
#else
            d = min(d, texelFetch(overlayDepth, p, 0).x);
#endif
        }
    }
    gl_FragDepth = d;
}
