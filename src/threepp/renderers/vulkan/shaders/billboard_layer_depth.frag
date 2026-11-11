#version 460

// The billboard HDR layer's own depth (BillboardHdrLayer): a one-sample copy of
// the overlay pass's depth at the same extent, written as gl_FragDepth by one
// fullscreen draw (compute cannot target a depth-aspect image).
//
// Needed in two cases. A multisampled overlay (Canvas antialiasing) keeps its
// occluders in a multisampled buffer the one-sample layer cannot attach, so the
// samples are reduced here; and a frame with a splat depth stamp stamps the
// clouds into this copy, leaving the overlay's own buffer as the overlay pass
// expects it.
//
// THE REDUCTION IS A MIN over the samples, under reverse-Z the FARTHEST, as in
// billboard_glow_depth.frag: the sprites' GREATER_OR_EQUAL test then keeps a
// sprite that is in front of any sample of the pixel, so along a silhouette a
// sprite behind the edge shows in that one pixel rather than leaving a gap.

#ifdef DEPTH_MS
layout(set = 0, binding = 0) uniform sampler2DMS overlayDepth;
#else
layout(set = 0, binding = 0) uniform sampler2D overlayDepth;
#endif

void main() {
    const ivec2 p = ivec2(gl_FragCoord.xy);
#ifdef DEPTH_MS
    float d = 1.0;// reverse-Z: 1 is the near plane, above every stored value
    const int n = textureSamples(overlayDepth);
    for (int s = 0; s < n; ++s) d = min(d, texelFetch(overlayDepth, p, s).x);
    gl_FragDepth = d;
#else
    gl_FragDepth = texelFetch(overlayDepth, p, 0).x;
#endif
}
