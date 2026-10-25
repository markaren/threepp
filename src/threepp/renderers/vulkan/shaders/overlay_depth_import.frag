#version 460

// The overlay pass's occluder depth when the ONLY overlay content is
// ParticleField billboards: the G-buffer's own depth, resampled to the
// display extent, instead of a second rasterization of the whole scene
// (overlay_depth.vert).
//
// The prepass exists so that hard-edged overlays — wireframes, lines, world
// sprites — are tested against UNJITTERED depth and do not shimmer with the
// TAA jitter. A field billboard is a soft alpha sprite that is already drawn
// outside the temporal filter; the jitter's half a render pixel at its
// intersection line is below what the eye resolves on a dust puff (and under
// BillboardRepr::softDistance the line is a fade anyway). Paying a full-scene
// depth pass at display resolution for that — 0.9 ms of GPU and 1.2 ms of
// CPU on a 15k-entry forest, for a few dozen sprites — was the fixed cost
// this replaces.
//
// NEAREST sampling (the sampler's filter): a bilinear depth would invent
// intermediate depths along every silhouette. With renderScale < 1 the
// G-buffer is smaller than the display, so a display pixel takes the depth
// of the render pixel under it. Written through gl_FragDepth with an
// always-pass test; under the MSAA overlay every covered sample takes the
// same value.

layout(set = 0, binding = 0) uniform sampler2D gbufDepth;

layout(location = 0) in vec2 vUv;

void main() {
    gl_FragDepth = texture(gbufDepth, vUv).x;
}
