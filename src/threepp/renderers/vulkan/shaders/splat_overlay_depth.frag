#version 460

// Splat depth stamp for the post-resolve overlay pass.
//
// A wireframe, a Line, a world sprite or a particle billboard is not in the
// G-buffer: it is drawn after the temporal resolve and depth-tested against
// the unjittered depth prepass (overlay_depth.vert). The splat compositor is
// a COMPUTE pass — it reads that geometry depth to stop accumulating behind
// it, but it has no depth attachment of its own and writes nothing back. So
// without this pass an overlay line BEHIND a cloud passes the depth test
// against whatever the raster left there and draws over the cloud at full
// strength, while GL — which draws the cloud last, in the transparent pass —
// blends it away. This is that missing depth write.
//
// A fullscreen triangle over the overlay's depth attachment, reading the
// splat depth AOV (a view-space distance in world units, 0 where no cloud
// owns the pixel) and re-expressing it as the reverse-Z NDC depth the overlay
// compares against. The pipeline's depth test is GREATER, so the stamp only
// wins where the cloud is NEARER than the geometry already there — a cloud
// behind a wall leaves the wall's depth alone.
//
// The AOV's coverage > 0.5 gate is inherited, and it is the whole resolution
// of this pass: a pixel the cloud more than half owns hides the overlay, one
// it owns less does not. A depth buffer has no vocabulary for "60% hidden",
// and the alternative — every overlay shader sampling and blending against
// splat transmittance — is a change to five pipelines for a fringe a pixel
// or two wide.

// GENERAL for the whole frame (SplatPass both stores to and clears it), so it
// is read as a storage image here rather than sampled: no layout transition,
// no sampler, and the nearest-texel fetch a depth stamp wants anyway.
layout(set = 0, binding = 0, r32f) uniform readonly image2D splatDepth;

layout(push_constant) uniform PC {
    vec2  aovScale;   // pane pixel -> AOV texel (render extent / pane extent)
    vec2  paneOrigin; // split-screen: this pane's origin, in swapchain pixels
    vec2  aovLimit;   // AOV extent - 1, for the fetch clamp
    float projA;      // reverse-Z projection [2][2]
    float projB;      // reverse-Z projection [3][2]
    uint  ortho;      // 1 when the camera is orthographic (clip w == 1)
} pc;

void main() {
    const vec2 aovPx = (gl_FragCoord.xy - pc.paneOrigin) * pc.aovScale;
    const ivec2 px   = ivec2(clamp(aovPx, vec2(0.0), pc.aovLimit));
    const float dist = imageLoad(splatDepth, px).x;
    // 0 is the AOV's "no cloud here" sentinel; NaN fails this too, which is
    // what we want — an unwritten texel must not stamp anything.
    if (!(dist > 0.0)) discard;

    // clip.z = A * z_view + B with z_view == -dist, and clip.w is dist for a
    // perspective projection, 1 for an orthographic one.
    const float clipZ = -pc.projA * dist + pc.projB;
    gl_FragDepth = clamp(pc.ortho != 0u ? clipZ : clipZ / dist, 0.0, 1.0);
}
