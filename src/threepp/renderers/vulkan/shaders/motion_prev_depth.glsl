// The motion G-buffer's "this surface's previous NDC depth", carried in .b AND .a.
//
// The attachment is rgba16f, and the temporal passes compare that depth against
// the prev depth BUFFER (a float32) to tell a reprojected surface from a
// disocclusion. One half float holds it well under a PERSPECTIVE projection:
// reverse-Z depth there is ~near/distance, a small number, and a half float's
// 11 bits are relative, so the view distance comes back to 0.05%.
//
// Under an ORTHOGRAPHIC projection the depth is linear in distance and sits just
// below 1 for everything near the camera, where a half float steps by 2^-11 of
// the WHOLE clip range: 4.9 m with far = 10000. The history tests allow 10% of
// the view distance, so across a slab of each step (and, closer than ~50 m,
// across most of the frame) every pixel failed them every frame and published
// its raw few-ray sample: ambient occlusion, GI and shadow history boiling on a
// static ortho view while the same view in perspective held still.
//
// So the depth is split: .b the part on a 1/2048 grid, which a half float holds
// exactly anywhere in [0, 1], and .a the remainder, below 2^-11, which keeps a
// half float's relative precision. The sum is good to 2^-22 of the clip range
// under either projection (2.4 mm at far = 10000), and never worse than the
// single half float was: a perspective depth below 1/2048 is all remainder.
//
// Both channels are linear in the depth, so a filtered tap sums the same way.
// Sky is the attachment's clear, 0 + 0 = reverse-Z far.

#ifndef MOTION_PREV_DEPTH_GLSL
#define MOTION_PREV_DEPTH_GLSL

// gbuffer.frag: prev NDC depth -> the motion attachment's .ba.
vec2 motionPackPrevDepth(float d) {
    const float hi = floor(clamp(d, 0.0, 1.0) * 2048.0) * (1.0 / 2048.0);
    return vec2(hi, d - hi);
}

// A motion attachment tap -> prev NDC depth.
float motionPrevDepth(vec4 motionTap) {
    return motionTap.b + motionTap.a;
}

#endif// MOTION_PREV_DEPTH_GLSL
