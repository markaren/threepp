#ifndef OVERLAY_DISPLAY_GLSL
#define OVERLAY_DISPLAY_GLSL

// The display transform an OVERLAY-pass shader has to apply by hand.
//
// Everything drawn after recordUpscaleAndPost composites onto a swapchain that
// post_composite.comp already exposed, tone-mapped and sRGB-ENCODED. A pass
// that arrives afterwards with linear HDR radiance in its hands — a lit
// billboard, an emissive spark — must therefore run the same curve on its own
// values before blending, or it lands in a different value domain and reads
// too bright and too saturated against the frame behind it.
//
// The curves are tonemap.glsl's, the ones post_composite.comp (the authority)
// and particle.frag's lit path include; odAcesNarkowicz has no counterpart
// there.
//
// White balance and the grading LUT are omitted by odDisplay: the old
// approximation for small translucent/additive overlay content, same as
// particle.frag's lit path. Under a strong grade it leaves sprites flatter
// and greyer than the surfaces behind them, so odDisplayGraded below applies
// both when VulkanRenderer::setOverlayColorGrade is on (OverlayGrade.flags).

#include "tonemap.glsl"

// Narkowicz ACES fit — close enough to the main pipeline's filmic curve for a
// preview pane, and a single fma chain rather than the full RRT/ODT.
vec3 odAcesNarkowicz(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Linear HDR radiance -> the swapchain's display-referred, sRGB-encoded domain.
vec3 odDisplay(vec3 hdr, uint mode, float exposure) {
    return linearToSRGB(toneMap(hdr, mode, exposure));
}

// The post stack's white balance and colour grade, as post_composite.comp
// applies them. MUST mirror vulkan::OverlayGradeGpu (ParticleFieldPass.hpp).
// flags bit0 = white balance, bit1 = grade; 0 = neither.
struct OverlayGrade {
    vec3  wbRow0;
    uint  flags;
    vec3  wbRow1;
    float saturation;
    vec3  wbRow2;
    float contrast;
    vec3  lift;
    float _pad0;
    vec3  gamma;
    float _pad1;
    vec3  gain;
    float _pad2;
};

// The function PostComposite::uploadLut bakes into the 33^3 LUT, evaluated
// directly on the sRGB-encoded value (the LUT is its trilinear sample):
// lift/gain toe and shoulder, gamma power, luma-preserving saturation,
// pivot-0.5 contrast.
vec3 odGrade(vec3 s, OverlayGrade g) {
    s = clamp(s, 0.0, 1.0);
    vec3 v = s * g.gain + g.lift * (1.0 - s);
    v = pow(max(v, vec3(0.0)), 1.0 / max(g.gamma, vec3(1e-3)));
    const float luma = dot(v, vec3(0.2126, 0.7152, 0.0722));
    v = luma + (v - luma) * g.saturation;
    v = (v - 0.5) * g.contrast + 0.5;
    return clamp(v, 0.0, 1.0);
}

// odDisplay plus the post stack's white balance (linear, before the tone map)
// and grade (encoded, after it), in post_composite.comp's order.
vec3 odDisplayGraded(vec3 hdr, uint mode, float exposure, OverlayGrade g) {
    if ((g.flags & 1u) != 0u) hdr = vec3(dot(g.wbRow0, hdr), dot(g.wbRow1, hdr), dot(g.wbRow2, hdr));
    vec3 s = linearToSRGB(toneMap(hdr, mode, exposure));
    if ((g.flags & 2u) != 0u) s = odGrade(s, g);
    return s;
}

#endif// OVERLAY_DISPLAY_GLSL
