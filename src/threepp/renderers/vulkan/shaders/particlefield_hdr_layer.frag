#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout  : require

// ParticleField billboard HDR layer — the composite (VulkanRenderer::
// BillboardHdrLayer.hpp).
//
// The sprites were drawn, in their usual order and against the overlay's depth,
// into two linear rgba16f targets at the display extent (the BB_HDR_LAYER
// variant of particlefield_billboard.frag). This fullscreen draw folds them
// in display space where the frame is already tone-mapped: inside the
// primary's overlay pass on the built-in TAA path, and onto a secondary view's
// colour target. (The DLSS/FSR path blends the layer in the post composite
// instead.) Two draws, one per layer, in this order:
//
//   part 0, the alpha-over layer: rgb = premultiplied radiance, a = coverage.
//     src = display(rgb / a) * a, alpha a, blended (ONE, ONE_MINUS_SRC_ALPHA).
//     For a single sprite this is what a sprite drawn on its own would give; for many
//     it tone-maps their coverage-weighted mean colour once, instead of
//     alpha-blending each sprite's own tone-mapped colour.
//
//   part 1, the additive layer: src = display(rgb), blended
//     (ONE_MINUS_DST_COLOR, ONE), i.e. dst + src * (1 - dst), a screen blend.
//     The sum of the sprites goes through the tone curve ONCE, so a dense
//     cloud rolls off toward the curve's shoulder and keeps its hue where the
//     old per-sprite sum clipped every channel at 1.0; the screen blend keeps
//     the result below 1.0 over a bright background too. It is the exact sum
//     under an exponential curve and a bounded, smooth stand-in under ACES.
//
// The same display function and grade the field glow uses (odDisplay /
// odDisplayGraded, particlefield_glow.frag).

#include "overlay_display.glsl"

layout(set = 0, binding = 0) uniform sampler2D overLayer;// alpha-over sprites
layout(set = 0, binding = 1) uniform sampler2D addLayer; // additive sprites

layout(push_constant, scalar) uniform Pc {
    float exposure;    // currentExposure() / preExposure(): the layer is pre-exposed
    uint  toneMapMode; // threepp::ToneMapping
    uint  part;        // 0 = alpha-over layer, 1 = additive layer
    uint  _pad;
    OverlayGrade grade;// setOverlayColorGrade; zero flags = the plain odDisplay
} pc;

layout(location = 0) out vec4 outColor;

vec3 display(vec3 hdr) {
    return pc.grade.flags == 0u ? odDisplay(hdr, pc.toneMapMode, pc.exposure)
                                : odDisplayGraded(hdr, pc.toneMapMode, pc.exposure, pc.grade);
}

// The layer is fp16 and blended: a saturated sum can reach inf, and inf * 0
// (an alpha-over sprite over it) is NaN. Neither may reach the tone map.
vec3 finite(vec3 v) { return clamp(mix(v, vec3(0.0), isnan(v)), 0.0, 65504.0); }

void main() {
    // The layers are at the display extent, pixel for pixel: a fetch, no filter.
    const ivec2 p = ivec2(gl_FragCoord.xy);
    if (pc.part == 0u) {
        const vec4 o = texelFetch(overLayer, p, 0);
        // No sprite here (or a NaN coverage): leave the frame exactly as it is.
        if (!(o.a > 0.0)) discard;
        const float a = min(o.a, 1.0);
        outColor = vec4(display(finite(o.rgb) / max(o.a, 1e-6)) * a, a);
    } else {
        const vec3 s = finite(texelFetch(addLayer, p, 0).rgb);
        if (max(s.r, max(s.g, s.b)) <= 0.0) discard;
        // Alpha is masked out of the write: the frame's alpha is not part of the composite.
        outColor = vec4(display(s), 0.0);
    }
}
