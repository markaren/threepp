#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout  : require

// ParticleField BILLBOARD GLOW composite — plans/particle-atmosphere.md F4 (1).
//
// ── WHY THIS PASS EXISTS AT ALL ─────────────────────────────────────────────
// Field billboards composite AFTER recordUpscaleAndPost, on the swapchain, in
// the overlay slot (F3 note 1). That placement is load-bearing: a 3-px spark
// crossing 20 px in a frame is precisely the content a temporal resolve
// mis-handles, and being outside TAA/DLSS/FSR is what makes them clean for
// free. But the scene's own bloom pyramid runs BEFORE that point, on sceneHdr,
// so a spark's radiance never reaches it — hence F3's "billboards get no bloom".
//
// Moving the composite earlier would buy bloom and forfeit the temporal
// property, which is a bad trade. So the sparks get their OWN pyramid instead:
// the glow-enabled fields are drawn a second time into a small offscreen HDR
// target, the SHARED bloom_down/bloom_up shaders run on that target alone, and
// this shader adds the result into the swapchain inside the same overlay
// render-pass instance the sharp quads land in. Nothing moves; a second,
// cheaper pyramid arrives.
//
// ── WHY A SEPARATE TARGET RATHER THAN A BRIGHTER QUAD ───────────────────────
// A glow is a LOW-FREQUENCY spatial spread of energy, which is what a pyramid
// computes and what no amount of falloff shaping inside a 3-px sprite can fake.
// The target is HALF the display extent and the pyramid starts at a quarter,
// because that is already finer than the feature it carries.
//
// ── VALUE DOMAIN ────────────────────────────────────────────────────────────
// The glow target holds LINEAR HDR (particlefield_billboard.frag skips its
// display transform when it renders there), so the bright pass and the 13-tap
// downsample operate in the domain they are defined in. The tone map is applied
// HERE, once, on the blurred result — the same place and the same curve the
// sharp quads use, so the two halves of a spark land in one domain.

#include "overlay_display.glsl"

// Bloom pyramid level 0 of the billboard-only chain. Sampled with a linear
// filter at half-of-half resolution, i.e. this fetch is doing the final
// upsample of the chain for free.
layout(set = 0, binding = 0) uniform sampler2D glowPyr;

layout(push_constant, scalar) uniform Pc {
    float intensity;  // scales the summed pyramid; folded with the level count
    float exposure;   // currentExposure(), for the display transform
    uint  toneMapMode;// threepp::ToneMapping
    uint  _pad;
    vec2  invDisplay; // 1 / swapchain extent. Passed rather than derived from
                      // textureSize(): the chain halves twice with integer
                      // truncation, so a 1281-wide frame's quarter level is 320
                      // and 4 * 320 is not 1281.
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    const vec2 uv = gl_FragCoord.xy * pc.invDisplay;
    const vec3 hdr = texture(glowPyr, uv).rgb * pc.intensity;
    // Additive (ONE, ONE) with alpha masked out of the write, exactly as the
    // sharp quads blend — so the order of the two draws inside the pass cannot
    // matter, and the sum is the same either way.
    outColor = vec4(odDisplay(hdr, pc.toneMapMode, pc.exposure), 0.0);
}
