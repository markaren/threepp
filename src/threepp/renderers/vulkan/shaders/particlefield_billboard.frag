#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// ParticleField BILLBOARD fragment stage — the soft procedural sprite.
//
// NO TEXTURE ASSET IS REQUIRED and none is loaded by default: the sprite is a
// radial falloff evaluated here, so a field of embers or rain needs nothing on
// disk and a demo that ships in this repo carries no new binary. When
// BillboardRepr::texture IS set, the renderer binds it at set 0 and it
// MODULATES this shape (rgb tint x alpha coverage); the untextured case binds
// the 1x1 white default the legacy billboard path already owns, so the sampler
// is always valid and the multiply is an exact no-op.
//
// The falloff is two terms rather than one: a wide skirt that gives the sprite
// its glow, plus a tight core that keeps it from reading as a soft blob at
// small sizes. `softness` slides between the two. That shape is the difference
// between an EMBER and an ORB — a single smoothstep over the radius, drawn at
// 30 px, is exactly the "large regular blob" this phase set out to replace.
//
// VALUE DOMAIN. This pass runs after the upscaler, onto a swapchain that
// post_composite.comp already tone-mapped and sRGB-encoded, so the linear HDR
// radiance the vertex stage computed has to go through the same curve before it
// is blended. Blending is plain ADDITIVE (ONE, ONE) with the coverage folded
// into the radiance here, so nothing is applied twice and the order the quads
// arrive in cannot matter.

#include "overlay_display.glsl"

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2  vLocal;// [-1,1]^2 parametric square
layout(location = 1) in vec4  vColor;// rgb = linear HDR radiance, a = alive
layout(location = 2) in float vSoft;
// F4: the display transform, flat from the vertex stage. A NEGATIVE exposure is
// the glow pass's signal to emit LINEAR HDR instead — that target is the input
// to a bright pass and a 13-tap downsample, both of which are defined on linear
// radiance and would be meaningless on a tone-mapped, sRGB-encoded value.
layout(location = 3) flat in float vExposure;
layout(location = 4) flat in uint  vToneMap;

layout(push_constant, scalar) uniform Pc {
    mat4     proj;
    vec4     mv[3];
    uint64_t paramsAddr;
    uint64_t viewAddr;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    const float r = length(vLocal);
    // Outside the inscribed disc there is nothing to add. Discarding rather
    // than adding zero matters at 300k: this content is fill-bound, and the
    // corners of a quad are 21% of its area.
    if (r >= 1.0 || vColor.a <= 0.0) discard;

    const float t = 1.0 - r;
    // Skirt vs core. The exponent runs 4 (hard, a spark) to 1.2 (soft, a glow)
    // with softness, and the core term is what survives when the sprite is only
    // a few pixels across.
    const float skirt = pow(t, mix(4.0, 1.2, vSoft));
    const float core  = pow(t, 9.0);
    float a = skirt + 0.85 * core;

    const vec4 tx = texture(tex, vLocal * 0.5 + 0.5);
    a *= tx.a;
    if (a <= 0.0) discard;

    const vec3 hdr = vColor.rgb * tx.rgb * a;
    // Alpha is masked out of the write (see the pipeline's colorWriteMask): the
    // swapchain's alpha channel is not part of the composite and an additive
    // pass has no business accumulating into it.
    outColor = vec4(vExposure < 0.0 ? hdr : odDisplay(hdr, vToneMap, vExposure), 0.0);
}
