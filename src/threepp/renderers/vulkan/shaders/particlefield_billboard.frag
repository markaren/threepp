#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

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
// VALUE DOMAIN: LINEAR radiance, always. Two builds of this file run. The
// BB_HDR_LAYER variant draws every field into BillboardHdrLayer's two targets
// (see below), whose sums are tone-mapped once, later. The plain build is the
// GLOW leg (BillboardGlowPass), whose target feeds a bright pass and a 13-tap
// downsample, both defined on linear radiance.

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2  vLocal;// [-1,1]^2 parametric square
layout(location = 1) in vec4  vColor;// rgb = linear HDR radiance, a = alive
layout(location = 2) in vec2  vSoft; // x = softness, y = core-term weight
// (Locations 3 and 4 carried the display transform until the sprites moved to
// the billboard HDR layer; nothing writes them now.)
// F5: > 0 draws an ANNULUS of this fractional width instead of the disc — the
// splash ring. Flat, so the branch is uniform over every fragment of the quad.
layout(location = 5) flat in float vRing;
// 4c: bit0 = composite premultiplied alpha-OVER instead of additive. Uniform
// over the draw (the pipeline's blend state is chosen to match), so the branch
// costs one compare.
layout(location = 6) flat in uint  vMode;
// 4c: the sprite's coverage scale — field opacity times fog transmittance and
// every other dimming factor, which in this mode make the sprite TRANSPARENT
// rather than dark. 1.0 in additive mode.
layout(location = 7) flat in float vCover;
// 4c: per-particle rotation of the texture lookup, as (cos, sin).
layout(location = 8) flat in vec2  vRot;
// BillboardRepr::softDistance: 1 / softDistance (0 = off), and this fragment's
// view depth in metres. See softFade().
layout(location = 9)  flat in float vSoftInv;
layout(location = 10) in float      vViewDepth;
const uint kModeAlphaOver = 1u;

layout(push_constant, scalar) uniform Pc {
    mat4     proj;
    vec4     mv[3];
    uint64_t paramsAddr;
    uint64_t viewAddr;
} pc;

layout(location = 0) out vec4 outColor;
#ifdef BB_HDR_LAYER
// The billboard HDR layer (BillboardHdrLayer.hpp). This variant draws into two
// rgba16f targets, LINEAR radiance pre-exposed like the scene:
//   location 0, the alpha-over sprites: premultiplied radiance + coverage,
//               blended (ONE, ONE_MINUS_SRC_ALPHA);
//   location 1, the additive sprites' radiance, blended (ONE, ONE); an
//               alpha-over sprite writes only its coverage here, which the
//               pipeline blends (ZERO, ONE_MINUS_SRC_ALPHA) so it dims the
//               additive light already behind it.
// A composite in the overlay pass tone-maps each layer ONCE, so a hundred
// overlapping sprites roll off through the curve instead of adding a hundred
// tone-mapped values and clipping at 1.0.
layout(location = 1) out vec4 outAdd;

// The frame's pre-exposure (BbView record, vulkan::kBbViewPreExposureOffset).
// The layer is stored PRE-EXPOSED, like sceneHdr: under a physical-units sun a
// lit sprite's raw radiance is ~1e5, past fp16's 65504, and the sum of many
// would be inf. Every composite reads the layer in this scale.
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer BbPreExposure { float v; };
const uint64_t kBbViewPreExposureOffset = 260ul;
float preExposure() { return BbPreExposure(pc.viewAddr + kBbViewPreExposureOffset).v; }
// One sprite never writes past fp16's range on its own; a sum can still
// saturate, which every composite treats as "very bright", never as inf.
vec3 layerValue(vec3 v) { return min(v * preExposure(), vec3(60000.0)); }
#endif

// ── Soft particles (BillboardRepr::softDistance) ────────────────────────────
// The tail of the BbView record, at vulkan::kBbViewSoftOffset: this view's
// linear scene depth, written by particlefield_soft_depth.comp from the depth
// these quads are tested against. A zero address means the view has none this
// frame (the field asked, the view could not provide), and the sprite keeps
// its hard edge.
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer BbSoft {
    uint64_t addr;  // row-major floats, positive view-depth metres
    uint     width;
    uint     height;
    float    scale; // fragment pixel -> buffer pixel (2 on the half-extent glow target)
};
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer SoftDepth { float v[]; };
const uint64_t kBbViewSoftOffset = 240ul;

// The coverage factor: 0 where the sprite meets the scene behind it, 1 from
// softDistance in front of it on. smoothstep rather than a linear ramp because
// the eye finds the KINK at the top of a linear ramp on a big flat sprite — a
// faint second line, parallel to the one this removes. 1.0 exactly once the
// gap is a full softDistance, and a multiply by 1.0 is exact, so a sprite with
// nothing near it renders the bits it did with the knob off.
float softFade() {
    BbSoft s = BbSoft(pc.viewAddr + kBbViewSoftOffset);// a reference cannot be `const`
    if (s.addr == 0ul) return 1.0;
    const ivec2 lim = ivec2(int(s.width) - 1, int(s.height) - 1);
    const ivec2 p   = clamp(ivec2(gl_FragCoord.xy * s.scale), ivec2(0), lim);
    const float scene = SoftDepth(s.addr).v[uint(p.y) * s.width + uint(p.x)];
    const float k = clamp((scene - vViewDepth) * vSoftInv, 0.0, 1.0);
    return k * k * (3.0 - 2.0 * k);
}

void main() {
    const bool alphaOver = (vMode & kModeAlphaOver) != 0u;
    const float r = length(vLocal);
    // Outside the inscribed disc there is nothing to add. Discarding rather
    // than adding zero matters at 300k: this content is fill-bound, and the
    // corners of a quad are 21% of its area.
    //
    // 4c: a TEXTURED sprite is the exception — its alpha lives in the whole
    // square and a spray puff is not a disc, so the alpha-over path keeps the
    // corners and lets the texture decide what is covered.
    if ((r >= 1.0 && !alphaOver) || vColor.a <= 0.0) discard;

    const float t = 1.0 - r;
    // Skirt vs core. The exponent runs 4 (hard, a spark) to 1.2 (soft, a glow)
    // with softness, and the core term is what survives when the sprite is only
    // a few pixels across.
    const float skirt = pow(t, mix(4.0, 1.2, vSoft.x));
    const float core  = pow(t, 9.0);
    float a = skirt + vSoft.y * core;

    // ── F5: the splash ANNULUS ──────────────────────────────────────────────
    // A splash is a rim of water thrown outward, not a filled blob: the disc
    // above would read as a growing puddle. The peak sits half a width inside
    // the quad's edge and falls off both ways, squared so the rim is soft
    // rather than a hard band — and the whole branch is uniform over the draw,
    // so a field with no splash pays one compare per fragment.
    if (vRing > 0.0) {
        const float w = max(vRing, 0.02);
        const float d = abs(r - (1.0 - 0.5 * w)) / (0.5 * w);
        const float k = max(1.0 - d, 0.0);
        a = k * k;
    }

    // 4c: the lookup is ROTATED about the sprite centre by the per-particle
    // angle the vertex stage hashed. Four atlas variants drawn at one
    // orientation tile visibly across a few hundred sprites; the same four at a
    // hashed angle do not. (cos, sin) = (1, 0) in additive mode, which is the
    // identity rotation and leaves that path's lookup untouched.
    const vec2 luv = vec2(vRot.x * vLocal.x - vRot.y * vLocal.y,
                          vRot.y * vLocal.x + vRot.x * vLocal.y);
    const vec4 tx = texture(tex, luv * 0.5 + 0.5);

    if (alphaOver) {
        // ── The sprite slice ────────────────────────────────────────────────
        // The TEXTURE is the shape here, not the procedural falloff: a spray
        // puff is a torn sheet with a ragged edge and the radial skirt above
        // would just darken its rim. A soft box feather keeps an UNTEXTURED
        // alpha field (the 1x1 white default) from drawing hard squares, and
        // costs a textured one nothing it does not already have.
        const float e  = max(abs(luv.x), abs(luv.y));
        float cv = tx.a * vCover * (1.0 - smoothstep(0.82, 1.0, e));
        // The depth fade thins the sprite, like every other dimming factor in
        // this mode. A uniform branch: a field that did not ask never reads.
        if (vSoftInv > 0.0) cv *= softFade();
        if (cv <= 0.0) discard;
        // PREMULTIPLIED, in linear radiance: the composite divides the
        // coverage back out and tone-maps the mean colour once.
        const vec3 disp = vColor.rgb * tx.rgb;
#ifdef BB_HDR_LAYER
        // The same radiance and coverage, premultiplied in LINEAR: the
        // composite divides the coverage back out and tone-maps the
        // coverage-weighted mean colour once.
        outColor = vec4(layerValue(disp) * cv, cv);
        outAdd   = vec4(0.0, 0.0, 0.0, cv);
#else
        // The glow leg draws every field additively (no alpha pipeline), so
        // this branch is the layer's only; kept well-defined regardless.
        outColor = vec4(disp * cv, cv);
#endif
        return;
    }

    a *= tx.a;
    if (vSoftInv > 0.0) a *= softFade();
    if (a <= 0.0) discard;

    const vec3 hdr = vColor.rgb * tx.rgb * a;
    // Alpha is masked out of the write (see the pipelines' colorWriteMask): an
    // additive target has no business accumulating coverage.
#ifdef BB_HDR_LAYER
    // Linear, summed by the blend; the pipeline masks location 0 out entirely
    // for an additive sprite, so the alpha-over layer is left untouched.
    outColor = vec4(0.0);
    outAdd   = vec4(layerValue(hdr), 0.0);
#else
    outColor = vec4(hdr, 0.0);
#endif
}
