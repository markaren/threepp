#version 460
#extension GL_EXT_scalar_block_layout    : require
#extension GL_EXT_buffer_reference       : require
#extension GL_EXT_buffer_reference2      : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// ParticleField BILLBOARD vertex stage — plans/particle-field.md §3.2, narrowed
// by plans/particle-atmosphere.md F-D to the additive / emissive-unlit slice.
//
// ── WHY VERTEX-LESS ─────────────────────────────────────────────────────────
// The legacy ParticleSystem path (particle.vert + ParticleGeomRec) stores FOUR
// coincident vertices per particle — position, a {size,angle,opacity} normal, a
// uv and a colour — and re-uploads all of it every frame from a CPU walk. That
// is 44 B/particle/frame across the bus and a hard cap (kMaxLitParticles =
// 16384) on how many can exist. Here there is no vertex buffer at all: the draw
// is vkCmdDrawIndirect(4 vertices, liveCount instances) and
//
//     gl_VertexIndex   -> which CORNER of the quad     (2 bits)
//     gl_InstanceIndex -> which PARTICLE               (the device-side count)
//
// with the position and radius pulled straight out of the field's position
// buffer by buffer_reference. Nothing is uploaded, nothing is capped, and the
// per-frame CPU cost of a 300k-particle field is the 112 B parameter record
// below.
//
// ── WHY NO SORT ─────────────────────────────────────────────────────────────
// This phase draws ADDITIVE billboards only, and addition COMMUTES: the frame
// buffer holds the same sum whatever order the quads arrive in, so there is
// nothing for a sort to fix. Normal-blend billboards do need back-to-front, and
// when they land they must reuse SplatPass's deterministic radix sort rather
// than grow a second one (plan §3.2 item 3) — that is deferred, not solved.
//
// ── WHY UNLIT ───────────────────────────────────────────────────────────────
// particle_light.comp is skipped ENTIRELY for a field billboard. That pass
// casts an RT shadow ray and walks the cluster list PER PARTICLE — µs-class
// work at 10^5 — to answer "how much light falls on this billboard", and the
// answer is worthless for the content this slice draws: an ember, a spark and a
// rain streak are emissive, and an emitter's own light in this renderer is the
// FireEffect PointLight, which already lights the world through the ordinary
// deferred path. Paying a shadow ray per spark to darken a thing that emits is
// the wrong trade at any particle count.
//
// ── CAMERA FACING, PERSPECTIVE AND ORTHO ────────────────────────────────────
// The quad is built in VIEW space — offset the particle's view-space centre
// along the view basis (1,0,0)/(0,1,0), then project. That is exact for both a
// perspective and an orthographic camera and needs no camera position, no
// special case and no world-space right/up vectors. (The legacy path instead
// scales in CLIP space by proj[1][1]/|viewPos|, which is a perspective-only
// approximation.) A secondary view runs this same shader with its own camera
// matrices, so a CameraSensor sees the billboards its scene contains.

// ParticlePos / physx::PxVec4 / GLSL vec4 under scalar layout are the same 16
// bytes — the identity the whole buffer contract is built on. w carries the
// per-particle radius when alive and the w < 0 DEAD sentinel when not: the one
// liveness rule every consumer tests.
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer PosBuf { vec4 v[]; };

// Per-field appearance, one record, written by ParticleFieldPass::prepareFrame
// into a per-frame-in-flight host-visible block and reached by DEVICE ADDRESS.
// A buffer reference rather than a descriptor on purpose: this pass therefore
// allocates no set, writes no set, and cannot possibly update a descriptor a
// frame in flight still names (VUID-03047). KEEP IN SYNC with
// BillboardParamsGpu in ParticleFieldPass.hpp.
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer BbParams {
    uint64_t posAddr;
    uint64_t prevPosAddr;   // == posAddr when the field has no previous state
    vec3     colorHot;      // linear HDR radiance at age 0
    float    sizeScale;     // BillboardRepr::sizeScale
    vec3     colorCool;     // linear HDR radiance at end of life
    float    uniformRadius; // used when w is NOT the radius (WSemantic::InvMass)
    float    stretchOverDt; // seconds of velocity per unit (pos - prevPos); 0 = round
    float    stretchMax;    // cap on the stretch, in multiples of the radius
    float    intensity;     // HDR scale on the colours
    float    softness;      // 0 = tight dot, 1 = broad glow
    float    fadePower;     // brightness = (1 - ageFrac)^fadePower
    float    brightJitter;  // +/- fraction of brightness, hashed per particle
    float    sizeTaper;     // radius *= (1 - sizeTaper * ageFrac)
    float    lifetime;      // emitter lifecycle; 0 = no age is knowable
    float    lifetimeJitter;
    float    duty;
    float    time;          // the emitter's ABSOLUTE t, never a wall clock
    uint     seed;
    uint     flags;         // bit0: w IS the radius (WSemantic::Radius)
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

layout(push_constant, scalar) uniform Pc {
    mat4     proj;       // 64  view -> clip, UNJITTERED (this is a post-TAA pass)
    vec4     mv[3];      // 48  ROWS of the affine view * model; row-major so a
                         //     view-space position is three dot products and no
                         //     matrix-layout convention has to be agreed on
    uint64_t paramsAddr; //  8  -> this field's BbParams record
    float    exposure;   //  4  currentExposure(), for the display transform
    uint     toneMapMode;//  4  threepp::ToneMapping
} pc;                    // 128 B exactly — the guaranteed minimum push range

layout(location = 0) out vec2  vLocal;// [-1,1]^2 parametric square
layout(location = 1) out vec4  vColor;// rgb = linear HDR radiance, a = coverage
layout(location = 2) out float vSoft;

// ── The hash ────────────────────────────────────────────────────────────────
// Bit-for-bit particle_emit.comp's hashU/rnd01, which is itself bit-for-bit
// FireEffect.cpp's. It is shared here for a specific reason: this shader
// RE-DERIVES a particle's age from the same closed form the emitter evaluated,
// rather than reading an age nobody wrote. There is no age channel in the
// position buffer (w is the radius, and a second buffer for one float per
// particle would undo the point of the entity), and the lifecycle is four lines
// of arithmetic over the same hash — so the cheapest correct answer is to
// compute it. KEEP IN SYNC with particle_emit.comp: streams 0 (birth phase) and
// 4 (period jitter) must mean there exactly what they mean here.
uint hashU(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
float rnd01(uint slot, uint stream, uint seed) {
    return float(hashU(slot * 0x9e3779b9u + stream * 0x85ebca6bu + seed) >> 8) *
           (1.0 / 16777216.0);
}
float rndS(uint slot, uint stream, uint seed) { return rnd01(slot, stream, seed) * 2.0 - 1.0; }

vec3 viewOf(vec4 p) {
    return vec3(dot(pc.mv[0], p), dot(pc.mv[1], p), dot(pc.mv[2], p));
}

void main() {
    BbParams   P  = BbParams(pc.paramsAddr);// a reference cannot be `const`
    const uint pi = uint(gl_InstanceIndex);  // firstInstance is 0 for this draw

    const vec4 pw = PosBuf(P.posAddr).v[pi];
    // Dead slot. NOT written as (w < 0): a NaN fails every comparison, so the
    // negated form catches garbage as dead too — the same form
    // particlefield_gbuf.vert uses, for the same reason.
    const bool dead = !(pw.w >= 0.0);

    // ── Age, from the emitter's own closed form ─────────────────────────────
    // lifetime == 0 means "this field's positions did not come from the device
    // emitter" (a HostRing field, a sim), and then no age exists: everything
    // that depends on it degrades to the age-0 value, which is the fully bright
    // untapered sprite.
    float ageFrac = 0.0;
    if (P.lifetime > 0.0) {
        const float period = max(P.lifetime * (1.0 + P.lifetimeJitter * rndS(pi, 4u, P.seed)), 1e-4);
        const float life   = max(period * P.duty, 1e-4);
        const float u      = P.time / period + rnd01(pi, 0u, P.seed);
        const float age    = (u - floor(u)) * period;// correct for negative t
        ageFrac = clamp(age / life, 0.0, 1.0);
    }

    // Radius. Under WSemantic::Radius the emitter wrote each particle's own
    // radius into w, which is where SIZE VARIETY comes from at zero cost; under
    // InvMass w says nothing about size and the field's uniformRadius stands in.
    float radius = (((P.flags & 1u) != 0u) ? pw.w : P.uniformRadius) * P.sizeScale;
    radius *= max(1.0 - P.sizeTaper * ageFrac, 0.0);

    const vec3 vp = viewOf(vec4(pw.xyz, 1.0));

    // ── The quad ────────────────────────────────────────────────────────────
    // Corner from two bits of the vertex index, as a triangle strip:
    //   0 (-1,-1)   1 (+1,-1)   2 (-1,+1)   3 (+1,+1)
    const uint vid = uint(gl_VertexIndex);
    const vec2 c   = vec2(((vid & 1u) != 0u) ? 1.0 : -1.0,
                          ((vid & 2u) != 0u) ? 1.0 : -1.0);

    // Screen-plane axes, in view space. The quad is round by default and
    // STRETCHED along the projected velocity when the field asks for it — which
    // is the whole rain streak: a drop at 9 m/s crosses ~20 px in a frame, so
    // drawn as a dot it reads as hail. The velocity is FREE and exact here,
    // because the emitter already wrote f(t) and f(t - dt) into two buffers:
    // (pos - prevPos) IS the frame's displacement, and stretchOverDt turns it
    // into "how many seconds of travel to smear over" without needing dt itself.
    vec2 axisMinor = vec2(1.0, 0.0);
    vec2 axisMajor = vec2(0.0, 1.0);
    float halfMinor = radius;
    float halfMajor = radius;
    if (P.stretchOverDt > 0.0 && !dead) {
        const vec4 pp = PosBuf(P.prevPosAddr).v[pi];
        // A slot reborn between the two samples has prevPos == pos by the
        // emitter's own cycle guard, so this needs no second dead test: the
        // displacement is exactly zero and the quad stays round.
        const vec2 d = (vp.xy - viewOf(vec4(pp.xyz, 1.0)).xy) * P.stretchOverDt;
        const float len = length(d);
        if (len > 1e-7) {
            axisMajor = d / len;
            axisMinor = vec2(-axisMajor.y, axisMajor.x);
            // Capped, because a fast particle near the near plane can project
            // to an arbitrarily long segment and one 2000-px streak across the
            // frame is a bug the eye reads instantly.
            halfMajor = radius + min(len, radius * P.stretchMax);
        }
    }

    vec3 vpos = vp;
    vpos.xy += c.x * axisMinor * halfMinor + c.y * axisMajor * halfMajor;
    // Dead -> collapse the quad onto its own centre, so it has exactly zero
    // area and covers no sample. Same idiom and the same cost (none) as the
    // mesh representation's dead-slot collapse.
    if (dead) vpos = vp;

    vec4 clip = pc.proj * vec4(vpos, 1.0);
    // threepp's projection follows the GL NDC convention (Y up); Vulkan's is
    // Y down, so flip at the clip boundary exactly as every other overlay
    // vertex stage in this renderer does. Reverse-Z already maps z to [0,1].
    clip.y = -clip.y;
    gl_Position = clip;

    // ── Colour ──────────────────────────────────────────────────────────────
    // Hot to cool over the life, with a per-particle offset on WHERE in that
    // ramp the particle sits, so a field of embers is not one colour animating
    // in lockstep. This is the blackbody story the flame's emission ramp tells,
    // told per particle instead of per height.
    const float cf = clamp(ageFrac + 0.40 * (rnd01(pi, 24u, P.seed) - 0.5), 0.0, 1.0);
    const float fade = pow(max(1.0 - ageFrac, 0.0), max(P.fadePower, 0.0));
    // Brightness varies per particle as well as over time: real embers are not
    // 6000 identical lamps, and a hash is the whole cost of saying so.
    const float bj = max(1.0 + P.brightJitter * rndS(pi, 23u, P.seed), 0.0);

    vColor = vec4(mix(P.colorHot, P.colorCool, cf) * (P.intensity * fade * bj),
                  dead ? 0.0 : 1.0);
    vLocal = c;
    vSoft  = clamp(P.softness, 0.0, 1.0);
}
