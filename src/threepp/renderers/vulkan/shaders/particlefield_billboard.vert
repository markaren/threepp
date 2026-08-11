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
    // ── F4 ──────────────────────────────────────────────────────────────────
    float    glow;          // > 0: this field feeds the offscreen glow pyramid
    float    stretchMaxScreen;// streak cap as a FRACTION OF FRAME HEIGHT; 0 = off
    float    nearFade;      // fade the sprite out below this camera distance, m
    float    lodNear;       // collapse the quad CLOSER than this (mesh draws it)
    float    lodFade;       // metres of ramp above lodNear
    uint     _pad0;
    uint     _pad1;
};

// ── F4: the per-VIEW record ─────────────────────────────────────────────────
// Everything here is a property of the CAMERA and the FRAME, not of the field,
// and it is a second buffer_reference rather than more push constants for one
// arithmetic reason: the push block was already exactly 128 B — the range every
// Vulkan implementation guarantees — and the fog model needs a dozen floats.
// It is not a descriptor for the reason the whole pass is not: a set written
// per view per frame is precisely the VUID-03047 exposure this design avoids.
//
// The renderer writes one record per (view, output mode) into a per-frame-in-
// flight host-visible block during recording of that frame, which is safe for
// the same reason ParticleFieldPass::prepareFrame's writes are: this slot's
// fence has already been waited on, so nothing in flight can be reading it.
// KEEP IN SYNC with BillboardViewGpu in ParticleFieldPass.hpp.
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer BbView {
    float exposure;     // currentExposure(), for the display transform
    uint  toneMapMode;  // threepp::ToneMapping
    uint  flags;        // bit0 = a fog medium is present, bit1 = LINEAR HDR out
    float hfDensity;    // air-medium sigma_t at baseY
    float hfBaseY;
    float hfFalloff;    // exponential height scale, metres
    float murkDensity;  // underwater murk sigma_t
    float waterSurfaceY;// world Y of the water surface; >= 1e29 = no clip
    float camWorldY;
    vec3  viewToWorldY; // world-Y row of the inverse view; reconstructs a
                        // particle's world Y from its VIEW-space position
};
const uint kViewFogActive = 1u;
const uint kViewLinearOut = 2u;

layout(push_constant, scalar) uniform Pc {
    mat4     proj;       // 64  view -> clip, UNJITTERED (this is a post-TAA pass)
    vec4     mv[3];      // 48  ROWS of the affine view * model; row-major so a
                         //     view-space position is three dot products and no
                         //     matrix-layout convention has to be agreed on
    uint64_t paramsAddr; //  8  -> this field's BbParams record
    uint64_t viewAddr;   //  8  -> this view's BbView record (F4; used to be the
                         //     exposure/toneMapMode pair, which moved INTO that
                         //     record to make room for the fog terms)
} pc;                    // 128 B exactly — the guaranteed minimum push range

layout(location = 0) out vec2  vLocal;// [-1,1]^2 parametric square
layout(location = 1) out vec4  vColor;// rgb = linear HDR radiance, a = coverage
layout(location = 2) out float vSoft;
// F4: the display transform travels with the vertex now that it lives in the
// per-view record rather than the push block — the fragment stage would
// otherwise have to dereference the same buffer_reference per FRAGMENT to read
// two scalars that are constant over the whole draw.
layout(location = 3) flat out float vExposure;
layout(location = 4) flat out uint  vToneMap;

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

// ── F4: fog attenuation on a billboard ──────────────────────────────────────
// A DELIBERATE SECOND COPY of particle.frag's overlay-fog closed forms, which
// are themselves in sync with heightFogOpticalDepth in
// deferred_shade_60_fog_volumetrics.glsl. Copied rather than shared because the
// legacy billboard path must stay byte-identical (the parent plan requires it
// untouched) and folding these onto a header would recompile it; the same
// deliberate-duplication call F3 note 6 made for the display curves. If a third
// copy ever appears, that is the moment to merge all of them.
//
// Numerically-stable (e^a − e^b)/x form: the plain difference of exponentials
// cancels catastrophically in fp32 when the falloff is large (near-uniform fog),
// which is exactly the fjord's murk.
float bbAirOpticalDepth(BbView V, float partY, float len) {
    if (V.hfDensity <= 0.0) return 0.0;
    const float H  = max(V.hfFalloff, 1e-3);
    const float ya = max(V.camWorldY - V.hfBaseY, 0.0);
    const float yb = max(partY       - V.hfBaseY, 0.0);
    const float clampedLen = min(len, 1.0e7);
    const float ea = exp(-ya / H);
    const float eb = exp(-yb / H);
    const float x  = (yb - ya) / H;
    const float f  = (abs(x) < 1e-3) ? (ea * (1.0 - 0.5 * x + x * x * (1.0 / 6.0)))
                                     : ((ea - eb) / x);
    return min(V.hfDensity * clampedLen * f, 80.0);
}

// Homogeneous murk over the BELOW-waterSurfaceY portion of the leg only.
float bbMurkOpticalDepth(BbView V, float partY, float len) {
    if (V.murkDensity <= 0.0) return 0.0;
    float d = len;
    if (V.waterSurfaceY < 1e29) {
        const float ya = V.camWorldY - V.waterSurfaceY;
        const float yb = partY       - V.waterSurfaceY;
        if (ya >= 0.0 && yb >= 0.0) d = 0.0;
        else if (!(ya < 0.0 && yb < 0.0)) {
            const float t = ya / (ya - yb);
            d *= (ya < 0.0) ? t : (1.0 - t);
        }
    }
    return V.murkDensity * d;
}

void main() {
    BbParams   P  = BbParams(pc.paramsAddr);// a reference cannot be `const`
    BbView     V  = BbView(pc.viewAddr);
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
    // Distance to the eye. For a perspective camera this is the literal camera
    // distance; for an orthographic one it is the distance to the view origin,
    // which is the only thing "near" can mean there and is what every
    // distance-driven term below wants.
    const float camDist = length(vp);

    // ── F4: the LOD gate and the near fade, as ONE brightness factor ────────
    // Both are ramps on camDist and both end up multiplying the sprite's
    // radiance, so they are computed together and applied once.
    //
    // lodNear is the NEAR half of the mesh/billboard split: inside it the field's
    // MeshRepr is drawing this same particle as a shaded solid (its
    // MeshRepr::lodFar shrinks it out over the SAME band, so the two cross-fade
    // rather than pop), and drawing the quad there as well would double the
    // flake. Outside it, the quad IS the particle and the proxy is gone. One
    // field, one position buffer, two vertex stages with complementary
    // predicates — no CPU, no compaction, no second field.
    float distFade = 1.0;
    if (P.lodNear > 0.0) {
        distFade *= (P.lodFade > 1e-4) ? clamp((camDist - P.lodNear) / P.lodFade, 0.0, 1.0)
                                       : ((camDist >= P.lodNear) ? 1.0 : 0.0);
    }
    // The near fade. An additive quad in a field the camera stands INSIDE
    // compounds with proximity twice over — coverage grows as 1/d^2 and a
    // stretched streak grows as 1/d on top of that — so the single closest
    // particle is reliably the brightest thing in the frame however modest the
    // authored intensity is. That is the "one anomalously bright near streak per
    // frame" of the F3 rain sequence, and this is its cause rather than its
    // symptom: it is not a bug in the stretch, it is 1/d.
    if (P.nearFade > 0.0) distFade *= clamp(camDist / P.nearFade, 0.0, 1.0);

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

    // ── F4: the streak cap IN THE DOMAIN THE DEFECT LIVES IN ────────────────
    // stretchMax above is expressed in RADII, i.e. in metres, so it bounds a
    // streak's length in the world and says nothing about how much of the FRAME
    // that becomes. A drop two metres from the lens projects its perfectly legal
    // 12 cm to a bar across a quarter of the image. This clamps the projected
    // half-length against a fraction of the frame height, measured by projecting
    // the axis itself — exact for perspective and orthographic alike, with no
    // small-angle approximation and no aspect ratio to plumb in (the x component
    // is converted to y-equivalent units by the projection's own anisotropy).
    if (P.stretchMaxScreen > 0.0 && halfMajor > radius) {
        const vec4 c0 = pc.proj * vec4(vp, 1.0);
        const vec4 c1 = pc.proj * vec4(vp + vec3(axisMajor * halfMajor, 0.0), 1.0);
        if (c0.w > 1e-4 && c1.w > 1e-4) {
            vec2 dn = c1.xy / c1.w - c0.xy / c0.w;
            dn.x *= pc.proj[1][1] / max(abs(pc.proj[0][0]), 1e-6);
            const float lenNdc = length(dn);
            // NDC spans 2 over the frame height, so a half-length of `s` in NDC
            // is a full length of `s` frame heights.
            if (lenNdc > P.stretchMaxScreen)
                halfMajor = max(radius, halfMajor * (P.stretchMaxScreen / lenNdc));
        }
    }

    // Everything that can remove this particle from the frame, in one predicate.
    // A collapsed quad has exactly zero area and covers no sample — the same
    // idiom and the same cost (none) as the mesh representation's dead-slot
    // collapse, and the reason a LOD split needs neither a compaction pass nor a
    // second draw.
    const bool cull = dead || (distFade <= 0.0);

    vec3 vpos = vp;
    vpos.xy += c.x * axisMinor * halfMinor + c.y * axisMajor * halfMajor;
    if (cull) vpos = vp;

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

    // ── F4: fog on the camera leg ───────────────────────────────────────────
    // TRANSMITTANCE ONLY, and the omission of the in-scatter term is the point
    // rather than a shortcut. These quads blend ADDITIVELY over a background the
    // deferred pass has ALREADY fogged, so the fog's own radiance along this leg
    // is in the frame buffer before the sprite arrives; adding it a second time
    // would double-count it and make distant embers BRIGHTER in murk, which is
    // the opposite of the effect. What an emitter loses to a medium is its own
    // light being extinguished, and that is exactly e^(-tau). (The legacy
    // alpha-blended path composites both terms because it REPLACES the
    // background rather than adding to it — same model, different compositing.)
    //
    // Per PARTICLE, not per fragment: a sprite is a handful of pixels wide and
    // the optical depth across it is constant to many decimal places.
    if ((V.flags & kViewFogActive) != 0u) {
        const float partY = dot(V.viewToWorldY, vp) + V.camWorldY;
        const float od    = bbAirOpticalDepth(V, partY, camDist) +
                            bbMurkOpticalDepth(V, partY, camDist);
        distFade *= exp(-od);
    }

    // The GLOW pass draws the same quads a second time into a small offscreen
    // HDR target that the bloom pyramid then runs on. `glow` scales what lands
    // there, so a field can bloom harder or softer than its sprite is bright,
    // and a field with glow == 0 is never drawn into that target at all (the
    // host skips it, so this multiply is never reached with a zero).
    const float outScale = ((V.flags & kViewLinearOut) != 0u) ? P.glow : 1.0;

    vColor = vec4(mix(P.colorHot, P.colorCool, cf) *
                          (P.intensity * fade * bj * distFade * outScale),
                  cull ? 0.0 : 1.0);
    vLocal = c;
    vSoft  = clamp(P.softness, 0.0, 1.0);
    // The display transform. In the glow pass the fragment stage must NOT apply
    // it: that target is linear HDR, which is the domain the bright pass and the
    // 13-tap downsample are defined in. Signalled by a negative exposure rather
    // than a fifth varying — the value is only ever used as an argument to
    // odDisplay(), and no real exposure is negative.
    vExposure = ((V.flags & kViewLinearOut) != 0u) ? -1.0 : V.exposure;
    vToneMap  = V.toneMapMode;
}
