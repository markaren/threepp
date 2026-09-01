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
    // ── F5: the splash ring ─────────────────────────────────────────────────
    // The emitter encodes a landed rain drop's splash phase in the RADIUS it
    // writes to w: it grows the particle from splashR0 to splashR1 over the
    // splash, and nothing else in the lifecycle ever grows one. So w >= R0
    // means "this is a ring" and (w - R0)/(R1 - R0) is how far through. No
    // extra channel, no second buffer, no age re-derivation — the one number
    // that was already there carries it. 0 = this field has no splash and the
    // whole path below is dead code the compiler removes.
    float    splashR0;
    float    splashR1;
    float    splashRingWidth;// annulus width as a fraction of the ring radius
    // ── 4c: the sprite slice ────────────────────────────────────────────────
    float    opacity;       // alphaOver coverage scale
    float    litPhaseG;     // HG asymmetry for the lit lobe
    float    litAmbient;    // ambient share of the lit radiance
    // ── VOLUMETRIC SPRITES (plans/particle-volumetric-sprites) ──────────────
    uint64_t attrAddr;      // Config::attributes; 0 = fall back to the ramp
    vec4     model[3];      // ROWS of the affine field->world matrix
    vec3     boxMin;        // the density volume's world min corner
    vec3     boxInvSize;    // 1 / (2 * halfExtent)
    float    volumeExtinction;// exponent on T_cam; 0 = no camera march
    float    volumeShadow;  // mix toward the sun term; 0 = no sun march
    float    volumeAmbient;
    float    volumeSunGain;
    // R8: this field's slice of the frame's transmittance buffer, written by
    // particlefield_transmit.comp just before this view's draws. 0 when the
    // field marches nothing, which is exactly when kBbVolume is clear.
    uint64_t transAddr;
    float    coreWeight;    // weight of the falloff's t^9 core; 0.85 = the old constant
    float    pad0;
};
// BbParams::flags bits.
const uint kBbRadiusInW = 1u;
const uint kBbAlphaOver = 2u;
const uint kBbLit       = 4u;
const uint kBbAttrs     = 8u; // rgb comes from the attribute buffer, not the ramp
const uint kBbVolume    = 16u;// at least one of the two marches is live

// Per-particle appearance, Config::attributes. rgb = linear HDR radiance in the
// same domain colorHot is authored in; a is reserved for the phase-2 alpha-over
// opacity and is not read here. Same 16 B stride and same buffer_reference
// route as the positions, deliberately: on the interop leg the two are one
// export apiece, snapshotted by the same copy list under the same barrier.
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer AttrBuf { vec4 v[]; };

// ── R8: the transmittance prepass's output ──────────────────────────────────
// One packHalf2x16(T_cam, T_sun) per SLOT, written this frame for THIS view by
// particlefield_transmit.comp. It replaces two eight-step marches that used to
// run in this stage — four times over, since a quad has four vertices and the
// march origin is the particle centre, so every corner recomputed the same two
// numbers. Measured: +9.4 ms at 4M sprites for work that was 75% redundant.
//
// A buffer_reference rather than a descriptor, like everything else here: the
// address is published per field in the record above, so the DRAW side gains
// no set from this change (the COMPUTE side binds the volume it marches).
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer TransBuf { uint v[]; };

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
    // ── 4c: the scene's sun, for BillboardRepr::lit ─────────────────────────
    // The lights UBO is a descriptor set this pass does not own,
    // and the sun is three floats — so the renderer snapshots the brightest
    // scene DirectionalLight (the same one-sun the deferred path shades with)
    // here, per view per frame. Already rotated into VIEW space by the host:
    // this stage has the view-space particle position and no way back to world
    // (only the world-Y ROW of the inverse view is carried), so the one vector
    // that has to change basis is transformed once on the CPU rather than
    // reconstructed per vertex.
    vec3  sunDir;       // VIEW space, TOWARD the sun
    float _pad0;
    vec3  sunRadiance;  // linear, colour x intensity; 0 = the scene has no sun
    float _pad1;
    vec3  ambient;      // the scene's summed AmbientLights
    float _pad2;
    // The volumetric marches' two WORLD-space inputs. sunDir above is already
    // in view space (the lit lobe wants it there), but the density volume is
    // world-anchored, so the marches need the world vector and the world eye.
    vec3  camWorld;     // the eye, world space
    float _pad3;
    vec3  sunDirWorld;  // unit, TOWARD the sun
    float _pad4;
};
const uint kViewFogActive = 1u;
const uint kViewLinearOut = 2u;

// NOTE: this stage sampled the field's r16f density mirror through a set of
// its own until plans/particle-volumetric-sprites R8 moved the marches into
// particlefield_transmit.comp. The image went with them, and with it the ONE
// descriptor this pass ever had beyond the sprite texture — so the billboard
// pipeline is back to set 0 alone and the whole per-field, per-view descriptor
// allocation that bound the volume is gone from the recorder.

layout(push_constant, scalar) uniform Pc {
    mat4     proj;       // 64  view -> clip, UNJITTERED (this is a post-TAA pass)
    vec4     mv[3];      // 48  ROWS of the affine view * model; row-major so a
                         //     view-space position is three dot products and no
                         //     matrix-layout convention has to be agreed on
    uint64_t paramsAddr; //  8  -> this field's BbParams record
    uint64_t viewAddr;   //  8  -> this view's BbView record (exposure,
                         //     toneMapMode and the fog terms ride there —
                         //     the push block itself is full)
} pc;                    // 128 B exactly — the guaranteed minimum push range

layout(location = 0) out vec2  vLocal;// [-1,1]^2 parametric square
layout(location = 1) out vec4  vColor;// rgb = linear HDR radiance, a = coverage
layout(location = 2) out vec2  vSoft; // x = softness, y = core-term weight
// F4: the display transform travels with the vertex now that it lives in the
// per-view record rather than the push block — the fragment stage would
// otherwise have to dereference the same buffer_reference per FRAGMENT to read
// two scalars that are constant over the whole draw.
layout(location = 3) flat out float vExposure;
layout(location = 4) flat out uint  vToneMap;
// F5: > 0 turns the sprite into an ANNULUS of this fractional width — a splash
// ring. 0 is the ordinary soft disc and costs the fragment stage one compare.
layout(location = 5) flat out float vRing;
// 4c: bit0 = composite alpha-over (premultiplied) rather than additive. Flat,
// so the fragment stage's branch is uniform over the whole draw.
layout(location = 6) flat out uint  vMode;
// 4c: the sprite's own coverage scale in alpha-over mode — the field opacity
// times every dimming factor (fog transmittance, near fade, LOD, life fade).
// An occluding sprite fades by becoming transparent, not by becoming dark: a
// dimmed opaque puff reads as a grey puff.
layout(location = 7) flat out float vCover;
// 4c: rotation of the sprite's texture lookup, hashed per slot. Two floats
// rather than an angle so the fragment stage does no trig.
layout(location = 8) flat out vec2  vRot;
const uint kModeAlphaOver = 1u;

// ── The hash ────────────────────────────────────────────────────────────────
// Bit-for-bit particle_emit.comp's hashU/rnd01, which is itself bit-for-bit
// FireEffect.cpp's. Shared because this shader re-derives a particle's age
// from the same closed form the emitter evaluated: there is no age channel in
// the position buffer (w is the radius), and the lifecycle is four lines of
// arithmetic over the same hash.
// KEEP IN SYNC with particle_emit.comp: streams 0 (birth phase) and
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

// Fresnel transmission at the air→water crossing, for a camera ABOVE the
// surface looking down at a submerged particle. The deferred water shader
// mixes (1−F)·transmit for whatever is under the same surface, so without
// this the sprite is the only thing in the frame that crosses the interface
// for free — a few percent looking down, everything toward grazing, where
// the reflection takes the surface and the ropes should vanish exactly as
// the hull does. Flat interface (the waves are centimetres, this acts over
// metres), Schlick with water's r0. 1.0 whenever the leg does not cross from
// the air side, so every other frame is untouched.
float bbWaterCrossingT(BbView V, float partY, float len) {
    if (V.murkDensity <= 0.0 || V.waterSurfaceY >= 1e29) return 1.0;
    if (!(V.camWorldY > V.waterSurfaceY && partY < V.waterSurfaceY)) return 1.0;
    const float cosI = clamp((V.camWorldY - partY) / max(len, 1e-6), 0.0, 1.0);
    const float r0   = 0.02;// ((1.333−1)/(1.333+1))²
    const float m    = 1.0 - cosI;
    const float m2   = m * m;
    return 1.0 - (r0 + (1.0 - r0) * m2 * m2 * m);
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
    float radius = (((P.flags & kBbRadiusInW) != 0u) ? pw.w : P.uniformRadius) * P.sizeScale;
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

    // ── F5: the splash ring ─────────────────────────────────────────────────
    // A landed rain drop has stopped being a drop. The emitter parked it at its
    // landing point (so pos == prevPos and its motion vector is exactly zero,
    // which is why the stretch above already collapsed on its own) and grew its
    // radius from splashR0 to splashR1 — the ONE thing nothing else in the
    // lifecycle does, and therefore the whole detector. What changes here is
    // the quad's ORIENTATION: a splash lies in the ground plane, so instead of
    // the screen-plane axes it is built along the field's own X and Z, taken
    // straight out of the affine (view * model) block as its columns.
    //
    // Field axes rather than world ones, matching the rest of this feature: the
    // emitter's positions are field-local and a rotated weather field is not
    // something surface interaction claims to support (the same documented
    // degradation as the follow wrap's axis-aligned torus).
    float ringHalf = 0.0;
    if (P.splashR0 > 0.0 && !dead && pw.w >= P.splashR0) {
        const float u = clamp((pw.w - P.splashR0) /
                                      max(P.splashR1 - P.splashR0, 1e-6),
                              0.0, 1.0);
        // The taper and the age fade are the DROP's curves and say nothing
        // about a splash; the ring's own life is u, and it dims as it spreads.
        ringHalf  = pw.w * P.sizeScale;
        distFade *= max(1.0 - u, 0.0);
        vRing     = P.splashRingWidth;
    } else {
        vRing = 0.0;
    }

    // Everything that can remove this particle from the frame, in one predicate.
    // A collapsed quad has exactly zero area and covers no sample — the same
    // idiom and the same cost (none) as the mesh representation's dead-slot
    // collapse, and the reason a LOD split needs neither a compaction pass nor a
    // second draw.
    const bool cull = dead || (distFade <= 0.0);

    vec3 vpos = vp;
    if (ringHalf > 0.0) {
        // Columns 0 and 2 of the affine view*model, i.e. the field's X and Z
        // axes expressed in view space. Offsetting in three dimensions is what
        // lays the quad FLAT; the camera-facing path only ever touches .xy.
        const vec3 axX = vec3(pc.mv[0].x, pc.mv[1].x, pc.mv[2].x);
        const vec3 axZ = vec3(pc.mv[0].z, pc.mv[1].z, pc.mv[2].z);
        vpos = vp + (c.x * ringHalf) * axX + (c.y * ringHalf) * axZ;
    } else {
        vpos.xy += c.x * axisMinor * halfMinor + c.y * axisMajor * halfMajor;
    }
    if (cull) vpos = vp;

    vec4 clip = pc.proj * vec4(vpos, 1.0);
    // threepp's projection follows the GL NDC convention (Y up); Vulkan's is
    // Y down, so flip at the clip boundary exactly as every other overlay
    // vertex stage in this renderer does. Reverse-Z already maps z to [0,1].
    clip.y = -clip.y;
    gl_Position = clip;

    // ── Colour ──────────────────────────────────────────────────────────────
    // Hot to cool over the life, with a per-particle offset on where in that
    // ramp the particle sits, so a field of embers is not one colour animating
    // in lockstep — the same blackbody ramp the flame's emission uses, applied
    // per particle instead of per height.
    const float cf = clamp(ageFrac + 0.40 * (rnd01(pi, 24u, P.seed) - 0.5), 0.0, 1.0);
    const float fade = pow(max(1.0 - ageFrac, 0.0), max(P.fadePower, 0.0));
    // Brightness varies per particle as well as over time, so embers are not
    // identical lamps; a hash is the whole cost.
    const float bj = max(1.0 + P.brightJitter * rndS(pi, 23u, P.seed), 0.0);

    // ── F4: fog on the camera leg ───────────────────────────────────────────
    // Transmittance only — the in-scatter term is intentionally omitted. These
    // quads blend additively over a background the deferred pass has already
    // fogged, so the fog's own radiance along this leg is in the frame buffer
    // before the sprite arrives; adding it a second time would double-count it
    // and make distant embers brighter in murk. What an emitter loses to a
    // medium is its own light being extinguished, which is exactly e^(-tau).
    // (The legacy alpha-blended path composites both terms because it replaces
    // the background rather than adding to it — same model, different
    // compositing.)
    //
    // Per particle, not per fragment: a sprite is a handful of pixels wide and
    // the optical depth across it is effectively constant.
    if ((V.flags & kViewFogActive) != 0u) {
        const float partY = dot(V.viewToWorldY, vp) + V.camWorldY;
        const float od    = bbAirOpticalDepth(V, partY, camDist) +
                            bbMurkOpticalDepth(V, partY, camDist);
        distFade *= exp(-od) * bbWaterCrossingT(V, partY, camDist);
    }

    // The GLOW pass draws the same quads a second time into a small offscreen
    // HDR target that the bloom pyramid then runs on. `glow` scales what lands
    // there, so a field can bloom harder or softer than its sprite is bright,
    // and a field with glow == 0 is never drawn into that target at all (the
    // host skips it, so this multiply is never reached with a zero).
    const float outScale = ((V.flags & kViewLinearOut) != 0u) ? P.glow : 1.0;

    // ── 4c: the lit term ────────────────────────────────────────────────────
    // One Henyey-Greenstein lobe about the scene's sun, evaluated per
    // particle. Not a full lighting path: no shadow ray, no cluster walk, no
    // per-fragment work. What it buys is forward-scatter behaviour an emissive
    // sprite cannot have — spray flares when the lens looks through it into
    // the sun and goes flat grey when the sun is behind the camera. The
    // ambient floor is what the shaded side sits at, so the sprite never goes
    // black.
    //
    // Approximation: the particle is an isotropic scattering parcel with no
    // self-shadowing and no sun-to-particle transmittance (a spray puff is
    // optically thin and metres across at most).
    vec3 lit = vec3(1.0);
    if ((P.flags & kBbLit) != 0u) {
        const vec3  rd = (camDist > 1e-6) ? (vp / camDist) : vec3(0.0, 0.0, -1.0);
        const float ct = clamp(dot(rd, V.sunDir), -1.0, 1.0);
        const float g  = P.litPhaseG;
        const float dn = 1.0 + g * g - 2.0 * g * ct;
        // The 1/4pi of the true phase function is folded away: this multiplies
        // an authored radiance, so an absolute normalisation would only mean
        // authoring 12.57x bigger numbers.
        const float hg = (1.0 - g * g) / max(pow(max(dn, 1e-4), 1.5), 1e-4);
        lit = V.ambient + vec3(P.litAmbient) + V.sunRadiance * hg;
    }

    // ── The volumetric terms (R4/R5, prepassed by R8) ───────────────────────
    // Two transmittances through the field's own density volume, both from THIS
    // sprite's position: one toward the eye (what the dust in front of it takes
    // away) and one toward the sun (what the dust between it and the key takes
    // away).
    //
    // Both depend only on the sprite's OWN position, which is the property that
    // keeps this out of the sorting problem entirely: the frame buffer still
    // holds a commutative sum, so nothing has to be ordered and the additive
    // blend is untouched. Occlusion BETWEEN sprites is the phase-2 alpha-over
    // slice; occlusion by the MEDIUM the sprites collectively are is this.
    //
    // ...and it is exactly that "own position" property that makes the marches
    // the wrong thing to do HERE. The origin is the particle centre, so all
    // four corners of the quad marched the same two rays and got the same two
    // numbers — four times the work for one answer. They now run once per slot
    // in particlefield_transmit.comp, dispatched for this view immediately
    // before these draws, and this stage does one load.
    //
    // The whole block is still a uniform branch on the flag, so a field that
    // leaves both knobs at 0 executes not one load and produces bit-identical
    // radiance: `volume` is exactly vec3(1.0) and a multiply by 1.0 is exact in
    // IEEE. That is the same no-op contract DensityRepr::emissiveIntensity has,
    // and R10 extends it to the dispatch: such a field records none.
    vec3 volume = vec3(1.0);
    if ((P.flags & kBbVolume) != 0u) {
        const vec2 T = unpackHalf2x16(TransBuf(P.transAddr).v[pi]);
        if (P.volumeExtinction > 0.0) {
            // An EXPONENT rather than a mix: 1 is the physically honest answer
            // and >1 is the "more dust" grade, without touching
            // DensityRepr::sigmaPerParticle, which the deferred fog march also
            // reads and which therefore cannot be pushed for the sprites alone.
            // Applied HERE and not in the prepass so the stored number stays a
            // transmittance rather than a graded one.
            volume *= pow(max(T.x, 1e-6), P.volumeExtinction);
        }
        if (P.volumeShadow > 0.0) {
            // The phase angle is a dot product of two directions and is
            // therefore basis-free; the VIEW-space pair is already to hand,
            // which is why the lobe stayed in this stage rather than moving
            // into the prepass with the marches.
            const vec3  rdv = (camDist > 1e-6) ? (vp / camDist) : vec3(0.0, 0.0, -1.0);
            const float ct  = clamp(dot(rdv, V.sunDir), -1.0, 1.0);
            const float g   = P.litPhaseG;
            const float dn  = 1.0 + g * g - 2.0 * g * ct;
            const float hg  = (1.0 - g * g) / max(pow(max(dn, 1e-4), 1.5), 1e-4);
            volume *= mix(1.0, T.y * (P.volumeAmbient + P.volumeSunGain * hg),
                          P.volumeShadow);
        }
    }

    // ── Per-particle colour (R3) ────────────────────────────────────────────
    // With attributes present the sprite's rgb IS the simulation's answer and
    // the hot/cool ramp is not read at all — no blend of the two schemes
    // exists. An interop field has no closed form and therefore no age, so
    // without this its billboards can only ever be colorHot at age 0, which is
    // one flat colour for the whole field.
    const vec3 ramp = mix(P.colorHot, P.colorCool, cf);
    const vec3 tint = ((P.flags & kBbAttrs) != 0u)
                              ? AttrBuf(P.attrAddr).v[pi].rgb
                              : ramp;

    // ── 4c: alpha-over vs additive ──────────────────────────────────────────
    // The same radiance, split between the two channels differently. Additive
    // folds every dimming factor into the colour, because the only thing it
    // can do to the frame buffer is add less. Alpha-over folds them into
    // coverage instead: a distant, fogged or dying sprite is more transparent,
    // not darker — the reason this mode exists.
    //
    // The additive branch is the original expression with one `* lit`
    // inserted, and `lit` is exactly vec3(1.0) when the flag is off — a
    // multiply by 1.0 is exact in IEEE, so an unlit additive field still
    // produces bit-identical radiance. That regression contract is why the two
    // branches are not factored.
    const bool alphaOver = (P.flags & kBbAlphaOver) != 0u;
    if (alphaOver) {
        vColor = vec4(tint * lit * volume *
                              (P.intensity * bj * outScale),
                      cull ? 0.0 : 1.0);
        vCover = cull ? 0.0 : (P.opacity * fade * distFade);
        vMode  = kModeAlphaOver;
    } else {
        vColor = vec4(tint * lit * volume *
                              (P.intensity * fade * bj * distFade * outScale),
                      cull ? 0.0 : 1.0);
        vCover = 1.0;
        vMode  = 0u;
    }
    // Rotation of the texture lookup, hashed per slot. Four puff variants
    // drawn at one orientation tile visibly; the same four at a hashed angle
    // do not, and a rotation is two hashed floats against a per-particle atlas
    // index this pass does not carry.
    const float ra = rnd01(pi, 31u, P.seed) * 6.2831853;
    vRot = alphaOver ? vec2(cos(ra), sin(ra)) : vec2(1.0, 0.0);
    vLocal = c;
    vSoft  = vec2(clamp(P.softness, 0.0, 1.0), max(P.coreWeight, 0.0));
    // The display transform. In the glow pass the fragment stage must NOT apply
    // it: that target is linear HDR, which is the domain the bright pass and the
    // 13-tap downsample are defined in. Signalled by a negative exposure rather
    // than a fifth varying — the value is only ever used as an argument to
    // odDisplay(), and no real exposure is negative.
    vExposure = ((V.flags & kViewLinearOut) != 0u) ? -1.0 : V.exposure;
    vToneMap  = V.toneMapMode;
}
