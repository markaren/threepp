
#include "threepp/extras/effects/FireEffect.hpp"

#include "threepp/math/Rng.hpp"
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

namespace {

    constexpr float kTau = 6.28318530718f;

    // ── Per-slot hash. NO RNG OBJECT, NO STATE ──────────────────────────────
    // A slot's numbers must be a pure function of (seed, slot, stream) so that
    // update(t) can be called at any t, in any order, in any process, and
    // produce the same field. A std::mt19937 walked once per frame would give
    // the same VISUAL result and none of that, and would also make the emitter
    // order-dependent — which is exactly the property the device emitter (plan
    // F2) cannot have, since its threads run in whatever order they run in.
    // The hash itself is math::Rng's counter mode; this alias keeps call
    // sites short.
    inline float rnd01(std::uint32_t seed, std::uint32_t slot, std::uint32_t stream) {
        return math::Rng::hash01(seed, slot, stream);
    }

    // Fractional part in [0,1) for a possibly-negative argument. std::fmod
    // keeps the sign, which would make a negative t run the cycle backwards
    // from a discontinuity at 0.
    inline float frac01(float x) { return x - std::floor(x); }

}// namespace

// ── The blackbody hue ───────────────────────────────────────────────────────
// Same two fitted curves as blackbodyRGB() in shaders/particle_density.glsl,
// same coefficients, WITHOUT that function's (T/2000K)^4 magnitude: a light
// wants a colour and carries its brightness in `intensity`. Red is the maximum
// channel over the whole 700-6500 K range the fit covers, which is why the
// normalisation is simply "red = 1".
Color FireEffect::blackbodyColor(float kelvin) {
    const float s = std::clamp(kelvin, 500.f, 12000.f) * 0.001f;// kilokelvin
    const float g = std::clamp(0.22394f * s - 0.18818f, 0.f, 1.f);
    const float u = std::max(s - 2.09f, 0.f);
    const float b = std::clamp(u * (0.17759f + 0.01179f * u), 0.f, 1.f);
    return Color(1.f, g, b);
}

// ── The flicker ─────────────────────────────────────────────────────────────
// Four sines whose frequencies share no common period, so the sum never
// repeats and has the rough 1/f character firelight has: a slow breathing
// under a fast tremble. Deterministic in t, phase-shifted by the seed, and
// completely stateless — the alternative (a random walk, or filtered noise
// integrated per frame) would be neither seekable nor reproducible, and would
// make a headless capture of frame 600 depend on frames 1..599.
float FireEffect::flickerAt(float timeSec) const {
    if (p_.flicker <= 0.f) return 1.f;
    const float p1 = rnd01(p_.seed, 0u, 101u) * kTau;
    const float p2 = rnd01(p_.seed, 0u, 102u) * kTau;
    const float p3 = rnd01(p_.seed, 0u, 103u) * kTau;
    const float p4 = rnd01(p_.seed, 0u, 104u) * kTau;
    const float f = 0.21f * std::sin(2.13f * timeSec + p1) +
                    0.14f * std::sin(4.79f * timeSec + p2) +
                    0.09f * std::sin(9.31f * timeSec + p3) +
                    0.05f * std::sin(17.57f * timeSec + p4);
    return 1.f + p_.flicker * f;
}

// ── The smoke volume's box: it must CONTAIN the plume ───────────────────────
// A DensityRepr box is not a framing device, it is the domain of the volume:
// particle_density_scatter.comp DROPS a particle outside it. So a plume that
// travels further downwind than the box reaches is not clipped softly, it stops
// — a column of smoke ending on an invisible vertical plane in mid-air. That is
// the geometry behind the "locked in place" report, and it is why this is
// derived from the wind rather than authored.
//
// The far end of the plume sits at wind × periodMax (the longest-lived parcel's
// full advection). An AABB therefore spans the segment from the fire to that
// point, inflated by the column's own widened radius at both ends: centre it at
// half the displacement and give it half the displacement plus the radius as a
// half-extent, per axis. Axis-aligned means a diagonal wind pays for the
// bounding rectangle of a diagonal plume, which is a handful of empty voxels
// and not worth a rotated volume.
//
// FIXED PER WIND VALUE, not fitted per frame to the particles: a box that
// tracked the actual bounds every frame would re-anchor the voxel lattice
// continuously and the medium would swim inside its own grid. It changes only
// when the wind does, which is a rare, deliberate event.
void FireEffect::recomputeSmokeBox() {

    // The longest period any parcel draws (emitSmoke's 3.2 + 2.4·u3, scaled).
    const float periodMax = (3.2f + 2.4f) * std::max(p_.smokeLifeScale, 0.f);
    // The tallest parcel, not the nominal one: smokeHeightJitter lets a few
    // slots overshoot smokeHeight by up to 0.60 × the jitter (emitSmoke's `hs`,
    // whose maximum is 1 + 0.60·j). A box sized to the nominal height would cut
    // those wisps off on a plane — which is the defect the jitter exists to
    // remove, reintroduced through the volume's DOMAIN instead of its emitter.
    const float hsMax     = 1.f + 0.60f * std::max(p_.smokeHeightJitter, 0.f);
    const float smokeTop  = p_.height * 0.80f + p_.smokeHeight * hsMax;
    const float smokeBot  = p_.height * 0.55f;
    // The RADIAL reach, and the max() is a bug fix rather than a flourish. A
    // parcel's distance from the column axis is r0 + wob, whose ceiling at s = 1
    // is `radius·0.55 + smokeSpread` plus the wobble's `0.16·smokeSpread` — so
    // the dispersion term actually reaches 1.16 × smokeSpread, not 1.0 ×. The
    // original `radius + smokeSpread + 0.15` covers that only while the SOURCE
    // radius dominates, which it does for a campfire (0.25 vs 0.096) and does
    // not for a chimney plume authored to spread 2 m (0.22 vs 0.34) — there the
    // widest wobbling parcels fall outside the box and the splat drops them,
    // which is defect 3.2's invisible clipping plane arriving through the SIDE
    // of the volume instead of its downwind end. Taking the larger of the two
    // leaves every existing caller's box at the value it already had.
    const float smokeR    = std::max(p_.radius, p_.radius * 0.55f + p_.smokeSpread * 0.16f) +
                            p_.smokeSpread + 0.15f;
    const float driftX    = p_.wind.x * periodMax;
    const float driftZ    = p_.wind.z * periodMax;
    smokeBoxLocal_ = Vector3(driftX * 0.5f, 0.5f * (smokeBot + smokeTop), driftZ * 0.5f);
    smokeHalf_     = Vector3(smokeR + std::fabs(driftX) * 0.5f,
                             0.5f * (smokeTop - smokeBot),
                             smokeR + std::fabs(driftZ) * 0.5f);
}

// The ember emitter's parameters. Extracted so the constructor and setWind()
// build the SAME block — sparks and smoke leaning different ways inside one
// fire is the exact class of disagreement this whole change is about.
ParticleField::EmitterParams FireEffect::emberEmitter() const {

    ParticleField::EmitterParams e;
    // Born in a thin disc-ish slab over the fuel bed, not through the whole
    // flame volume: sparks are thrown off the burning surface.
    e.spawnCenter.set(0.f, p_.height * 0.16f, 0.f);
    e.spawnHalfExtent.set(p_.radius * 0.55f, p_.height * 0.06f, p_.radius * 0.55f);
    // Buoyant: a spark leaves fast, keeps accelerating in the thermal
    // column, and the wind pushes it the same way the smoke leans.
    e.velocity.set(p_.wind.x, p_.emberRise, p_.wind.z);
    e.speedSpread = p_.emberSpread;
    e.accel.set(0.f, 0.45f, 0.f);
    // driftGrowth 1: the base of the column is steady and the tips wander,
    // which is what turns a cone of sparks into a plume of them.
    e.driftAmplitude = 0.13f;
    e.driftFrequency = 0.55f;
    e.driftGrowth    = 1.f;
    e.driftScale     = 0.7f;
    e.lifetime = std::max(p_.emberLife, 0.05f);
    // Wide, because sparks that all die at the same height give the plume a
    // flat top. With 0.65 the ceiling is spread over 3:1 and the column
    // thins out with height the way the flame's own per-parcel ceiling does.
    e.lifetimeJitter = 0.65f;
    // Duty < 1: a fraction of the slots are dead at any instant, so the
    // spark count BREATHES instead of sitting at a constant N. This is what
    // exercises the w < 0 sentinel in the scene the plan cares about.
    e.dutyCycle  = 0.80f;
    e.size       = std::max(p_.emberSize, 1e-4f);
    e.sizeJitter = std::max(0.f, std::min(1.f, p_.emberSizeJitter));
    e.seed       = p_.seed ^ 0x5bf03635u;
    return e;
}

void FireEffect::setWind(const Vector3& worldWind) {

    p_.wind.copy(worldWind);
    recomputeSmokeBox();
    // The BOX moves; the volume's centre is re-published from it in update(),
    // which is where the effect's world position is also read. Nothing here
    // touches the volume's resolution, which stays latched (churn contract).
    smoke_->densityRepr().halfExtent.copy(smokeHalf_);
    if (embers_) embers_->setEmitter(emberEmitter());
}

FireEffect::FireEffect(const Params& params): p_(params) {

    // ── SMOKE ONLY ──────────────────────────────────────────────────────────
    // Normalised once, here, so every `if (p_.embers ...)` below reads the same
    // truth and no later code has to remember to test two flags. The three
    // things suppressed are exactly the three that cost something to exist
    // rather than to be lit: a density volume holds a slot out of
    // kMaxDensityFields whether or not any particle is alive in it, and a
    // PointLight sits in the cluster list at intensity 0.
    if (p_.smokeOnly) p_.embers = p_.legacyEmbers = false;

    // ── The boxes ───────────────────────────────────────────────────────────
    // FIXED, not fitted per frame to the particles' actual bounds. A box that
    // moved with the flame would re-anchor the voxel lattice every frame and
    // the medium would visibly swim inside its own grid; a fixed, generous box
    // costs only empty voxels. The margins account for the swirl amplitude and
    // the wind drift, which are what actually push a particle outside the
    // nominal envelope.
    const float swirl = 0.22f * p_.radius + 0.05f;
    const float flameR = p_.radius * 1.25f + swirl;
    flameBoxLocal_ = Vector3(0.f, p_.height * 0.5f, 0.f);
    flameHalf_     = Vector3(flameR, p_.height * 0.5f + 0.04f, flameR);

    recomputeSmokeBox();

    // ── The fields. Created ONCE, at final capacity, PARKED ─────────────────
    if (!p_.smokeOnly) {
        ParticleField::Config cfg;
        cfg.capacity      = std::max(p_.flameParticles, 1u);
        cfg.ownership     = ParticleField::Ownership::HostRing;
        cfg.uniformRadius = 0.02f;
        flame_ = ParticleField::create(cfg);
        flame_->name = "fire.flame";
        flame_->setDensityRepr(flameBoxLocal_, flameHalf_, p_.flameSigma, p_.flameResolution);
        auto& dr = flame_->densityRepr();
        dr.albedo            = p_.flameAlbedo;
        dr.anisotropy        = 0.f;
        dr.emissiveIntensity = p_.emissiveIntensity;
        dr.tempBottomK       = p_.tempBottomK;
        dr.tempTopK          = p_.tempTopK;
        dr.tempFalloff       = p_.tempFalloff;
        add(flame_);
    }
    {
        ParticleField::Config cfg;
        cfg.capacity      = std::max(p_.smokeParticles, 1u);
        cfg.ownership     = ParticleField::Ownership::HostRing;
        cfg.uniformRadius = 0.05f;
        smoke_ = ParticleField::create(cfg);
        smoke_->name = "fire.smoke";
        smoke_->setDensityRepr(smokeBoxLocal_, smokeHalf_, p_.smokeSigma, p_.smokeResolution);
        auto& dr = smoke_->densityRepr();
        dr.albedo     = p_.smokeAlbedo;
        dr.anisotropy = p_.smokeAnisotropy;
        // No emission: smoke is the field that proves F0's per-field params
        // were needed. Sharing one albedoAniso would have made it glow warm.
        dr.emissiveIntensity = 0.f;
        add(smoke_);
    }

    if (flame_) flameHost_.resize(flame_->capacity());
    smokeHost_.resize(smoke_->capacity());

    // ── The light: the piece that puts the fire INTO the scene ──────────────
    // Absent under smokeOnly: a stove behind a wall lights nothing outside it,
    // and an intensity-0 PointLight is not free — it is a cluster entry and a
    // shadow-casting light the renderer still has to carry.
    if (!p_.smokeOnly) {
        light_ = PointLight::create(blackbodyColor(p_.lightTempK), p_.lightIntensity,
                                    p_.lightRange, /*decay*/ 2.f);
        light_->name = "fire.light";
        light_->position.set(0.f, p_.lightHeight, 0.f);
        light_->radius     = p_.lightRadius;
        light_->castShadow = true;
        light_->intensity  = 0.f;// parked until ignite()
        add(light_);
    }

    // ── Embers: a third ParticleField, device-emitted, drawn as billboards ──
    //
    // The look this replaces read as "large regular blobs" — a fixed-size soft
    // texture at a fixed brightness, integrated per frame by an RNG. Four things
    // fix that, and all four are properties of the field, not of a texture:
    //
    //   1. SIZE. Centimetre scale, with 60% per-particle jitter written into w
    //      by the emitter, so no two sparks are the same size.
    //   2. BRIGHTNESS. Hashed per particle in the billboard shader (a spark is
    //      not one of N identical lamps) AND faded over the slot's own life.
    //   3. COLOUR. A blackbody ramp from hot to cool over that same life, with
    //      a per-particle offset into the ramp — the same physics the flame's
    //      emission ramp expresses over height, expressed here over time.
    //   4. SHAPE. A slight stretch along the analytic velocity, so a rising
    //      spark reads as a moving thing rather than a floating orb.
    //
    // The age all three of the last ones need is re-derived in the shader from
    // this emitter's own closed form — there is no age buffer and no per-frame
    // CPU work of any kind here.
    if (p_.embers && !p_.legacyEmbers) {
        ParticleField::Config cfg;
        cfg.capacity  = std::max(p_.emberParticles, 1u);
        cfg.ownership = ParticleField::Ownership::Renderer;
        // The emitter writes each spark's own radius into w, which is where the
        // size variety comes from; uniformRadius is the nominal it jitters
        // around and the fallback the billboard uses if the semantic ever
        // changes underneath it.
        cfg.wSemantic     = ParticleField::WSemantic::Radius;
        cfg.uniformRadius = std::max(p_.emberSize, 1e-4f);
        embers_ = ParticleField::create(cfg);
        embers_->name = "fire.embers";

        embers_->setEmitter(emberEmitter());
        embers_->setEmitterTime(0.f, 0.f);

        embers_->setBillboardRepr(blackbodyColor(p_.emberHotK),
                                  blackbodyColor(p_.emberCoolK),
                                  p_.emberIntensity);
        auto& br = embers_->billboardRepr();
        br.texture   = p_.emberTexture;// null = the procedural spark, the default
        br.softness  = 0.34f;          // a spark, not a glow
        br.fadePower = 1.75f;          // holds bright, then goes out
        br.brightJitter    = 0.55f;
        br.sizeTaper       = 0.65f;// a cooling ember gets SMALLER
        br.stretchSeconds  = std::max(p_.emberStretch, 0.f);
        br.stretchMax      = 10.f;
        // ── F4: the spark's own bloom ───────────────────────────────────────
        // Field billboards composite after the upscaler, which is what keeps
        // them clear of TAA/DLSS/FSR — and also what puts them past the scene's
        // bloom pyramid, so until now a spark had no glow but the one its own
        // 3-px falloff could paint. This turns on the billboard-only pyramid:
        // the ember field is drawn a second time into a small linear-HDR target
        // and blurred there. Set it to 0 and not one instruction of that chain
        // is recorded.
        br.glow          = std::max(p_.emberGlow, 0.f);
        br.glowThreshold = std::max(p_.emberGlowThreshold, 0.f);
        // A spark one lens-length from the camera is a hot streak across a
        // quarter of the frame. Both caps are new in F4 and both are about the
        // SAME 1/d compounding, in the two domains it shows up in.
        br.nearFade         = 0.25f;
        br.stretchMaxScreen = 0.06f;
        // Parked until ignite(): a Renderer field's live count is its capacity
        // and is set at construction, so parking is the explicit act here.
        embers_->setLiveCount(0);
        add(embers_);
    }

    // ── Embers, legacy path: the pre-F3 ParticleSystem, verbatim ────────────
    if (p_.embers && p_.legacyEmbers && p_.emberTexture) {
        legacyEmbers_ = std::make_shared<ParticleSystem>();
        auto& s = legacyEmbers_->settings();
        s.makeDefault();
        s.positionStyle  = ParticleSystem::Type::BOX;
        s.positionBase   = Vector3(0.f, p_.height * 0.25f, 0.f);
        s.positionSpread = Vector3(p_.radius * 0.6f, p_.height * 0.15f, p_.radius * 0.6f);
        s.velocityStyle  = ParticleSystem::Type::BOX;
        // Buoyant, with the same wind the smoke column feels so the sparks and
        // the smoke lean the same way.
        s.velocityBase   = Vector3(p_.wind.x, 1.35f, p_.wind.z);
        s.velocitySpread = Vector3(0.45f, 0.75f, 0.45f);
        s.accelerationBase = Vector3(0.f, 0.35f, 0.f);
        s.particlesPerSecond = p_.emberRate;
        s.particleDeathAge   = p_.emberLife;
        s.emitterDeathAge    = 1e9f;
        s.blendStyle         = Blending::Additive;
        s.colorBase          = Vector3(0.06f, 1.0f, 0.62f);// HSL: ember orange
        s.colorSpread        = Vector3(0.02f, 0.f, 0.10f);
        s.texture            = p_.emberTexture;
        // Sparks brighten as they leave the flame, then burn out. Size shrinks
        // — a cooling ember gets smaller, not bigger like smoke.
        // Sizes are WORLD units: an ember is a few pixels at conversational
        // distance, not a glowing pill. The first pass of this shipped at 0.11
        // and rendered 30-px blobs that hid the flame behind them.
        s.setOpacityTween({0.f, 0.25f, p_.emberLife}, {0.f, 0.9f, 0.f})
                .setSizeTween({0.f, p_.emberLife}, {0.030f, 0.008f});
        legacyEmbers_->initialize();
        add(legacyEmbers_);
    }
}

FireEffect::~FireEffect() = default;

std::shared_ptr<FireEffect> FireEffect::create(const Params& params) {
    return std::make_shared<FireEffect>(params);
}

// The defaulted form. It lives here and not as a `= {}` in the header because
// the header is inside FireEffect's own class body, where GCC has not yet
// parsed Params's default member initializers; see the note on the
// declarations.
std::shared_ptr<FireEffect> FireEffect::create() {
    return create(Params{});
}

void FireEffect::ignite() {
    lit_ = true;
    // The two HostRing fields get their live count from the next submit(). The
    // ember field is Ownership::Renderer — no submit exists there, and its
    // count is its capacity — so un-parking it is an explicit call.
    if (embers_) embers_->setLiveCount(embers_->capacity());
    // The light is immediate so a fire ignited and rendered in the same frame
    // already casts.
    if (light_) light_->intensity = p_.lightIntensity;
}

void FireEffect::extinguish() {
    lit_ = false;
    if (flame_) flame_->setLiveCount(0);
    smoke_->setLiveCount(0);
    // Park, don't remove: a field with liveCount 0 stays in the scene at one
    // entry, skips its emit dispatch and draws nothing (churn contract).
    if (embers_) embers_->setLiveCount(0);
    if (light_) light_->intensity = 0.f;
}

// ── The flame ───────────────────────────────────────────────────────────────
// Position = f(seed_i, t), start to finish. Each slot owns a PERIOD and, inside
// it, a shorter LIFE: outside its life the slot writes w < 0 and every consumer
// (the density splat included) drops it on the one shared predicate. The duty
// cycle is not decoration — it is what makes the flame's mass breathe instead
// of sitting at a constant 6000 particles, and it is the reason the sentinel is
// exercised here rather than only in a sim.
void FireEffect::emitFlame(float t) {
    const std::uint32_t n = flame_->capacity();
    const float swirlAmp = 0.22f * p_.radius + 0.05f;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float u0 = rnd01(p_.seed, i, 0u);// birth phase
        const float u1 = rnd01(p_.seed, i, 1u);// radial
        const float u2 = rnd01(p_.seed, i, 2u);// azimuth
        const float u3 = rnd01(p_.seed, i, 3u);// period scale
        const float u4 = rnd01(p_.seed, i, 4u);// swirl phase A
        const float u5 = rnd01(p_.seed, i, 5u);// swirl phase B
        const float u6 = rnd01(p_.seed, i, 6u);// duty
        const float u7 = rnd01(p_.seed, i, 7u);// how high THIS parcel gets

        const float period = 0.55f + 0.55f * u3;      // s, one rise
        const float duty   = 0.80f + 0.20f * u6;      // fraction of it alive
        const float cyc    = frac01(t / period + u0);
        if (cyc >= duty) {                            // dead slot, this instant
            flameHost_[i] = {0.f, 0.f, 0.f, -1.f};
            continue;
        }
        const float s = cyc / duty;// normalised age, [0,1)

        // Buoyant rise: a parcel leaving the fuel bed accelerates, so height
        // is superlinear in age rather than a constant velocity.
        //
        // Each parcel gets its OWN ceiling, and the distribution is skewed low
        // (u7^1.6) so few reach the top. Without this every slot terminates at
        // exactly `height` and the flame renders with a FLAT, hard-edged cap —
        // a red slab sitting on a cone, which is what the first build of this
        // looked like. Varying ceilings make the density thin out gradually
        // with height, which is both what a diffusion flame does and what turns
        // the blackbody ramp's cool end into wisps instead of a plateau.
        const float ceil = 0.42f + 0.58f * std::pow(u7, 1.6f);
        const float y = p_.height * ceil * (0.52f * s + 0.48f * s * s);
        // Necking: the column pinches as it rises and the tips are thin. sqrt
        // on the radial draw is what makes the base disc uniformly covered
        // rather than crowded at the centre.
        const float r0 = p_.radius * std::sqrt(u1);
        const float rr = r0 * (1.04f - 0.80f * std::pow(s, 1.15f));
        const float th = u2 * kTau + 1.30f * s;// twist with height

        // Swirl: phase-shifted sines, amplitude growing with height, so the
        // base is steady and the tips wander — a curl field without a curl
        // field. Incommensurate frequencies again, for the same reason.
        const float w = swirlAmp * s * s;
        const float ph1 = u4 * kTau, ph2 = u5 * kTau;
        const float sx = 0.58f * std::sin(2.30f * t + ph1) +
                         0.42f * std::sin(3.71f * t + ph2 + 1.70f * s);
        const float sz = 0.58f * std::cos(1.93f * t + ph2) +
                         0.42f * std::sin(3.13f * t + ph1 + 2.30f * s);

        flameHost_[i] = {rr * std::cos(th) + w * sx + p_.wind.x * 0.25f * s * s,
                         y,
                         rr * std::sin(th) + w * sz + p_.wind.z * 0.25f * s * s,
                         1.f};
    }
    flame_->submit(flameHost_.data(), n);
}

// ── The smoke ───────────────────────────────────────────────────────────────
// The same closed form with a different profile: a DECELERATING rise (the
// plume cools and slows), a column that widens instead of pinching, and the
// full wind drift, which is what makes the smoke lean away while the flame
// stays put.
//
// ── THE PLUME STREAMS (defect fix, 2026-08-11) ──────────────────────────────
// A parcel is followed for `period × smokeLifeScale` seconds and is advected
// the whole time, so the plume's downwind reach is |wind| × period × scale.
// At scale 1 every expression below is the pre-fix one, multiplied by 1.0f —
// bit-identical, deliberately, so a scene that says nothing gets the campfire
// it had. At the fjord's scale of ~4 with a 0.63 m/s wind the same parcels
// stream 8-14 m downwind, spreading and thinning as they go.
//
// THE DISPERSAL COMES FREE, and it is worth being explicit about why no knob
// was added for it: the density volume conserves total optical mass (the
// trilinear splat is a partition of unity), so spreading the SAME N parcels
// over a longer, wider plume divides the same σ over more voxels. The medium
// thins out downwind by construction, with no per-particle opacity — which the
// density representation could not express anyway.
//
// THE TAPER is the other half. Without it every parcel dies at the same age and
// the plume ends on a plane; `smokeTaper` gives each parcel its own ceiling,
// skewed so only a few reach the end. This is exactly F1 note 2's fix for the
// flame's flat red cap, applied to the downwind end of the other field.
void FireEffect::emitSmoke(float t) {
    const std::uint32_t n = smoke_->capacity();
    const float baseY = p_.height * 0.55f;
    const float lifeScale = std::max(p_.smokeLifeScale, 0.f);
    const float taper     = std::max(0.f, std::min(1.f, p_.smokeTaper));
    const float hJitter   = std::max(p_.smokeHeightJitter, 0.f);
    for (std::uint32_t i = 0; i < n; ++i) {
        const float u0 = rnd01(p_.seed, i, 16u);
        const float u1 = rnd01(p_.seed, i, 17u);
        const float u2 = rnd01(p_.seed, i, 18u);
        const float u3 = rnd01(p_.seed, i, 19u);
        const float u4 = rnd01(p_.seed, i, 20u);
        const float u5 = rnd01(p_.seed, i, 21u);

        const float period = (3.2f + 2.4f * u3) * lifeScale;
        const float cyc    = frac01(t / period + u0);
        // Per-parcel ceiling. taper == 0 leaves ceil at exactly 1.0 and cyc is
        // strictly below 1 by frac01's contract, so the branch is never taken
        // and no slot is lost — the pre-fix "no duty gap, the column is
        // continuous" behaviour, textually.
        if (taper > 0.f) {
            const float u6 = rnd01(p_.seed, i, 22u);
            // u6^1.5 skews the draw LOW, so most parcels give out early and a
            // few carry on: the plume's end frays rather than stopping.
            const float ceil = 1.f - taper * (1.f - std::pow(u6, 1.5f));
            if (cyc >= ceil) {
                smokeHost_[i] = {0.f, 0.f, 0.f, -1.f};
                continue;
            }
        }
        const float s = cyc;

        // ── THREE CLOCKS, NOT ONE — and this is what makes it a plume ───────
        // The pre-fix form drove the rise, the widening AND the wind drift off
        // the same `rise = s^0.82`, which has a consequence that is obvious in
        // hindsight and was invisible at 0.22 m/s: if x and y are both
        // proportional to rise then x/y is CONSTANT and the plume is a perfectly
        // straight ray leaving the fire at a fixed angle. At the fjord's wind it
        // rendered as a collimated white jet — a hard-edged cone, not smoke.
        //
        //   rise  = s^0.62  — buoyancy, DECELERATING: the parcel cools, loses
        //                     its lift and levels off. The exponent came DOWN
        //                     from 0.82 for the same reason: at 0.82 the climb
        //                     is nearly linear, so even with a linear advection
        //                     the plume left the fire at 22 degrees and arrived
        //                     at 17 — a bend nobody can see. A bent-over plume's
        //                     height goes as a low power of time (Briggs puts it
        //                     near 1/3); 0.62 is inside that family and is what
        //                     makes the column go UP first and lie over after.
        //   age   = s·period — advection, LINEAR: the wind does not care how
        //                     old the parcel is, it pushes it at wind speed the
        //                     whole time.
        //   disp  = s^0.55  — turbulent dispersion, ~sqrt(t): a plume keeps
        //                     spreading long after it has stopped climbing.
        //                     Tying the width to the rise made it stop widening
        //                     the moment it levelled — the other half of "jet".
        //
        // With the three separated the column climbs steeply, bends over as the
        // buoyancy gives out, and runs downwind widening and thinning — which is
        // what a chimney plume in wind does, and what the one 30 m away in this
        // same scene has always done.
        // ── AND A FOURTH THING, WHICH IS NOT A CLOCK: the parcel's OWN CEILING
        // `rise` is normalised by the parcel's own age, so it is independent of
        // that parcel's period — which means the three clocks above stagger
        // where a parcel gets to horizontally and stagger NOTHING about where it
        // ends up vertically. Every trajectory in the field topped out at
        // exactly `baseY + smokeHeight`, and a few thousand parcels sharing one
        // ceiling is a plane: the flat, hard-edged top this plume had in every
        // scene that ever drew it, most visibly in vulkan_fire against a night
        // sky. It is F1 note 2's flat red flame cap, in the other field, and it
        // survived the F4 defect pass because the taper — which does fix it —
        // was left opt-in and off.
        //
        // MEAN-PRESERVING, so this is a raggedness knob and not a height knob:
        // E[u^1.5] = 0.4, so E[g] = 0.6 + 1.0 × 0.4 = 1 exactly, and hs is 1 at
        // jitter 0 (the old single ceiling, textually). Skewed low — most
        // parcels fall short of the nominal top, a few overshoot into wisps —
        // which is what makes the density decay toward the top instead of
        // ending at it.
        const float g  = 0.60f + 1.00f * std::pow(rnd01(p_.seed, i, 23u), 1.5f);
        const float hs = 1.f + hJitter * (g - 1.f);

        const float age  = s * period;
        const float rise = std::pow(s, 0.62f);
        const float disp = std::pow(s, 0.55f);
        const float y    = baseY + p_.smokeHeight * hs * rise;
        const float r0 = (p_.radius * 0.55f + p_.smokeSpread * disp) * std::sqrt(u1);
        const float th = u2 * kTau + 0.9f * rise;

        const float ph1 = u4 * kTau, ph2 = u5 * kTau;
        const float wob = 0.16f * p_.smokeSpread * disp;
        const float sx  = std::sin(0.77f * t + ph1 + 2.1f * rise);
        const float sz  = std::cos(0.61f * t + ph2 + 1.7f * rise);

        smokeHost_[i] = {r0 * std::cos(th) + wob * sx + p_.wind.x * age,
                         y,
                         r0 * std::sin(th) + wob * sz + p_.wind.z * age,
                         1.f};
    }
    smoke_->submit(smokeHost_.data(), n);
}

void FireEffect::update(float timeSec) {

    // The delta. The LEGACY ember path needs it because it integrates; the
    // ember FIELD needs it only as the motion-vector interval its emit dispatch
    // evaluates f(t - dt) at. Clamped either way, so a hitch, a pause or a seek
    // neither integrates a hundred sparks into one frame nor asks the emitter
    // for a previous position half a second in the past.
    //
    // A repeated update(t) at the same t therefore gives dt == 0, which freezes
    // the ember field EXACTLY (both evaluations are the same expression, so
    // every spark reprojects onto itself and TAA converges completely) — which
    // is what makes a --t capture of the fire converge WITH the embers in it.
    const float dt = haveTime_ ? std::clamp(timeSec - lastTime_, 0.f, 0.10f) : 0.f;
    lastTime_ = timeSec;
    haveTime_ = true;

    if (!lit_) {
        if (legacyEmbers_) legacyEmbers_->update(dt);// let live sparks burn out
        return;
    }

    if (flame_) emitFlame(timeSec);
    emitSmoke(timeSec);
    // TWO FLOATS — the entire per-frame CPU cost of the ember field, whatever
    // its capacity. Absolute t, never a delta and never a wall clock: the
    // trajectory is closed form, so a headless capture is a function of its
    // frame index and nothing else.
    if (embers_) embers_->setEmitterTime(timeSec, dt);

    // DensityRepr's box is WORLD-space by contract (one volume serves every
    // view, so it cannot be per-field-local), while the positions submitted
    // above are effect-LOCAL and are transformed on the way in by the field's
    // matrixWorld. Both halves therefore have to be told where the effect is,
    // and this is the line that keeps them agreeing when the fire is moved.
    updateMatrixWorld();
    Vector3 origin;
    getWorldPosition(origin);
    if (flame_) flame_->densityRepr().center = flameBoxLocal_ + origin;
    smoke_->densityRepr().center = smokeBoxLocal_ + origin;

    // smokeOnly: no light, and no embers of either kind (the ctor forces both
    // flags off), so the plume above is the whole effect and this is the end.
    if (!light_) return;

    // ── The light ───────────────────────────────────────────────────────────
    // Brightness AND warmth flicker together: a fire that only pulses in
    // brightness reads as a failing bulb. The temperature swing is small (the
    // blackbody hue moves fast in this range) and the emissive body's own ramp
    // is left alone, so the light and the flame never disagree about colour by
    // more than the flicker itself.
    const float f = flickerAt(timeSec);
    light_->intensity = p_.lightIntensity * std::max(f, 0.05f);
    light_->color.copy(blackbodyColor(p_.lightTempK * (0.94f + 0.09f * f)));
    // The apparent source wanders with the flame it stands for — a few
    // centimetres, enough that cast shadows breathe rather than pivot about a
    // fixed point. Same sine family, so it stays a pure function of t.
    const float px = rnd01(p_.seed, 0u, 201u) * kTau;
    const float pz = rnd01(p_.seed, 0u, 202u) * kTau;
    light_->position.set(0.28f * p_.radius * std::sin(2.7f * timeSec + px),
                         p_.lightHeight + 0.10f * p_.height * (f - 1.f),
                         0.28f * p_.radius * std::cos(3.3f * timeSec + pz));

    if (legacyEmbers_) legacyEmbers_->update(dt);
}
