
#include "threepp/extras/effects/FireEffect.hpp"

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
    inline std::uint32_t hashU(std::uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }
    // stream lets one slot draw several independent numbers.
    inline float rnd01(std::uint32_t seed, std::uint32_t slot, std::uint32_t stream) {
        const std::uint32_t h = hashU(slot * 0x9e3779b9u + stream * 0x85ebca6bu + seed);
        // 24 bits -> [0,1). Exact in fp32, and never reaches 1.0.
        return float(h >> 8) * (1.f / 16777216.f);
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

FireEffect::FireEffect(const Params& params): p_(params) {

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

    const float smokeTop = p_.height * 0.80f + p_.smokeHeight;
    const float smokeBot = p_.height * 0.55f;
    const float smokeR   = p_.radius + p_.smokeSpread + 0.15f;
    const float driftX   = std::fabs(p_.wind.x) * p_.smokeHeight * 0.55f;
    const float driftZ   = std::fabs(p_.wind.z) * p_.smokeHeight * 0.55f;
    smokeBoxLocal_ = Vector3(driftX * 0.5f, 0.5f * (smokeBot + smokeTop), driftZ * 0.5f);
    smokeHalf_     = Vector3(smokeR + driftX * 0.5f, 0.5f * (smokeTop - smokeBot),
                             smokeR + driftZ * 0.5f);

    // ── The fields. Created ONCE, at final capacity, PARKED ─────────────────
    {
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

    flameHost_.resize(flame_->capacity());
    smokeHost_.resize(smoke_->capacity());

    // ── The light: the piece that puts the fire INTO the scene ──────────────
    light_ = PointLight::create(blackbodyColor(p_.lightTempK), p_.lightIntensity,
                                p_.lightRange, /*decay*/ 2.f);
    light_->name = "fire.light";
    light_->position.set(0.f, p_.lightHeight, 0.f);
    light_->radius     = p_.lightRadius;
    light_->castShadow = true;
    light_->intensity  = 0.f;// parked until ignite()
    add(light_);

    // ── Embers: the legacy billboard path, verbatim and untouched ───────────
    if (p_.embers && p_.emberTexture) {
        embers_ = std::make_shared<ParticleSystem>();
        auto& s = embers_->settings();
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
        embers_->initialize();
        add(embers_);
    }
}

FireEffect::~FireEffect() = default;

std::shared_ptr<FireEffect> FireEffect::create(const Params& params) {
    return std::make_shared<FireEffect>(params);
}

void FireEffect::ignite() {
    lit_ = true;
    // liveCount is set by the next submit(); the light is immediate so a fire
    // ignited and rendered in the same frame already casts.
    light_->intensity = p_.lightIntensity;
}

void FireEffect::extinguish() {
    lit_ = false;
    flame_->setLiveCount(0);
    smoke_->setLiveCount(0);
    light_->intensity = 0.f;
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
void FireEffect::emitSmoke(float t) {
    const std::uint32_t n = smoke_->capacity();
    const float baseY = p_.height * 0.55f;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float u0 = rnd01(p_.seed, i, 16u);
        const float u1 = rnd01(p_.seed, i, 17u);
        const float u2 = rnd01(p_.seed, i, 18u);
        const float u3 = rnd01(p_.seed, i, 19u);
        const float u4 = rnd01(p_.seed, i, 20u);
        const float u5 = rnd01(p_.seed, i, 21u);

        const float period = 3.2f + 2.4f * u3;
        const float cyc    = frac01(t / period + u0);
        const float s      = cyc;// smoke has no duty gap: the column is continuous

        const float rise = std::pow(s, 0.82f);
        const float y    = baseY + p_.smokeHeight * rise;
        // Widens with height AND fades out at the very top, where the plume has
        // spread thin enough to disappear into the air.
        const float r0 = (p_.radius * 0.55f + p_.smokeSpread * rise) * std::sqrt(u1);
        const float th = u2 * kTau + 0.9f * rise;

        const float ph1 = u4 * kTau, ph2 = u5 * kTau;
        const float wob = 0.16f * p_.smokeSpread * rise;
        const float sx  = std::sin(0.77f * t + ph1 + 2.1f * rise);
        const float sz  = std::cos(0.61f * t + ph2 + 1.7f * rise);

        smokeHost_[i] = {r0 * std::cos(th) + wob * sx + p_.wind.x * period * rise,
                         y,
                         r0 * std::sin(th) + wob * sz + p_.wind.z * period * rise,
                         1.f};
    }
    smoke_->submit(smokeHost_.data(), n);
}

void FireEffect::update(float timeSec) {

    // The legacy embers are the only stateful piece; they need a delta. Clamp
    // it so a hitch, a pause or a seek does not integrate a hundred sparks into
    // one frame.
    const float dt = haveTime_ ? std::clamp(timeSec - lastTime_, 0.f, 0.10f) : 0.f;
    lastTime_ = timeSec;
    haveTime_ = true;

    if (!lit_) {
        if (embers_) embers_->update(dt);// let live sparks finish burning out
        return;
    }

    emitFlame(timeSec);
    emitSmoke(timeSec);

    // DensityRepr's box is WORLD-space by contract (one volume serves every
    // view, so it cannot be per-field-local), while the positions submitted
    // above are effect-LOCAL and are transformed on the way in by the field's
    // matrixWorld. Both halves therefore have to be told where the effect is,
    // and this is the line that keeps them agreeing when the fire is moved.
    updateMatrixWorld();
    Vector3 origin;
    getWorldPosition(origin);
    flame_->densityRepr().center = flameBoxLocal_ + origin;
    smoke_->densityRepr().center = smokeBoxLocal_ + origin;

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

    if (embers_) embers_->update(dt);
}
