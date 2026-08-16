
#include "threepp/extras/uav/DownwashEffect.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Rng.hpp"
#include "threepp/utils/Parallel.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;
using namespace threepp::uav;

namespace {

    constexpr float kTau = 6.28318530718f;

    // Pure function of (seed, slot, stream) — FireEffect's hash, same reasons.
    inline float rnd01(std::uint32_t seed, std::uint32_t slot, std::uint32_t stream) {
        return math::Rng::hash01(seed, slot, stream);
    }

    // How long a parked slot waits before it looks at the air again. Short
    // enough that a flare of throttle lights the field within a couple of
    // frames, long enough that a parked field is ~5 checks/second/slot.
    constexpr float kRetrySeconds = 0.20f;

    // Slots per puff (the lobe unit — see the PUFFS note in update()).
    constexpr std::uint32_t kPuffSlots = 64u;

}// namespace

std::shared_ptr<DownwashEffect> DownwashEffect::create(const Params& params) {
    return std::make_shared<DownwashEffect>(params);
}

std::shared_ptr<DownwashEffect> DownwashEffect::create() {
    return create(Params{});
}

DownwashEffect::DownwashEffect(const Params& params): p_(params) {

    ParticleField::Config cfg;
    cfg.capacity = p_.capacity;
    cfg.ownership = ParticleField::Ownership::HostRing;
    field_ = ParticleField::create(cfg);
    field_->name = "downwashDust";

    // Placeholder box; the first update() recentres it on the real source.
    // resolution is LATCHED here (ParticleField contract), the rest is live.
    field_->setDensityRepr(Vector3(0.f, 2.f, 0.f),
                           Vector3(p_.maxRadius * 1.3f, p_.ringHeight * 1.9f,
                                   p_.maxRadius * 1.3f),
                           p_.sigmaPerParticle, p_.resolution);
    field_->densityRepr().albedo = p_.albedo;
    field_->densityRepr().anisotropy = p_.anisotropy;
    // Visually nil, deliberately non-zero: any emissive field flips the dust
    // march onto its emissive path — 32 steps and the per-pixel step DITHER
    // (particle-atmosphere F0). A pure-dust march is 24 unjittered steps, and
    // through a thin ground sheet seen at grazing angles those steps render
    // as diagonal onion-shell banding — exactly the high-contrast case the
    // dither was built for, reached through the per-field API it shipped on.
    field_->densityRepr().emissiveIntensity = 1e-3f;
    field_->setLiveCount(0);// parked until the air says otherwise
    add(field_);

    host_.resize(p_.capacity);
    birth_.assign(p_.capacity, -1e9f);// every slot is due a retry immediately
    strength_.assign(p_.capacity, 0.f);
    ax_.assign(p_.capacity, 0.f);
    az_.assign(p_.capacity, 0.f);
    gy_.assign(p_.capacity, 0.f);
    cycle_.assign(p_.capacity, 0u);

    // Per-slot life is STATIC (stream 3), so bake it once: the serial
    // lifecycle pass tests every slot every frame and should not re-hash.
    lifeI_.resize(p_.capacity);
    for (std::uint32_t i = 0; i < p_.capacity; ++i) {
        const float u3 = rnd01(p_.seed, i, 3u);
        lifeI_[i] = p_.life * (1.f + p_.lifeJitter * (2.f * u3 - 1.f));
    }
}

void DownwashEffect::setWind(const Vector3& worldWind) {
    wind_ = worldWind;
    // Slow EMA of the wind magnitude, for the BOX only: the box must contain
    // the downwind drift (FireEffect::recomputeSmokeBox, same defect class)
    // but must not breathe with every gust — parcels take the instantaneous
    // wind, the box takes the average. update() advances the EMA (it owns the
    // clock); here we just re-derive the box from the stored anchor.
    if (boxValid_) {
        recomputeBox(boxAnchorLocal_.x, boxAnchorLocal_.z, boxAnchorLocal_.y);
    }
}

void DownwashEffect::recomputeBox(float srcX, float srcZ, float groundLocalY) {
    // EVERYTHING here is quantized, and that is the whole point: setWind is
    // called per frame with the GUSTY wind, and a box whose size or centre
    // rides the gust sines re-phases the entire volume against its own
    // lattice every frame — the cloud visibly pumps in sync with the gust
    // envelope, which the user read (correctly) as "wavelike". The parcels
    // ride the gusts; the BOX sizes off the slow-averaged wind and moves only
    // in discrete, rare steps.
    const float windXZ = windSlow_;
    const float halfX = std::ceil((p_.maxRadius * 1.30f + windXZ * p_.life * 0.7f) / 1.5f) * 1.5f;
    const float halfY = std::max(3.2f, p_.ringHeight * 1.9f);

    // Snap the anchor to a two-voxel lattice: a per-frame-fitted box re-phases
    // the volume against its own grid and the haze visibly swims (the same
    // error EmitterParams::followSnap exists to prevent).
    const float snap = std::max(0.25f, 4.f * halfX / float(p_.resolution));
    boxAnchorLocal_.set(std::round(srcX / snap) * snap, groundLocalY,
                        std::round(srcZ / snap) * snap);

    // Centre: half a box above the ground (dust lives from the ground up).
    // Nudged downwind so the tail gets the extra room, not the upwind side —
    // snapped to the SAME lattice as the anchor, for the same reason.
    Vector3 center(boxAnchorLocal_.x, groundLocalY + halfY - 0.6f,
                   boxAnchorLocal_.z);
    if (windXZ > 0.05f) {
        const float shift = std::min(halfX * 0.25f, windXZ * p_.life * 0.35f);
        center.x += std::round(wind_.x / windXZ * shift / snap) * snap;
        center.z += std::round(wind_.z / windXZ * shift / snap) * snap;
    }

    Object3D::updateWorldMatrix(true, false);
    Vector3 wp;
    getWorldPosition(wp);
    field_->densityRepr().center = Vector3(wp.x + center.x, wp.y + center.y,
                                           wp.z + center.z);
    field_->densityRepr().halfExtent = Vector3(halfX, halfY, halfX);
    boxValid_ = true;
}

// ── The ground reservoir ────────────────────────────────────────────────────
// Dust is CONSERVED: the ground under the site carries a finite mass of loose
// soil (a 2D grid of parcel quanta), entrainment ERODES one quantum from the
// cell under the parcel's birth anchor — a cell with nothing left spawns
// nothing — and a parcel that dies DEPOSITS its quantum in the cell it was
// over. Dust is genuinely moved from A to B: a hover slowly blows its pad
// clean (the brownout fades as the loose soil is exhausted — exactly what
// repeated helicopter landings do), and the downwind band gets thicker.
//
// The grid is anchored ONCE, at the first entrainment, and never moves: mass
// that teleports with a re-anchored grid is not conserved. Positions outside
// the grid clamp to the border cells, so mass blown past the edge piles up
// there instead of vanishing — an accepted border approximation, stated here.

void DownwashEffect::initGrid(float srcX, float srcZ) {
    const float half = p_.maxRadius * 1.30f + 8.f;// room for the downwind drift
    gridN_ = std::max(8, static_cast<int>(std::ceil(2.f * half / p_.gridCell)));
    gridAnchorX_ = srcX - 0.5f * static_cast<float>(gridN_) * p_.gridCell;
    gridAnchorZ_ = srcZ - 0.5f * static_cast<float>(gridN_) * p_.gridCell;
    const float perCell = p_.groundDustPerM2 * p_.gridCell * p_.gridCell;
    grid_.assign(static_cast<std::size_t>(gridN_) * gridN_, perCell);
    groundInitial_ = static_cast<double>(perCell) * static_cast<double>(grid_.size());
    groundTotal_ = groundInitial_;
    gridValid_ = true;
}

std::size_t DownwashEffect::cellOf(float x, float z) const {
    const auto ix = std::clamp(
            static_cast<int>(std::floor((x - gridAnchorX_) / p_.gridCell)), 0, gridN_ - 1);
    const auto iz = std::clamp(
            static_cast<int>(std::floor((z - gridAnchorZ_) / p_.gridCell)), 0, gridN_ - 1);
    return static_cast<std::size_t>(iz) * gridN_ + ix;
}

float DownwashEffect::groundDustAt(float localX, float localZ) const {
    if (!gridValid_) return p_.groundDustPerM2 * p_.gridCell * p_.gridCell;
    return grid_[cellOf(localX, localZ)];
}

void DownwashEffect::update(float t, const Vector3& dronePosWorld,
                            float thrust01, float aglMetres) {

    Object3D::updateWorldMatrix(true, false);
    Vector3 wp;
    getWorldPosition(wp);
    const Vector3 drone(dronePosWorld.x - wp.x, dronePosWorld.y - wp.y,
                        dronePosWorld.z - wp.z);
    const float groundY = drone.y - aglMetres;

    // Advance the slow wind average the box is sized from (τ ≈ 3 s).
    {
        const float dt = haveT_ ? std::clamp(t - lastT_, 0.f, 0.25f) : 0.f;
        const float mag = std::hypot(wind_.x, wind_.z);
        windSlow_ = haveT_ ? windSlow_ + (mag - windSlow_) * std::min(dt / 3.f, 1.f) : mag;
        lastT_ = t;
        haveT_ = true;
    }

    // ── Entrainment strength: thrust x ground proximity ─────────────────────
    const float prox = 1.f - math::smoothstep(p_.fullAgl, p_.engageAgl,
                                              std::isfinite(aglMetres) ? aglMetres : 1e9f);
    const float strength =
            std::min(1.3f, std::max(0.f, thrust01) * p_.thrustGain) * prox;
    strengthNow_ = strength;

    // ── Box + reservoir anchoring ───────────────────────────────────────────
    if (strength > 0.02f) {
        if (!gridValid_) initGrid(drone.x, drone.z);
        const float dx = drone.x - boxAnchorLocal_.x;
        const float dz = drone.z - boxAnchorLocal_.z;
        if (!boxValid_ || std::hypot(dx, dz) > p_.maxRadius * 0.45f) {
            recomputeBox(drone.x, drone.z, groundY);
        }
    }

    // ── Pass A (SERIAL): lifecycle + the conserved reservoir ────────────────
    // Every erode/deposit mutates the shared grid, so this pass stays serial
    // and in slot order — which is also what keeps the effect deterministic:
    // when a cell is nearly empty, WHICH parcel gets the last quantum must not
    // depend on thread scheduling.
    std::uint32_t alive = 0;
    const std::uint32_t n = p_.capacity;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float lifeI = lifeI_[i];
        if (t - birth_[i] < lifeI) {
            if (strength_[i] > 0.f) ++alive;
            continue;// mid-flight or parked-not-yet-due
        }

        // The slot cycles NOW. A parcel that was airborne gives its quantum
        // BACK to the ground it is over — this is the "moved from A to B".
        if (strength_[i] > 0.f && gridValid_) {
            grid_[cellOf(host_[i].x, host_[i].z)] += 1.f;
            groundTotal_ += 1.0;
        }

        const float accept = rnd01(p_.seed + cycle_[i], i, 9u);
        ++cycle_[i];
        // Per-PUFF duty, re-drawn every ~1.7 s — with a per-puff EPOCH PHASE.
        // A shared epoch boundary re-rolled every lobe on one global clock and
        // the whole ring pulsed in sync ("wavelike", the user's exact word).
        const std::uint32_t puffId = i / kPuffSlots;
        const auto epoch = static_cast<std::uint32_t>(
                t / 1.7f + rnd01(p_.seed ^ 0x51ED5EEDu, puffId, 0u));
        const float puffDuty =
                0.35f + 0.85f * rnd01(p_.seed ^ 0xABCD1234u, puffId, epoch);

        bool spawned = false;
        if (strength > 0.02f && accept < strength * 1.25f * puffDuty) {
            // Where the parcel WOULD be born, on the impingement annulus.
            const float uR = rnd01(p_.seed + cycle_[i], i, 1u);
            const float uAa = rnd01(p_.seed + cycle_[i], i, 2u);
            const float r0 = p_.sourceRadius * (0.35f + 0.75f * std::sqrt(uR));
            const float a0 = uAa * kTau;
            const float bx = drone.x + std::cos(a0) * r0;
            const float bz = drone.z + std::sin(a0) * r0;
            // ...and only if the ground THERE still has loose soil to give.
            // A depleted cell spawns nothing; the retry redraws the azimuth,
            // so entrainment migrates to wherever dust remains.
            auto& cell = grid_[cellOf(bx, bz)];
            if (cell >= 1.f) {
                cell -= 1.f;
                groundTotal_ -= 1.0;
                birth_[i] = t;
                strength_[i] = strength;
                ax_[i] = bx;
                az_[i] = bz;
                gy_[i] = groundY;
                spawned = true;
                ++alive;
            }
        }
        if (!spawned) {
            birth_[i] = t - lifeI + kRetrySeconds;// retry soon, not next period
            strength_[i] = 0.f;
            host_[i] = {0.f, 0.f, 0.f, -1.f};
        }
    }
    alive_ = alive;

    if (alive == 0) {
        field_->setLiveCount(0);// parked: skip the scatter entirely
        return;
    }

    // ── Pass B (PARALLEL): closed-form positions for the airborne ───────────
    // Disjoint per-slot writes only (host_[i]); the lifecycle above already
    // settled who is alive, so the serial fallback is bit-identical
    // (utils/Parallel.hpp contract).
    constexpr std::uint32_t kChunk = 4096;
    const std::uint32_t chunks = (n + kChunk - 1) / kChunk;
    if (chunkIdx_.size() != chunks) {
        chunkIdx_.resize(chunks);
        for (std::uint32_t c = 0; c < chunks; ++c) chunkIdx_[c] = c;
    }
    parallelForEach(chunkIdx_.begin(), chunkIdx_.end(), [&](std::uint32_t c) {
        const std::uint32_t iEnd = std::min(n, (c + 1) * kChunk);
        for (std::uint32_t i = c * kChunk; i < iEnd; ++i) {
            if (strength_[i] <= 0.f) continue;// parked; sentinel already set

            const float lifeI = lifeI_[i];
            const float tau = t - birth_[i];
            const float sN = std::min(tau / lifeI, 1.f);// normalised age
            const float st = strength_[i];

            // ── PUFFS: shared fate makes billow ─────────────────────────────
            // Fully independent parcels smear into a featureless haze — the
            // density volume averages them out. Real rotor wash is LUMPY:
            // turbulent lobes rolling around the ring. Slots share a good part
            // of their draws with a puff (azimuth above all — clustered
            // azimuths are what make discrete lobes on the donut), keeping an
            // individual remainder plus the diffusion spread below so a lobe
            // reads as a fluffy mass, not a clump of coincident points.
            const std::uint32_t puff = i / kPuffSlots;
            const auto draw = [&](std::uint32_t stream, float puffShare) {
                const float uI = rnd01(p_.seed, i, stream);
                const float uP = rnd01(p_.seed ^ 0x9e3779b9u, puff, stream);
                return uI + (uP - uI) * puffShare;
            };
            const auto puffDraw = [&](std::uint32_t stream) {
                return rnd01(p_.seed ^ 0x9e3779b9u, puff, stream);
            };
            const float uA = draw(2u, 0.78f);// flight azimuth: strongly clustered
            const float u6 = draw(6u, 0.66f);// radial reach
            const float u7 = draw(7u, 0.62f);// ceiling
            const float u8 = draw(8u, 0.70f);// tangential drift

            // ── CHAOS, not waves ────────────────────────────────────────────
            // The first build gave every parcel the SAME wobble frequencies
            // (only phases differed) and the cloud swayed like one body of
            // water — "wavelike". Turbulence has no shared beat, so:
            //   • every LOBE draws its own frequencies (f1/f2),
            //   • phases decorrelate SPATIALLY (they depend on the parcel's
            //     own birth anchor, so neighbouring lobes never sync up),
            //   • each lobe ORBITS a private eddy (signed rate — half roll one
            //     way, half the other) instead of swaying side to side,
            //   • and the radial front advances in per-lobe PULSES rather
            //     than one smooth exponential for everyone.
            const float f1 = 1.1f + 2.4f * puffDraw(20u);
            const float f2 = 1.9f + 3.1f * puffDraw(21u);
            const float eddyW = (puffDraw(22u) < 0.5f ? -1.f : 1.f) *
                                (0.7f + 1.5f * puffDraw(23u));
            const float eddyR0 = 0.30f + 0.85f * puffDraw(24u);
            const float surgeF = 0.45f + 0.85f * puffDraw(25u);
            const float surgeP = puffDraw(26u) * kTau;
            const float ph1 = rnd01(p_.seed, i, 4u) * kTau + 1.7f * ax_[i] + 2.3f * az_[i];
            const float ph2 = rnd01(p_.seed, i, 5u) * kTau + 2.9f * az_[i] - 1.3f * ax_[i];
            const float eddyPh = rnd01(p_.seed, i, 13u) * kTau;

            // Wall jet: decelerating radial reach, advancing in pulses.
            const float reach = p_.maxRadius * std::pow(st, 0.7f) * (0.62f + 0.48f * u6);
            const float front = 1.f - std::exp(-tau / p_.tauRadial);
            float radial = reach * front * (1.f + 0.10f * std::sin(surgeF * t + surgeP) * front);
            // The rolled-up front curls back INWARD over the top of the ring.
            radial -= 0.16f * reach * math::smoothstep(0.55f, 1.f, front) *
                      math::smoothstep(0.35f, 0.9f, sN);

            // Roll-up: parcels lift as they reach the front and as they age.
            // Each gets its own ceiling, skewed low (FireEffect note 2 — one
            // shared ceiling renders a hard-edged slab; same bug, dust ring).
            const float ceilI = 0.40f + 0.95f * std::pow(u7, 1.55f);
            const float y = gy_[i] + 0.10f +
                            p_.ringHeight * std::min(st * 1.25f, 1.f) * ceilI *
                                    std::pow(front, 1.9f) *
                                    (0.30f + 0.70f * std::min(sN * 1.6f, 1.f));

            // Slow spiral + the eddy orbit + per-lobe-frequency wobble.
            const float th = uA * kTau + (u8 - 0.5f) * 0.5f * tau;
            const float amp = (0.26f + 0.70f * front) * (0.5f + 0.5f * sN) *
                              std::min(1.f, st * 1.3f);
            const float re = eddyR0 * (0.2f + 1.0f * std::min(sN * 1.5f, 1.f)) *
                             std::min(1.f, st * 1.3f);
            const float ex = re * std::cos(eddyW * tau + eddyPh);
            const float ey = 0.45f * re * std::sin(1.31f * eddyW * tau + eddyPh + 1.1f);
            const float ez = re * std::sin(eddyW * tau + eddyPh);
            const float wx = amp * (0.6f * std::sin(f1 * tau + ph1) +
                                    0.4f * std::sin(1.13f * f2 * tau + ph2));
            const float wy = 0.5f * amp * (0.55f * std::sin(f2 * tau + ph2) +
                                           0.45f * std::sin(0.71f * f1 * tau + ph1));
            const float wz = amp * (0.6f * std::cos(0.87f * f1 * tau + ph1 + 2.1f) +
                                    0.4f * std::sin(f2 * tau + ph2 + 1.3f));

            // Diffusion: each parcel owns a fixed offset direction that GROWS
            // over its life — a fresh lobe is tight, an old one has fluffed
            // out. Kept SMALL: at 2·(0.20 + 1.15·sN) the spread diluted the
            // cloud into thin haze and the ring lost its self-shadowed body.
            const float d1 = rnd01(p_.seed, i, 10u) - 0.5f;
            const float d2 = rnd01(p_.seed, i, 11u) - 0.5f;
            const float d3 = rnd01(p_.seed, i, 12u) - 0.5f;
            const float spread = 2.f * (0.10f + 0.30f * sN);

            // Wind: a parcel picks the ambient wind up only once it has left
            // the jet (the jet's own momentum dominates near the source).
            constexpr float tauW = 0.9f;
            const float wAdv = (tau - tauW * (1.f - std::exp(-tau / tauW))) *
                               (0.35f + 0.65f * front);

            host_[i] = {ax_[i] + std::cos(th) * radial + wx + ex + d1 * spread + wind_.x * wAdv,
                        std::max(y + wy + ey + d2 * spread * 0.55f, gy_[i] + 0.05f),
                        az_[i] + std::sin(th) * radial + wz + ez + d3 * spread + wind_.z * wAdv,
                        1.f};
        }
    });

    field_->submit(host_.data(), n);// dead slots carry the w<0 sentinel
}
