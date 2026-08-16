
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
}

void DownwashEffect::setWind(const Vector3& worldWind) {
    wind_ = worldWind;
    // The box must CONTAIN the downwind drift or the splat clips the tail on
    // an invisible plane (FireEffect::recomputeSmokeBox, same defect class).
    // Re-derive it from the stored ANCHOR (source point + ground height).
    if (boxValid_) {
        recomputeBox(boxAnchorLocal_.x, boxAnchorLocal_.z, boxAnchorLocal_.y);
    }
}

void DownwashEffect::recomputeBox(float srcX, float srcZ, float groundLocalY) {
    const float windXZ = std::hypot(wind_.x, wind_.z);
    const float halfX = p_.maxRadius * 1.30f + windXZ * p_.life * 0.7f;
    const float halfY = std::max(3.2f, p_.ringHeight * 1.9f);

    // Snap the anchor to a two-voxel lattice: a per-frame-fitted box re-phases
    // the volume against its own grid and the haze visibly swims (the same
    // error EmitterParams::followSnap exists to prevent).
    const float snap = std::max(0.25f, 4.f * halfX / float(p_.resolution));
    boxAnchorLocal_.set(std::round(srcX / snap) * snap, groundLocalY,
                        std::round(srcZ / snap) * snap);

    // Centre: half a box above the ground (dust lives from the ground up).
    // Nudged downwind so the tail gets the extra room, not the upwind side —
    // applied AFTER the snap: the offset is continuous in the wind, and what
    // must not move per frame is the lattice phase, which is the anchor.
    Vector3 center(boxAnchorLocal_.x, groundLocalY + halfY - 0.6f,
                   boxAnchorLocal_.z);
    if (windXZ > 0.05f) {
        const float shift = std::min(halfX * 0.25f, windXZ * p_.life * 0.35f);
        center.x += wind_.x / windXZ * shift;
        center.z += wind_.z / windXZ * shift;
    }

    Object3D::updateWorldMatrix(true, false);
    Vector3 wp;
    getWorldPosition(wp);
    field_->densityRepr().center = Vector3(wp.x + center.x, wp.y + center.y,
                                           wp.z + center.z);
    field_->densityRepr().halfExtent = Vector3(halfX, halfY, halfX);
    boxValid_ = true;
}

void DownwashEffect::update(float t, const Vector3& dronePosWorld,
                            float thrust01, float aglMetres) {

    Object3D::updateWorldMatrix(true, false);
    Vector3 wp;
    getWorldPosition(wp);
    const Vector3 drone(dronePosWorld.x - wp.x, dronePosWorld.y - wp.y,
                        dronePosWorld.z - wp.z);
    const float groundY = drone.y - aglMetres;

    // ── Entrainment strength: thrust x ground proximity ─────────────────────
    const float prox = 1.f - math::smoothstep(p_.fullAgl, p_.engageAgl,
                                              std::isfinite(aglMetres) ? aglMetres : 1e9f);
    const float strength =
            std::min(1.3f, std::max(0.f, thrust01) * p_.thrustGain) * prox;
    strengthNow_ = strength;

    // ── Box anchoring: recentre only when the source strays ─────────────────
    if (strength > 0.02f) {
        const float dx = drone.x - boxAnchorLocal_.x;
        const float dz = drone.z - boxAnchorLocal_.z;
        if (!boxValid_ || std::hypot(dx, dz) > p_.maxRadius * 0.45f) {
            recomputeBox(drone.x, drone.z, groundY);
        }
    }

    // ── Parcels ─────────────────────────────────────────────────────────────
    // Parallel over chunks: every slot writes only its own state (birth_,
    // strength_, anchors, cycle_, host_[i]), the shared inputs are read-only,
    // and per-chunk alive counts avoid a contended counter — so the serial
    // fallback is bit-identical (utils/Parallel.hpp contract).
    const std::uint32_t n = p_.capacity;
    constexpr std::uint32_t kChunk = 4096;
    const std::uint32_t chunks = (n + kChunk - 1) / kChunk;
    if (chunkAlive_.size() < chunks) chunkAlive_.resize(chunks);
    if (chunkIdx_.size() != chunks) {
        chunkIdx_.resize(chunks);
        for (std::uint32_t c = 0; c < chunks; ++c) chunkIdx_[c] = c;
    }
    parallelForEach(chunkIdx_.begin(), chunkIdx_.end(), [&](std::uint32_t c) {
        std::uint32_t chunkAlive = 0;
        const std::uint32_t iEnd = std::min(n, (c + 1) * kChunk);
        for (std::uint32_t i = c * kChunk; i < iEnd; ++i) {
            const float u3 = rnd01(p_.seed, i, 3u);// life scale
            const float lifeI = p_.life * (1.f + p_.lifeJitter * (2.f * u3 - 1.f));
            float tau = t - birth_[i];

            // Slot finished its flight (or is parked and due a retry — parked
            // slots carry birth = t − lifeI + retry, so the same test serves both):
            // consult the air NOW.
            if (tau >= lifeI) {
                const float accept = rnd01(p_.seed + cycle_[i], i, 9u);
                ++cycle_[i];
                // Per-PUFF duty, re-drawn every ~1.7 s: whole puffs sit a cycle
                // out while their neighbours fill in, so the ring is built of
                // distinct lobes that come and go rather than a uniform disc.
                const std::uint32_t puffId = i / kPuffSlots;
                const auto epoch = static_cast<std::uint32_t>(t / 1.7f);
                const float puffDuty =
                        0.35f + 0.85f * rnd01(p_.seed ^ 0xABCD1234u, puffId, epoch);
                if (strength > 0.02f && accept < strength * 1.25f * puffDuty) {
                    // Entrain: latch the annulus anchor to where the drone IS.
                    const float uR = rnd01(p_.seed + cycle_[i], i, 1u);
                    const float uA = rnd01(p_.seed + cycle_[i], i, 2u);
                    const float r0 = p_.sourceRadius * (0.35f + 0.75f * std::sqrt(uR));
                    const float a0 = uA * kTau;
                    birth_[i] = t;
                    strength_[i] = strength;
                    ax_[i] = drone.x + std::cos(a0) * r0;
                    az_[i] = drone.z + std::sin(a0) * r0;
                    gy_[i] = groundY;
                    tau = 0.f;
                } else {
                    // Parked: retry soon, not next period.
                    birth_[i] = t - lifeI + kRetrySeconds;
                    strength_[i] = 0.f;
                    host_[i] = {0.f, 0.f, 0.f, -1.f};
                    continue;
                }
            }
            if (strength_[i] <= 0.f) {// parked, not yet due
                host_[i] = {0.f, 0.f, 0.f, -1.f};
                continue;
            }

            // ── Closed form between latches ─────────────────────────────────────
            const float sN = std::min(tau / lifeI, 1.f);// normalised age
            const float st = strength_[i];

            // ── PUFFS: shared fate makes billow ─────────────────────────────────
            // Fully independent parcels smear into a featureless haze — the
            // density volume averages them out. Real rotor wash is LUMPY:
            // turbulent lobes rolling around the ring. So slots share a good part
            // of their draws with a puff (azimuth above all — clustered azimuths
            // are what make discrete lobes on the donut), keeping an individual
            // remainder plus the diffusion spread below so a lobe reads as a
            // fluffy mass, not a clump of coincident points. Shares tuned DOWN
            // from the first pass, which read as coarse blobs.
            const std::uint32_t puff = i / kPuffSlots;
            const auto draw = [&](std::uint32_t stream, float puffShare) {
                const float uI = rnd01(p_.seed, i, stream);
                const float uP = rnd01(p_.seed ^ 0x9e3779b9u, puff, stream);
                return uI + (uP - uI) * puffShare;
            };
            const float uA = draw(2u, 0.78f);       // flight azimuth: strongly clustered
            const float u6 = draw(6u, 0.66f);       // radial reach
            const float u7 = draw(7u, 0.62f);       // ceiling
            const float u8 = draw(8u, 0.70f);       // tangential drift
            const float p1 = draw(4u, 0.84f) * kTau;// wobble phases: a lobe rolls
            const float p2 = draw(5u, 0.84f) * kTau;// mostly as one body

            // Wall jet: exponentially decelerating radial reach.
            const float reach = p_.maxRadius * std::pow(st, 0.7f) * (0.62f + 0.48f * u6);
            const float front = 1.f - std::exp(-tau / p_.tauRadial);
            float radial = reach * front;
            // The rolled-up front curls back INWARD over the top of the ring.
            radial -= 0.16f * reach * math::smoothstep(0.55f, 1.f, front) *
                      math::smoothstep(0.35f, 0.9f, sN);

            // Roll-up: parcels lift as they reach the front and as they age. Each
            // gets its own ceiling, skewed low (FireEffect note 2 — one shared
            // ceiling renders a hard-edged slab, and a dust ring has the same bug).
            const float ceilI = 0.40f + 0.95f * std::pow(u7, 1.55f);
            float y = gy_[i] + 0.10f +
                      p_.ringHeight * std::min(st * 1.25f, 1.f) * ceilI *
                              std::pow(front, 1.9f) * (0.30f + 0.70f * std::min(sN * 1.6f, 1.f));

            // Slow spiral around the source + growing sine wobble. With the puff
            // sharing above, the wobble moves whole lobes — billow, not shimmer —
            // so the amplitude is authored at lobe scale.
            const float th = uA * kTau + (u8 - 0.5f) * 0.5f * tau;
            const float amp = (0.30f + 0.85f * front) * (0.5f + 0.5f * sN) *
                              std::min(1.f, st * 1.3f);
            const float wx = amp * (0.6f * std::sin(1.9f * tau + p1) +
                                    0.4f * std::sin(3.1f * tau + p2));
            const float wy = 0.5f * amp * (0.55f * std::sin(2.5f * tau + p2) + 0.45f * std::sin(1.3f * tau + p1));
            const float wz = amp * (0.6f * std::cos(2.2f * tau + p1) +
                                    0.4f * std::sin(2.7f * tau + p2 + 1.3f));

            // Diffusion: each parcel owns a fixed offset direction that GROWS over
            // its life, so a fresh lobe is tight and an old one has fluffed out —
            // turbulent mixing as a closed form, and the thing that keeps a puff
            // from rendering as one dense clump.
            const float d1 = rnd01(p_.seed, i, 10u) - 0.5f;
            const float d2 = rnd01(p_.seed, i, 11u) - 0.5f;
            const float d3 = rnd01(p_.seed, i, 12u) - 0.5f;
            // Kept SMALL: at 2·(0.20 + 1.15·sN) the spread diluted the whole
            // cloud into thin haze and the ring lost its self-shadowed body.
            // Enough to de-clump a puff, no more.
            const float spread = 2.f * (0.10f + 0.30f * sN);

            // Wind: a parcel picks the ambient wind up only once it has left the
            // jet (the jet's own momentum dominates near the source).
            constexpr float tauW = 0.9f;
            const float wAdv = (tau - tauW * (1.f - std::exp(-tau / tauW))) *
                               (0.35f + 0.65f * front);

            host_[i] = {ax_[i] + std::cos(th) * radial + wx + d1 * spread + wind_.x * wAdv,
                        std::max(y + wy + d2 * spread * 0.55f, gy_[i] + 0.05f),
                        az_[i] + std::sin(th) * radial + wz + d3 * spread + wind_.z * wAdv,
                        1.f};
            ++chunkAlive;
        }
        chunkAlive_[c] = chunkAlive;
    });

    std::uint32_t alive = 0;
    for (std::uint32_t c = 0; c < chunks; ++c) alive += chunkAlive_[c];

    if (alive == 0) {
        field_->setLiveCount(0);// parked: skip the scatter entirely
        return;
    }
    field_->submit(host_.data(), n);// dead slots carry the w<0 sentinel
}
