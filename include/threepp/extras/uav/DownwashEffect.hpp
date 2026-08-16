// DownwashEffect — rotor-wash dust (brownout), built out of the pieces the
// renderer already has.
//
// Nothing here is a new renderer feature: the cloud is ONE ParticleField whose
// DensityRepr does all the optical work — Beer-Lambert extinction (the drone
// and the world genuinely disappear inside the cloud, and so does a camera
// that descends into it), sun in-scatter with an HG phase (the rim of the
// cloud glows against the light), a centroid shadow ray, and point lights
// glowing through the dust. The per-frame cost is one host memcpy and one
// density scatter, independent of how many cameras look at it. FireEffect is
// the template; this is the same pattern pointed at the ground instead of the
// sky.
//
// ── VULKAN ONLY (inherited) ─────────────────────────────────────────────────
// ParticleField has no GL implementation, so under `--api gl` the effect
// renders nothing and costs (almost) nothing. Gate creation on the backend.
//
// ── THE PHYSICS, one paragraph ──────────────────────────────────────────────
// Rotor downwash impinges on the ground and turns into a radial WALL JET; the
// jet decelerates, its front rolls up into a ground vortex ring, and entrained
// dust forms the expanding donut that eventually engulfs the aircraft — the
// brownout. The model here is that shape as a closed form: parcels are born on
// the impingement annulus under the vehicle, shoot outward with an
// exponentially decelerating radial reach, lift into the rolled-up front as
// they get there, churn on per-lobe eddies and frequencies (no two alike, and
// no shared beat — feedback_procedural_avoid_perfect_repetition), ride the
// world's wind once they leave the jet, and die. Entrainment is gated on
// thrust x ground proximity: no dust in a high hover, a wall of it in the
// last metres.
//
// ── THE DUST IS CONSERVED ───────────────────────────────────────────────────
// The ground carries a finite reservoir of loose soil (a 2D mass grid over
// the landing site). Spawning a parcel ERODES one quantum from the cell under
// its birth point — an exhausted cell spawns nothing — and a dying parcel
// DEPOSITS its quantum wherever the wind carried it. Dust genuinely moves
// from A to B: a hover slowly blows its own pad clean and the brownout fades,
// the downwind band thickens, and ground + airborne totals stay exactly
// equal to the initial seeding (the unit test holds this to the quantum).
//
// ── DETERMINISM: latched, not stateless ─────────────────────────────────────
// FireEffect's emitter is a pure closed form f(seed, t) because a campfire
// does not move. A downwash source DOES — the parcels' anchors depend on where
// the drone was when they were entrained — so each slot LATCHES (birth time,
// anchor, strength) at re-spawn and is closed-form between latches. The effect
// is therefore deterministic given the same sequence of update() calls (a
// scripted flight at a fixed dt reproduces exactly), but it cannot seek: the
// latched state IS the history. That is the honest trade, and it is stated
// here rather than discovered.
//
// ── CHURN ───────────────────────────────────────────────────────────────────
// The field is created ONCE, in the constructor, at final capacity, and PARKED
// (liveCount 0) whenever the air is clean. See ParticleField.hpp's churn
// contract: parking is free, creating mid-flight is a structural rebuild.
//
// Usage:
//     auto dust = uav::DownwashEffect::create();   // parked
//     scene.add(dust);
//     ... each frame, with SIM time (never a wall clock):
//     dust->setWind(worldWind);                    // the scene's ONE wind
//     dust->update(simTimeSec, droneWorldPos, meanThrottle01, aglMetres);

#ifndef THREEPP_EXTRAS_UAV_DOWNWASHEFFECT_HPP
#define THREEPP_EXTRAS_UAV_DOWNWASHEFFECT_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/ParticleField.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace threepp::uav {

    class DownwashEffect: public Object3D {

    public:
        struct Params {
            // Capacity, fixed for life (churn contract). 26k parcels splat a
            // smooth medium at resolution 96 well before the count reads as
            // grain; the CPU cost is ~26k closed-form evaluations + one memcpy.
            // Sized TOGETHER with resolution by the fire plan's speckle rule:
            // the particles are the only thing filling the volume, so the
            // number that matters is splat taps per occupied voxel. The cloud
            // occupies ~160k voxels at resolution 160; 90k parcels x 8-voxel
            // trilinear splats ≈ 4.4 taps/voxel — smooth. 26k at the same
            // resolution was 1.3 taps/voxel and rendered as coarse speckle.
            // The emitter loop is parallel (disjoint per-slot state), so the
            // CPU cost stays well under a millisecond.
            std::uint32_t capacity   = 90'000;
            // Voxels per axis of the density volume. Latched at construction.
            // 160 over the ~21 m box is ~13 cm voxels — the difference between
            // fog and billow: at 96 the trilinear splat smears every lobe into
            // smooth haze. ~25 MB of volume; the march cost is per-pixel steps
            // and does not grow with this.
            std::uint32_t resolution = 160;

            // ── Source geometry (metres) ────────────────────────────────────
            // Outer radius of the impingement annulus parcels are born on —
            // roughly the rotor disc's footprint on the ground.
            float sourceRadius = 0.9f;
            // How far the wall jet carries dust at full strength, and how high
            // the rolled-up front reaches. The cloud's box derives from these.
            // Tuned together with sigma: a tighter, taller ring holds enough
            // density to SELF-SHADOW, which is what gives the cloud a bright
            // sunlit crown over a dark underbelly instead of flat haze.
            float maxRadius  = 7.2f;
            float ringHeight = 3.4f;
            // Wall-jet deceleration time constant: dist = reach·(1 − e^(−τ/tau)).
            float tauRadial = 1.05f;

            // ── Parcel life ─────────────────────────────────────────────────
            // Long enough that the cloud HANGS after a motor cut instead of
            // evaporating within two seconds — fine dust lingers.
            float life       = 4.5f; // seconds airborne, before the jitter
            float lifeJitter = 0.6f; // ± fraction of life, per slot

            // ── Entrainment gate ────────────────────────────────────────────
            // strength = min(1.3, thrust01·thrustGain) · proximity, with
            // proximity ramping from 0 at engageAgl to 1 at fullAgl. Hover
            // throttle on the QuadSim defaults is ~0.41, so gain 2.4 puts a
            // low hover at strength ~1 — a landing IS the full effect.
            float engageAgl  = 6.0f;
            float fullAgl    = 1.4f;
            float thrustGain = 2.4f;

            // ── The medium ──────────────────────────────────────────────────
            // sigma_t of ONE parcel. The number to reason with is total optical
            // mass N·sigma (see FireEffect::Params). Tuned by LOOK: 1.5 read
            // as a solid steam bank — extinction saturated and every billow
            // vanished into one white mass; 0.55 keeps the core thick enough
            // to swallow the vehicle while the edges stay structured.
            // Slightly above mass parity with the 26k x 0.8 build (= 0.24):
            // the smooth field spreads the same mass over more voxels, so the
            // peaks that used to carry the look need the small top-up.
            float sigmaPerParticle = 0.30f;
            // Dry SOIL, not steam: real dirt albedo is dark (~0.2-0.35) and
            // warm, and the sky's pale in-scatter washes anything lighter to
            // white. Strong forward scatter keeps the sunward rim glowing and
            // gives the interior a brighter-toward-the-sun gradient — shape,
            // even when the camera is inside the cloud.
            Color albedo{0.45f, 0.33f, 0.19f};
            float anisotropy = 0.62f;

            // ── The ground reservoir (conservation) ─────────────────────────
            // Loose soil available for entrainment, in parcel QUANTA per m²,
            // held in a 2D grid anchored at the first landing site. Erosion
            // takes a quantum per spawned parcel from the cell under its birth
            // anchor; a dying parcel deposits its quantum where it ends up.
            // Total dust (ground + airborne) is exactly conserved.
            //
            // Sized so the impingement disc (~6 m² at the widened birth
            // annulus) holds roughly ONE big event's worth of turnover. The
            // depth IS the narrative: at 40k/m² the site survived a takeoff
            // AND a landing AND the next takeoff at full strength, so every
            // event read as "spawning new dust" (user report, twice). At 20k
            // a fresh-pad takeoff blasts the big cloud, the landing moments
            // later raises visibly less from what the takeoff left, and the
            // next cycle is thin — cleared means cleared. Raise this for
            // dustier ground, at the cost of the story taking longer to tell.
            float groundDustPerM2 = 20'000.f;
            float gridCell        = 0.8f;// metres per reservoir cell

            std::uint32_t seed = 20260816u;
        };

        // Two overloads rather than `const Params& = {}` — same GCC
        // nested-aggregate constraint FireEffect documents.
        static std::shared_ptr<DownwashEffect> create(const Params& params);
        static std::shared_ptr<DownwashEffect> create();

        // Advance to ABSOLUTE sim time t and feed the vehicle state:
        //   dronePosWorld — vehicle position, world space
        //   thrust01      — mean filtered throttle, [0,1] (QuadSim::motorLevel)
        //   aglMetres     — height of the vehicle above the ground under it;
        //                   pass a large value (or +inf) when out of range.
        // The ground the dust lives on is dronePosWorld.y − aglMetres.
        //
        // Call once per RENDER frame with the sim's own clock. Time must be
        // monotonic; the effect is deterministic under a reproduced call
        // sequence (see the header note) but cannot seek backwards.
        //
        // The effect's own transform: TRANSLATION only. Positions are
        // submitted in effect-local space (world − effect world position);
        // a rotated or scaled DownwashEffect is not supported.
        void update(float timeSec, const Vector3& dronePosWorld,
                    float thrust01, float aglMetres);

        // The scene's STEADY wind (world space, m/s). Advects airborne parcels
        // and stretches the density box downwind. Free to call every frame.
        //
        // Pass the BASE wind, not a gust-modulated sample: a globally shared
        // gust signal advects every parcel in sync and the whole cloud surges
        // on the gust sines — "reads as waves", verbatim user report. Gusts
        // belong to setGustiness below, where each LOBE draws its own.
        void setWind(const Vector3& worldWind);
        [[nodiscard]] const Vector3& wind() const { return wind_; }

        // Gust strength [0..1]. Applied PER LOBE — each lobe modulates the
        // steady wind's magnitude and direction on its own drawn frequencies
        // and phases, so lobes surge independently (turbulence) instead of
        // the cloud pumping as one body (a wave).
        void setGustiness(float g) { gust_ = std::clamp(g, 0.f, 1.f); }
        [[nodiscard]] float gustiness() const { return gust_; }

        // Current entrainment strength [0..~1.3] — 0 means clean air. Handy
        // for HUDs and for driving sensor-degradation stories off the same
        // number the visuals use.
        [[nodiscard]] float dustiness() const { return strengthNow_; }

        // ── Reservoir readouts ──────────────────────────────────────────────
        // Quanta on the ground / initially seeded (equal before first
        // entrainment). The conservation invariant, and the thing a test can
        // hold exactly: groundDustInitial() == groundDustQuanta() + airborne().
        // DOUBLE, not float: the site totals ~10^8 quanta and float loses ±1
        // arithmetic past 2^24 — the invariant drifted by exactly the lost
        // increments until these were widened. Per-CELL masses stay float
        // (they never leave the 10^5 range, where float counts exactly).
        [[nodiscard]] double groundDustQuanta() const { return groundTotal_; }
        [[nodiscard]] double groundDustInitial() const { return groundInitial_; }
        [[nodiscard]] std::uint32_t airborne() const { return alive_; }
        // Quanta in the reservoir cell under an effect-local ground point.
        [[nodiscard]] float groundDustAt(float localX, float localZ) const;

        [[nodiscard]] const std::shared_ptr<ParticleField>& field() const { return field_; }
        [[nodiscard]] const Params& params() const { return p_; }

        [[nodiscard]] std::string type() const override { return "DownwashEffect"; }

        explicit DownwashEffect(const Params& params);
        ~DownwashEffect() override = default;

    private:
        Params  p_;
        Vector3 wind_{0.f, 0.f, 0.f};
        float   strengthNow_ = 0.f;
        float   gust_ = 0.f;// per-lobe gust strength, see setGustiness
        // Slow-averaged wind magnitude (box sizing only — see setWind).
        float windSlow_ = 0.f;
        float lastT_ = 0.f;
        bool  haveT_ = false;

        std::shared_ptr<ParticleField> field_;
        std::vector<ParticlePos>       host_;// staging, sized once

        // Latched per-slot state (see the determinism note): parcels anchor to
        // where the drone WAS when they were entrained.
        std::vector<float>         birth_;   // latch time; parked slots retry
        std::vector<float>         strength_;// 0 = parked
        std::vector<float>         ax_, az_; // annulus anchor, effect-local xz
        std::vector<float>         gy_;      // ground height at latch, local y
        std::vector<std::uint32_t> cycle_;   // respawn counter → fresh hashes
        std::vector<float>         lifeI_;   // static per-slot life, baked once
        // Parallel position-pass plumbing (the lifecycle pass is SERIAL — it
        // owns the shared reservoir; see update()).
        std::vector<std::uint32_t> chunkIdx_;
        std::uint32_t              alive_ = 0;

        // ── The ground reservoir (see the .cpp note) ────────────────────────
        std::vector<float> grid_;// parcel quanta per cell
        int    gridN_ = 0;
        float  gridAnchorX_ = 0.f, gridAnchorZ_ = 0.f;
        double groundTotal_ = 0., groundInitial_ = 0.;
        bool   gridValid_ = false;

        void initGrid(float srcX, float srcZ);
        [[nodiscard]] std::size_t cellOf(float x, float z) const;

        // Density-box ANCHOR, effect-local: xz = snapped source point, y = the
        // ground height there. The box CENTER is derived from this every
        // recompute — never fed back in (feeding the derived center back as
        // the ground made the box climb by halfY per frame). Recentred only
        // when the source strays: a per-frame-fitted box makes the volume swim
        // against its own lattice (see EmitterParams::followSnap).
        Vector3 boxAnchorLocal_{0.f, 0.f, 0.f};
        bool    boxValid_ = false;

        void recomputeBox(float srcX, float srcZ, float groundLocalY);
    };

}// namespace threepp::uav

#endif// THREEPP_EXTRAS_UAV_DOWNWASHEFFECT_HPP
