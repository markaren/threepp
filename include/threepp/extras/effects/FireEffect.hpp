// FireEffect — a campfire built out of the pieces the renderer already has.
//
// plans/particle-atmosphere.md F-B. Nothing here is a new renderer feature:
// the flame is a ParticleField whose DensityRepr carries F0's blackbody
// emission ramp, the smoke is a SECOND field with its own albedo (the reason
// F0 made the medium params per field), the embers are the legacy
// ParticleSystem billboard path untouched, and the light the fire casts on the
// world is one ordinary PointLight. That last piece is the important one: a
// volume that emits does not, by itself, light anything. The PointLight is
// what puts the fire into the cluster list, the froxel glow, the deferred
// surface shading and the ray-traced shadows — every one of them existing
// machinery, with zero renderer change in this file's name.
//
// ── VULKAN ONLY (inherited) ─────────────────────────────────────────────────
// ParticleField has no GL implementation, so under `--api gl` the flame, the
// smoke and (since F3) the EMBERS all render nothing — the whole effect is now
// three fields plus a light. The PointLight still works, so a GL scene gets a
// flickering fire-coloured light and no fire. Use the Vulkan backend.
// Params::legacyEmbers is the one piece that still draws on GL.
//
// ── THE EMITTER IS STATELESS AND SEEDED ─────────────────────────────────────
// Every slot's position is a CLOSED FORM f(seed_i, t): a hashed birth phase, a
// hashed lifetime, a buoyant rise and a sum of phase-shifted sines for the
// swirl, with the w < 0 dead-slot sentinel covering the part of each slot's
// period when it is not alive. There is no integration, no per-frame state and
// no RNG object anywhere in it, which buys three things:
//
//   • determinism — the same seed and the same t give the same bytes, in this
//     process and in the next one, which is the contract the whole particle
//     subsystem is held to;
//   • free resume and free seeking — update(t) at any t is valid with no
//     warm-up, so a headless capture can jump straight to t = 8 s;
//   • it is the SAME model the device emitter (plan F2) will run in a compute
//     shader, so moving this fire onto the GPU later is a port, not a redesign.
//
// The per-frame CPU cost is ~18k closed-form evaluations plus one memcpy per
// field — microseconds, and independent of how many cameras look at the fire.
//
// ── EMBERS: THE EXCEPTION THAT IS NO LONGER ONE (F3) ────────────────────────
// The ember sparks used to ride the legacy ParticleSystem, which owns its own
// RNG and integrates per frame — so they were the ONE part of this effect that
// was neither deterministic nor seekable, and every metric capture had to turn
// them off to get a comparable image. They are now a THIRD ParticleField,
// Ownership::Renderer, drawn by the billboard representation: the same stateless
// closed form as the flame, evaluated on the device, with the whole effect
// therefore reproducible with the embers ON. Params::legacyEmbers brings the old
// path back for an A/B, and brings its properties back with it.
//
// ── CHURN ───────────────────────────────────────────────────────────────────
// Both fields are created ONCE, in the constructor, at their final capacity,
// and are PARKED (liveCount 0) until ignite(). Creating or destroying a field
// is a structural scene change — entry re-expansion, a vkDeviceWaitIdle and a
// cleared TAA history — so extinguish() parks rather than removes, and a fire
// that will ever be lit should be constructed before the scene starts running.
// See the churn contract in ParticleField.hpp.
//
// ── SMOKE ONLY: a chimney is a fire you cannot see ──────────────────────────
// Params::smokeOnly builds the plume and NOTHING else — one field, no flame
// volume, no embers, no PointLight. That is not a convenience switch, it is a
// budget one: a density field costs a slot out of kMaxDensityFields whether it
// is parked or not, and a PointLight costs a cluster entry and a shadow ray
// whether or not it is bright. A stove burning behind a wall emits neither, and
// the thing a scene actually wants from this class in that case is the PLUME —
// the three-clock buoyancy/advection/dispersion model and the wind-derived box
// that contains it, both of which are the expensive parts to get right and
// neither of which has anything to do with flames.
//
// Everything that shapes the plume keeps its meaning: `height` and `radius` are
// the SOURCE geometry (the flue's mouth, rather than a flame's envelope), and
// smokeHeight / smokeSpread / smokeLifeScale / smokeTaper / smokeSigma are the
// same knobs they always were. flameField(), emberField() and light() return
// null, and ignite()/extinguish() park and un-park the one field there is.
//
// Usage:
//     auto fire = FireEffect::create();          // parked
//     fire->position.set(0.f, 0.f, 3.f);
//     scene.add(fire);
//     fire->ignite();
//     ... each frame:  fire->update(elapsedSeconds);

#ifndef THREEPP_FIREEFFECT_HPP
#define THREEPP_FIREEFFECT_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/ParticleField.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace threepp {

    class ParticleSystem;
    class Texture;

    class FireEffect: public Object3D {

    public:
        struct Params {
            // ── SMOKE ONLY ──────────────────────────────────────────────────
            // Build the plume and nothing else: no flame field, no ember field,
            // no PointLight. See the note at the top of this file — the reason
            // this is a construction-time flag and not "ignite the smoke and
            // park the rest" is that a parked density field still holds one of
            // the four volume slots and a zero-intensity PointLight is still a
            // light in the cluster list. Not created is the only free.
            //
            // A chimney, a smouldering vent, a steam pipe. `height`/`radius`
            // become the SOURCE geometry (the flue mouth) instead of a flame
            // envelope; nothing else changes meaning.
            bool smokeOnly = false;

            // ── Flame envelope (metres, in the effect's own space) ──────────
            float height = 1.20f;// base of the fire to the flame tips
            float radius = 0.25f;// flame radius at the base

            // Capacities. Fixed for the life of the fields (churn contract).
            // 6k/12k is a campfire: the density volume is a smooth medium well
            // before the particle count becomes visible as grain, because each
            // particle splats over 8 voxels.
            std::uint32_t flameParticles = 6'000;
            std::uint32_t smokeParticles = 12'000;
            // Voxels per axis for each volume. Latched when the field first
            // enables its DensityRepr and never changed. Deliberately COARSE:
            // resolution beyond ~8 taps per occupied voxel buys nothing but
            // speckle, since the particles are the only thing filling it.
            std::uint32_t flameResolution = 32;
            std::uint32_t smokeResolution = 32;

            // ── The medium ──────────────────────────────────────────────────
            // sigma_t contributed by ONE particle. R-3 of the plan: the trap
            // here is authoring, not range — too much sigma makes a SOOT-BLACK
            // flame, because extinction beats emission. These are tuned values,
            // not 1.0.
            //
            // The number to reason with is TOTAL OPTICAL MASS, N * sigma: the
            // trilinear splat conserves it, so the mean sigma_t inside the body
            // is N * sigma / (occupied voxels) and the marched depth is that
            // times the chord. A field with a TENTH the particle count needs
            // TEN TIMES the sigma to be the same medium — which is why these
            // read high next to a 10^5-particle dust cloud's 0.5.
            float flameSigma = 2.20f;
            float smokeSigma = 0.34f;
            // Flames are mostly soot: what little they scatter, they scatter
            // dark. Smoke is grey and forward-scattering.
            Color flameAlbedo{0.32f, 0.27f, 0.24f};
            Color smokeAlbedo{0.25f, 0.25f, 0.26f};
            float smokeAnisotropy = 0.30f;

            // ── Emission (F0's blackbody ramp) ──────────────────────────────
            float emissiveIntensity = 34.f;  // radiance of a 2000 K flame; ACES wants tens
            float tempBottomK       = 1950.f;// core
            float tempTopK          = 1080.f;// where the flame gives out to smoke
            float tempFalloff       = 1.55f; // holds the core temperature, drops late

            // ── Smoke column ────────────────────────────────────────────────
            float smokeHeight = 2.20f;// above the flame tips
            float smokeSpread = 0.60f;// how far the column widens by the top

            // ── THE WORLD'S WIND, not the fire's ────────────────────────────
            // Horizontal air velocity in m/s, WORLD space. Every scene that has
            // wind has exactly one, and this is where a fire is told about it:
            // it leans the flame, advects the smoke plume, carries the embers,
            // and (through smokeLifeScale below) sets how far downwind the
            // plume reaches before it disperses.
            //
            // The defect this exists to fix: the fjord's campfire smoke read as
            // "locked in place", rising inside a small authored box and
            // stopping at an invisible edge, while the chimney plume and the
            // trees in the same frame streamed with the scene's wind. The
            // effect had a `wind` already — but it was a private authored
            // constant that no caller ever set from the world's own wind, so
            // there was nothing making the two agree. Pass the SAME vector the
            // rest of the scene uses; setWind() exists so it can change.
            //
            // Default is the value F1 shipped, so a caller that says nothing
            // gets exactly the campfire vulkan_fire has always drawn.
            Vector3 wind{0.22f, 0.f, 0.09f};

            // How long a smoke parcel is followed, as a MULTIPLE of the
            // effect's own 3.2-5.6 s per-parcel period. 1 is the pre-fix column
            // and is bit-identical to it; the plume's downwind reach is
            // |wind| × period × this, so a 0.63 m/s wind at 4 gives roughly
            // 8-14 m of streaming plume instead of 2-3.
            //
            // This is the knob a SCENE sets, not a physical constant: how far a
            // plume stays coherent before it is mixed away is a property of the
            // air, and a demo that wants to see its smoke travel says so.
            float smokeLifeScale = 1.f;
            // ── The DOWNWIND end ────────────────────────────────────────────
            // Fraction of the life the shortest-lived parcels give up, so a
            // WIND-BLOWN plume's far end frays instead of stopping on a plane.
            // Each parcel draws its own ceiling and dies there, so the live
            // population thins continuously over the last stretch.
            //
            // STILL OPT-IN, and that is now a measured choice rather than a
            // byte-stability one. A taper kills slots — mean ceiling is
            // 1 − 0.6·taper, so 0.55 leaves a THIRD of the capacity dead at any
            // instant — and it removes them entirely from the second half of
            // the trajectory, which is where a plume's visible body is. Turned
            // on by default at 0.55 it took vulkan_fire's mid-column from
            // clearly absorbing against the night sky to very nearly invisible
            // (a row-wise excess-luminance profile over the column moved from
            // about −2.5 to about +1.5), i.e. it fixed the top by deleting the
            // plume. The VERTICAL end is fixed by smokeHeightJitter below,
            // which costs no slots at all; this knob is for the horizontal one,
            // and a scene with real wind should set it.
            float smokeTaper = 0.f;

            // ── The VERTICAL end: every parcel needs its own ceiling ────────
            // Spread of the per-parcel height scale, mean-preserving. This is
            // F1 note 2 — the per-slot ceiling that removed the FLAME's flat
            // red cap — finally applied to the smoke field, and it closes a
            // defect that had been in every plume this class ever drew.
            //
            // THE DEFECT: the rise is `smokeHeight · s^0.62`, normalised by the
            // parcel's own age, so it is INDEPENDENT of that parcel's period.
            // The period jitter staggers how far downwind a parcel gets and
            // stagger's nothing about where it ends up vertically — every one
            // of the 12 000 trajectories topped out at exactly the same height.
            // Against a night sky in vulkan_fire that reads as what it is: a
            // column with a flat, hard-edged top and straight cone sides, smoke
            // that does not disperse but STOPS.
            //
            // With a spread the ensemble has no single ceiling: the parcels
            // that fall short spend their late life drifting at a lower
            // altitude (which thickens the plume downwind, as a real one does)
            // and the few that overshoot become the wisps above the body. The
            // draw is skewed LOW — most parcels fall short, a few carry on — so
            // the density decays smoothly toward the top rather than ending.
            //
            // MEAN-PRESERVING BY CONSTRUCTION: the scale is 1 + j·(g − 1) with
            // E[g] = 1, so raising this changes the plume's RAGGEDNESS and not
            // its average height, and no sigma re-tune rides along with it. At
            // 1.0 the scale spans [0.60, 1.60]; recomputeSmokeBox() grows the
            // volume's top to cover that reach, because a box that does not
            // contain the tall wisps clips them on a plane and reintroduces the
            // very defect this fixes (F4 defect 3.2).
            //
            // 0 restores the pre-2026-08-11 single-ceiling column exactly.
            float smokeHeightJitter = 1.f;

            // ── The light the fire casts ────────────────────────────────────
            // This is what makes fire light the WORLD. Intensity is modulated
            // by the seeded flicker below; the colour is the blackbody hue at
            // lightTempK, itself wobbled a little so the flicker shifts warmth
            // and not just brightness (a pure brightness flicker reads as a
            // failing bulb, not as fire).
            float lightIntensity = 26.f;
            float lightRange     = 12.f;
            float lightTempK     = 1850.f;
            float lightHeight    = 0.30f;// above the fire's base
            // Physical source size -> ray-traced soft shadows with a widening
            // penumbra, which is most of why firelight reads as firelight.
            float lightRadius = 0.10f;
            // 0 = a steady lamp, 1 = the full seeded flicker. The flicker is a
            // sum of INCOMMENSURATE sines: it never repeats, it needs no RNG
            // state, and it is a pure function of t (so it survives a seek).
            float flicker = 1.0f;

            // ── Embers (a third ParticleField, device-emitted, additive) ────
            // Sparks rising off the fire, drawn as vertex-less camera-facing
            // quads by the billboard representation (F3). This was the ONE
            // non-deterministic and non-seekable part of the effect while it
            // rode the legacy ParticleSystem — that path owns an RNG and
            // integrates per frame, so a capture of frame 600 depended on
            // frames 1..599 and two runs never matched. As a Renderer-owned
            // field it is the same closed form as the flame: seekable, exactly
            // reproducible, and free of per-frame CPU work.
            bool embers = true;
            // Capacity, fixed for life. ~120 alive at a time at the default
            // duty — a campfire throws sparks, not a firework. Tuned DOWN by
            // eye: the first build shipped 260 and the plume read as a
            // sparkler, which is a different object.
            std::uint32_t emberParticles = 150;
            float emberLife = 2.2f;// seconds, before the per-slot jitter
            // WORLD radius, and this is the number the look lives or dies on:
            // an ember is a centimetre-scale spark seen from a metre or two, so
            // anything over ~0.02 paints glowing pills across the flame. Half
            // the value is per-particle jitter, which is what stops the field
            // reading as N identical dots.
            float emberSize       = 0.011f;
            float emberSizeJitter = 0.60f;
            float emberRise       = 1.45f;// m/s off the fuel bed
            float emberSpread     = 0.50f;// isotropic velocity spread, m/s
            float emberIntensity  = 3.4f; // HDR radiance scale on the sprite
            // Blackbody temperatures the spark's colour ramps BETWEEN over its
            // life. Hot end sits under the flame's own tempBottomK — a spark
            // that has left the flame is already cooling — and the cool end is
            // the deep red just before it goes out.
            float emberHotK  = 2100.f;
            float emberCoolK = 1150.f;
            // Seconds of travel the quad is smeared over, along the spark's own
            // screen-projected velocity. SMALL on purpose: an ember drifts, and
            // this is the knob that decides whether it reads as a spark or as a
            // dash — 0.020 gave a field of 4:1 tick marks all leaning the same
            // way. The velocity itself is free: the emit dispatch already wrote
            // f(t) and f(t - dt).
            float emberStretch = 0.012f;
            // ── F4: the spark's own bloom ───────────────────────────────────
            // Field billboards composite AFTER the scene's bloom pyramid (which
            // is the price of compositing after the upscaler, and that is a
            // price worth paying — see BillboardGlowPass.hpp), so a spark used
            // to glow only as far as its own 3-px falloff reached. > 0 renders
            // the ember field a second time into a small linear-HDR target and
            // runs the shared bloom_down/up chain on that alone, then adds the
            // result in the same overlay slot the quads land in. It scales the
            // radiance written to that target, so the halo can be authored
            // independently of how bright the spark itself is.
            //
            // 0 disables the whole chain — no target, no pipeline, no pass.
            //
            // Tuned by eye and it reads high for a reason worth writing down: a
            // spark is a 3-px source, the pyramid spreads its energy over a
            // ~40-px halo, and the composite divides by the level count to stay
            // resolution independent — so about two orders of magnitude of area
            // and five of level normalisation sit between this number and the
            // pixels. 1.15 (the sprite's own radiance) produced a peak delta of
            // 28/255 against the night sky: the halo was measurable and not
            // visible.
            float emberGlow          = 8.0f;
            // Bright-pass knee for the ember pyramid. 0 = none, which is what a
            // target containing nothing but sparks wants (see
            // BillboardRepr::glowThreshold — thresholding it suppresses the
            // subject).
            float emberGlowThreshold = 0.f;

            // Optional sprite texture. Null is the SHIPPED look: the billboard
            // fragment shader draws a procedural soft spark and the effect
            // pulls in no data folder at all. A texture MODULATES that shape.
            std::shared_ptr<Texture> emberTexture;

            // ── Escape hatch: the pre-F3 legacy ParticleSystem embers ────────
            // Kept so the migration can be A/B'd in ONE binary at ONE seed, and
            // for a caller who wants the old look. It brings back the old
            // properties with it: its own RNG, per-frame integration, no
            // seeking, and a scene that is not bit-reproducible. Requires
            // emberTexture (that path has no procedural sprite).
            bool legacyEmbers = false;
            int  emberRate    = 26;// legacy path only: particles per second

            std::uint32_t seed = 20260811u;
        };

        // Creates both fields at full capacity and PARKED. Nothing burns until
        // ignite(); nothing structural happens after this call.
        //
        // Two overloads rather than `const Params& = {}`, and that is a
        // COMPILER constraint, not a style choice: GCC parses a nested class's
        // default member initializers only once the OUTERMOST class is
        // complete, so at this line — still inside FireEffect — Params has no
        // formable `{}` yet and GCC rejects the default argument outright
        // ("could not convert '{}' to 'const Params&'"). MSVC and Clang accept
        // it, which is why it survived to CI. The .cpp is past the class, so
        // the default is spelled there. Keep Params an AGGREGATE while fixing
        // this — an out-of-line `Params()` would also compile, but it would
        // make Params non-aggregate and silently break
        // `create({.height = 2.f})` at every call site.
        static std::shared_ptr<FireEffect> create(const Params& params);
        static std::shared_ptr<FireEffect> create();

        // Advance to ABSOLUTE time t (seconds since whatever origin the caller
        // likes). Not a delta: the emitter is closed-form in t, so calling this
        // with t = 0, then t = 5, then t = 5 again is all well defined and all
        // reproducible. Call once per frame, before render.
        //
        // (The legacy ember path is the exception — it integrates, so it is
        // driven by the difference between consecutive calls, clamped to keep a
        // hitch or a seek from launching every spark at once.)
        void update(float timeSec);

        // ── Re-aim the fire at the world's wind ─────────────────────────────
        // Free to call every frame: it writes a vector, recomputes the smoke
        // volume's world box (three multiplies) and republishes the ember
        // emitter's O(1) parameter block. Nothing structural — no field is
        // created, resized or removed, so the churn contract is untouched.
        //
        // The smoke's DENSITY BOX is recomputed here rather than left authored,
        // and that is the point: the box has to CONTAIN the plume, and where
        // the plume goes is a function of the wind. A box that does not contain
        // it clips the downwind tail out of the volume — the splat drops
        // anything outside — which draws a plume that ends at an invisible
        // plane in mid-air. (The pre-fix box did exactly that, mildly, even at
        // the default wind.)
        void setWind(const Vector3& worldWind);
        [[nodiscard]] const Vector3& wind() const { return p_.wind; }

        // Un-park both fields and turn the light on. Cheap: no allocation, no
        // device idle, no TAA history clear — the fields already exist.
        void ignite();
        // Park both fields (liveCount 0) and turn the light off. The fields
        // stay in the scene at one entry each, which is the whole reason the
        // churn contract says park instead of remove.
        void extinguish();
        [[nodiscard]] bool lit() const { return lit_; }

        [[nodiscard]] const Params& params() const { return p_; }

        // Null under Params::smokeOnly.
        [[nodiscard]] const std::shared_ptr<ParticleField>& flameField() const { return flame_; }
        // The one field that always exists.
        [[nodiscard]] const std::shared_ptr<ParticleField>& smokeField() const { return smoke_; }
        // Null when Params::embers is off, Params::legacyEmbers is on, or
        // Params::smokeOnly is set.
        [[nodiscard]] const std::shared_ptr<ParticleField>& emberField() const { return embers_; }
        // Null under Params::smokeOnly.
        [[nodiscard]] const std::shared_ptr<PointLight>&    light() const { return light_; }

        // The HUE of a blackbody at `kelvin`, normalised so the maximum channel
        // is 1 — a light colour, with the brightness left to the light's own
        // intensity. This is the host mirror of blackbodyRGB() in
        // shaders/particle_density.glsl, minus that function's
        // Stefan-Boltzmann (T/2000K)^4 magnitude: same two fitted curves, same
        // coefficients. KEEP THE COEFFICIENTS IN SYNC — the point of sharing
        // them is that the flame body and the light it casts are the same
        // colour at the same temperature.
        [[nodiscard]] static Color blackbodyColor(float kelvin);

        // The flicker envelope at time t, in roughly [0.55, 1.45]. Exposed
        // because a scene often wants something ELSE to flicker in step — a
        // lantern, an emissive material, an audio gain — and re-deriving it
        // would put a second, drifting copy of the sines in the caller.
        [[nodiscard]] float flickerAt(float timeSec) const;

        [[nodiscard]] std::string type() const override { return "FireEffect"; }

        explicit FireEffect(const Params& params);
        ~FireEffect() override;

    private:
        Params p_;
        bool   lit_      = false;
        float  lastTime_ = 0.f;// ONLY for the legacy embers' dt; see update()
        bool   haveTime_ = false;

        std::shared_ptr<ParticleField> flame_;
        std::shared_ptr<ParticleField> smoke_;
        // The ember field (Ownership::Renderer, billboards). Null when the
        // legacy escape hatch is on or embers are off.
        std::shared_ptr<ParticleField> embers_;
        std::shared_ptr<PointLight>    light_;
        // The pre-F3 path, alive only under Params::legacyEmbers.
        std::shared_ptr<ParticleSystem> legacyEmbers_;

        // Staging for the two submits. Sized once, never grown — the memcpy
        // into the field is the only per-particle cost in the design and it
        // should not share a frame with an allocation.
        std::vector<ParticlePos> flameHost_;
        std::vector<ParticlePos> smokeHost_;

        // The volumes' world boxes, recomputed from the effect's world matrix
        // each update (the boxes are WORLD-space by DensityRepr's contract,
        // while the particles this emitter writes are effect-LOCAL and are
        // transformed on the way in by the field's matrixWorld).
        Vector3 flameBoxLocal_, flameHalf_, smokeBoxLocal_, smokeHalf_;

        void emitFlame(float t);
        void emitSmoke(float t);
        // The smoke volume's local box, derived from the wind and the plume
        // length. Called from the constructor and from setWind().
        void recomputeSmokeBox();
        // The ember field's emitter parameters, rebuilt from p_. Called from
        // the constructor and from setWind() — sparks ride the same air the
        // smoke does, so a wind change has to reach both or the two disagree
        // in one frame.
        [[nodiscard]] ParticleField::EmitterParams emberEmitter() const;
    };

}// namespace threepp

#endif// THREEPP_FIREEFFECT_HPP
