// Rotor downwash and ring sources: what a hovering drone's jet does to the
// water, the grass and the dust under it (Vulkan renderer only; the GL
// renderer ignores it).
//
//     auto dw = Downwash::create();
//     auto& jet = dw->addJet({x, rotorY, z}, 90.f /* N */, 0.5f /* m */);
//     renderer.setDownwash(dw);
//     // each frame: jet.center = the rotor plane's centre
//
// A source is either a JET (a rotor's wake: a centre on the rotor plane, a
// thrust and an effective disc radius) or RINGS (concentric rings leaving a
// point on the water at a speed, one after another: a stone, a fish, a
// creature surfacing). Up to kMaxSources of them, in one list.
//
// The jet's strength follows momentum theory: the induced velocity is
// v_i = sqrt(T / (2 rho A)), and the share every consumer scales by is v_i over
// the reference rotor's (an M350: 90 N through a 1 m disc, about 6.8 m/s). It
// falls off with the height of the rotor above the surface it strikes: full
// below 2 m, gone by 9 m. The struck surface comes from the consumer: the
// fragment's own height on the water, the ground under each grass blade, the
// particle's own height (or Source::surfaceY when it is set).
//
// What reads it:
//   * a material with MeshStandardMaterial::ripples > 0 (flat water): the
//     G-buffer normal takes concentric rings travelling outward from each
//     jet (wavelength growing with radius, amplitude rising to the wall
//     jet's ring and dying past it, a slight lobed swell) and from each ring
//     source, and the struck disc is rougher. Never a random chop. The
//     deferred shade's reflections see the perturbed normal.
//   * every GrassMesh: blades lean out from each jet and are pressed down
//     under it, on top of the field's own wind. A field with a jet near it is
//     never frozen by the distance gate.
//   * Renderer-owned ParticleField emitters: the jet blows a hollow under
//     itself, particles pushed outward and lifted at its rim. Host-owned
//     fields (HostRing, Interop) are written by the host, not the engine;
//     displace() applies the same warp to their positions on the CPU.
//
// The profile is a look, not a flow solve (no CFD): its constants are the
// pine forest's stills (pines/world/downwash.py), measured by eye.
#ifndef THREEPP_DOWNWASH_HPP
#define THREEPP_DOWNWASH_HPP

#include "threepp/math/Vector3.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace threepp {

    class Downwash {

    public:
        static constexpr std::size_t kMaxSources = 8;

        // The reference rotor the look was tuned on: an M350 hovering (about
        // 9.2 kg), its four rotors taken as one 1 m disc.
        static constexpr float kReferenceThrust = 90.f;// N
        static constexpr float kReferenceRadius = 0.5f;// m
        static constexpr float kAirDensity = 1.225f;   // kg/m^3

        enum class Kind : std::uint32_t {
            Jet = 1,
            Rings = 2
        };

        struct Source {
            Kind kind = Kind::Jet;
            bool enabled = true;
            // Jet: the centre of the rotor plane, world. Rings: the centre the
            // rings leave from (its y is not used).
            Vector3 center;

            // ── Jet ─────────────────────────────────────────────────────
            float thrust = kReferenceThrust;// N, all the rotors this jet stands for
            // >= 0: the jet's share directly (1 = the reference rotor), and
            // thrust is not used. Negative (the default): from thrust.
            float strength = -1.f;
            float radius = kReferenceRadius;// effective disc radius, m
            int lobes = 4;                  // the swell's lobes on the water (the rotors); 0 = round
            float lobePhase = 0.6f;         // rad
            // The struck surface's height for consumers without one of their
            // own (particles, displace()). NaN: they use the particle's own y.
            float surfaceY = std::numeric_limits<float>::quiet_NaN();

            // ── Rings ───────────────────────────────────────────────────
            float startTime = 0.f;  // renderer time (sim_time) the first ring leaves
            float speed = 0.32f;    // m/s, outward
            float spacing = 0.29f;  // m between successive rings
            int count = 6;          // rings, one after another
            float amplitude = 0.004f;// m, the first ring's height
            float startRadius = 0.1f;// m, where a ring is born
            float wavelength = 0.16f;// m, the crest train inside one ring
        };

        // ── The look's knobs (defaults: the pine forest's stills) ──────────
        float rippleAmplitude = 0.0045f;// m, the jet's rings on the water at share 1
        float ringWavelength = 0.09f;   // m, the jet's shortest ring (at its axis)
        float ringWavelengthGrowth = 0.035f;// m of wavelength per m of radius
        float ringFrequency = 1.6f;     // wavelengths a second the jet's rings travel outward
        float roughnessBoost = 0.16f;   // added to the water's roughness in the struck disc
        float leanMax = 0.9599f;        // rad (55 deg), a blade's lean at the wall jet's peak
        float pressMax = 0.45f;         // share of a blade's height pressed down under the jet
        float flutter = 0.12f;          // the lean's beat in the jet (share)
        float hollowRadius = 2.6f;      // m, the hollow the jet blows in a particle cloud at share 1
        float rimLift = 0.3f;           // the hollow's rim lifted, a share of the hollow's radius

        static std::shared_ptr<Downwash> create() {
            return std::make_shared<Downwash>();
        }

        // A jet at `center` (the rotor plane's centre). The reference stays
        // valid until the source is removed or the list cleared.
        Source& addJet(const Vector3& center, float thrust = kReferenceThrust, float radius = kReferenceRadius);
        // Rings leaving `center` from `startTime` (renderer time).
        Source& addRings(const Vector3& center, float startTime, float speed = 0.32f, float spacing = 0.29f,
                         int count = 6, float amplitude = 0.004f);

        [[nodiscard]] std::size_t size() const { return count_; }
        [[nodiscard]] bool empty() const { return count_ == 0; }
        Source& source(std::size_t i);
        [[nodiscard]] const Source& source(std::size_t i) const;
        // Removes source i; the ones after it move down one place.
        void remove(std::size_t i);
        void clear() { count_ = 0; }

        // ── The profile, shared with the shaders (shaders/downwash.glsl) ────
        // v_i = sqrt(T / (2 rho pi R^2)), m/s.
        static float inducedVelocity(float thrust, float radius, float airDensity = kAirDensity);
        // The jet's share before the height fade: strength when set, else
        // v_i / v_i(reference). Clamped to [0, 3]. 0 for a ring source.
        [[nodiscard]] static float jetShare(const Source& s);
        // The height fade: 1 below 2 m of rotor height over the struck
        // surface, 0 from 9 m, smooth between.
        static float heightFade(float h);
        // The wall jet's peak radius and its e-fold past the peak, m, for a
        // disc of `radius` at rotor height h.
        static float peakRadius(float radius, float h);
        static float decayRadius(float radius, float h);

        // The particle warp (the hollow) for one world point, as the
        // Renderer-owned emitters apply it. Returns the displaced point.
        // `jitter` in [0, 1] is the particle's own share of the hollow's
        // radius (0.45 + 0.55 jitter of it): a stable per-particle value, so
        // the rim thins out instead of cutting a hard circle.
        [[nodiscard]] Vector3 warp(const Vector3& p, float jitter = 1.f) const;
        // The same warp over host-owned positions: n points, `stride` floats
        // apart (3 for xyz, 4 for xyzw), x y z first, in place. World space;
        // a field with a transform passes its world positions. Each point's
        // jitter is a hash of its index, so a field must keep a particle in
        // the same row from frame to frame.
        void displace(float* xyz, std::size_t n, std::size_t stride = 4) const;

        // Bumped by every add/remove/clear (not by edits through a reference).
        [[nodiscard]] std::uint64_t version() const { return version_; }

    private:
        std::array<Source, kMaxSources> sources_{};
        std::size_t count_ = 0;
        std::uint64_t version_ = 0;
    };

}// namespace threepp

#endif//THREEPP_DOWNWASH_HPP
