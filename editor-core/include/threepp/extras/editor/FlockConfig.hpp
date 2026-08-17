// Ambient flock authoring, stored on the object itself.
//
// A flock is a Group carrying `userData["flock"]` — the curated subset of
// Flock::Params worth a slider. THE NODE'S POSITION IS THE TERRITORY'S HOME:
// like the particle field's emitter frame, the transform gizmo is the
// authoring tool for where the birds live, so `home` is never stored here.
//
// The birds themselves are never document nodes: FlockPlaySession builds a
// Flock from this entry when Play starts (baking perches against the scene as
// it stands), and the Stop snapshot restores a document that never saw them.
// A saved scene therefore carries a dozen scalars, not 2 256 vertices of bird.
//
// What makes a Group a flock is the presence of the entry — no enabled flag,
// and write() never erases, for TreeConfig's reason: the entry IS the flock.
//
// Storage is the flat `key=value;…` string the Config family shares. Every key
// is the FlockConfig field name; unknown keys are ignored on read so a
// document written by a newer editor still loads.

#ifndef THREEPP_EDITOR_FLOCKCONFIG_HPP
#define THREEPP_EDITOR_FLOCKCONFIG_HPP

#include "threepp/extras/fauna/Flock.hpp"

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}// namespace threepp

namespace threepp::editor {

    struct FlockConfig {

        // ── Population ──────────────────────────────────────────────────────
        int seed = 1337;
        // 18 reads as "a place where birds live"; 200 reads as "a bird
        // simulation" (Flock.hpp's own words). Clamped to [0, 256] downstream.
        int birdCount = 18;
        // Drives wingbeat frequency allometrically, and the bird's size with it.
        float massKg = 0.078f;

        // ── Territory (home = the node's world position) ────────────────────
        float roamRadius = 42.f;
        float cruiseAltitude = 14.f;
        // ± fraction of cruiseAltitude, per bird — the thickness of the loose
        // altitude band the flock flies in. 0 collapses it to a plane, which
        // reads as a formation, not a flock.
        float altitudeSpread = 0.35f;
        float cruiseSpeed = 9.f;

        // ── Perching ────────────────────────────────────────────────────────
        bool perching = true;
        float maxPerchedFraction = 0.55f;

        // ── Look ────────────────────────────────────────────────────────────
        bool castShadow = false;
        float windX = 0.7f;
        float windZ = 0.7f;

        static constexpr const char* userDataKey = "flock";

        [[nodiscard]] std::string encode() const;
        // Never nullopt: an empty or unparsable string decodes to the defaults.
        [[nodiscard]] static std::optional<FlockConfig> decode(const std::string& text);

        // nullopt when the object carries no flock entry.
        [[nodiscard]] static std::optional<FlockConfig> read(const Object3D& object);
        // Always writes the entry — see the header note.
        void write(Object3D& object) const;
        static void erase(Object3D& object);

        [[nodiscard]] static bool isFlock(const Object3D& object);

        // The runtime parameter block: defaults everywhere this config has no
        // opinion, `home` from the authored node's world position. Flock's own
        // sanitise() owns the clamping.
        [[nodiscard]] Flock::Params makeParams(const Vector3& home) const;

        bool operator==(const FlockConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_FLOCKCONFIG_HPP
