// Conveyor authoring, stored on the object itself.
//
// A conveyor is a Group whose DIRECT CHILDREN are its path waypoints, in child
// order — exactly the spline model (see SplineConfig): any untagged Object3D
// parented to the conveyor is a waypoint, its local position is the waypoint's
// position in the conveyor's space, and serialization, undo/redo, the gizmo and
// deletion are the editor's ordinary ones. A waypoint may additionally carry
// `userData["conveyorWp"]` (ConveyorWaypointConfig) rounding its corner with
// an exact tangent fillet arc, or choosing the surface of the segment leaving
// it (flat / rollers / cleats).
//
// What makes a Group a conveyor is `userData["conveyor"]`, one flat
// `key=value;…` string like PhysicsConfig, carrying the belt parameters.
//
// The GENERATED content is one Group child tagged `userData["conveyorDerived"]`
// holding the visual meshes — belt ribbon, rollers, cleat bars, separator wall,
// and the support frame (side rails, legs, end drums; first-party procedural
// geometry, no imported assets). Each generated mesh is tagged with its role in
// `userData["conveyorRole"]` (belt/roller/cleat/wall/frame/drum) so the play
// session and any external consumer can find the moving parts. The content is
// REGENERATED WHOLESALE whenever the path or config changes: unlike a spline's
// single tube, the part count varies, so hand edits inside the derived group do
// not survive a regeneration. The group node itself is preserved (same uuid).
//
// The whole conveyor — config, waypoints, derived meshes — is plain scene
// content, so a saved document carries it with no editor present. Belt PHYSICS
// is built from the same description by conveyor::ConveyorPhysics (see
// ConveyorPlaySession, or build it directly against a PhysxWorld in an external
// app: read() + spec() on a loaded scene is the export path).

#ifndef THREEPP_EDITOR_CONVEYORCONFIG_HPP
#define THREEPP_EDITOR_CONVEYORCONFIG_HPP

#include "threepp/extras/conveyor/ConveyorGeometry.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;

}// namespace threepp

namespace threepp::editor {

    // Per-waypoint settings, on the waypoint node itself so they follow the
    // node through reorder, undo and serialization. Absent key = all defaults.
    //
    // cornerRadius rounds the corner AT this waypoint with a tangent fillet
    // (see conveyor::cornerFillet) — the waypoint stays on the path and the
    // arc's centre and tangent points are derived, so a bend cannot kink and
    // the radius shrinks itself to what the neighbouring segments allow.
    struct ConveyorWaypointConfig {

        float cornerRadius = 0.f;// 0 = sharp corner
        conveyor::SegKind segKind = conveyor::SegKind::Flat;

        static constexpr const char* userDataKey = "conveyorWp";

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static ConveyorWaypointConfig decode(const std::string& text);

        // Defaults when the node carries no entry — every waypoint has a
        // readable config.
        [[nodiscard]] static ConveyorWaypointConfig read(const Object3D& object);
        void write(Object3D& object) const;
        static void erase(Object3D& object);

        [[nodiscard]] static const char* label(conveyor::SegKind kind);

        bool operator==(const ConveyorWaypointConfig&) const = default;
    };

    struct ConveyorConfig {

        float width = 0.6f;
        float speed = 0.6f;
        bool reverse = false;
        bool smooth = true;
        bool separator = false;
        float wallHeight = 0.5f;
        float rollerRadius = 0.05f;
        float cleatHeight = 0.15f;
        float cleatSpacing = 0.6f;
        int samples = 12;
        bool frame = true;

        static constexpr const char* userDataKey = "conveyor";
        // Marks the generated group. Presence is the whole tag.
        static constexpr const char* derivedKey = "conveyorDerived";
        // On each generated mesh: belt / roller / cleat / wall / frame / drum.
        static constexpr const char* roleKey = "conveyorRole";
        // Beyond this a segment costs more than it shows.
        static constexpr int maxSamples = 64;

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<ConveyorConfig> decode(const std::string& text);

        // nullopt when the object is not a conveyor.
        [[nodiscard]] static std::optional<ConveyorConfig> read(const Object3D& object);
        void write(Object3D& object) const;
        static void erase(Object3D& object);

        [[nodiscard]] static bool isConveyor(const Object3D& object);
        // The conveyor `object` is a waypoint of, or nullptr. Only DIRECT
        // children qualify, and the derived group is not a waypoint.
        [[nodiscard]] static Object3D* conveyorOf(const Object3D& object);

        [[nodiscard]] static bool isDerived(const Object3D& object);
        static void markDerived(Object3D& object);
        // The conveyor's generated group, or nullptr.
        [[nodiscard]] static Object3D* derivedGroup(const Object3D& conveyor);
        // The role tag of a generated mesh, "" when untagged.
        [[nodiscard]] static std::string roleOf(const Object3D& object);

        // Direct children that are waypoints, in order — every child but the
        // generated group.
        [[nodiscard]] static std::vector<Object3D*> waypointNodes(const Object3D& conveyor);

        // Position of `child` among the waypoints, or the waypoint count when
        // it is not one. See SplineConfig::pointIndexOf — never interchangeable
        // with the child index.
        [[nodiscard]] static std::size_t pointIndexOf(const Object3D& conveyor,
                                                      const Object3D& child);
        // The child index a new waypoint must be inserted at to land at
        // waypoint index `pointIndex`.
        [[nodiscard]] static std::size_t childSlotForPointIndex(const Object3D& conveyor,
                                                                std::size_t pointIndex);

        // The resolved, self-contained description, in the conveyor's LOCAL
        // space: this config's fields plus each waypoint's position and
        // per-waypoint settings. The physics side transforms it to world space
        // through the conveyor's matrixWorld.
        [[nodiscard]] conveyor::ConveyorSpec spec(const Object3D& conveyor) const;

        // Bring the generated group in line with the config + waypoints:
        // creates/adopts the tagged group child and REPLACES its content with
        // freshly generated meshes (see the header note — regeneration is
        // wholesale). Fewer than two waypoints leaves the group empty. Replaced
        // geometries are disposed; materials are not (they may be shared).
        void syncDerived(Object3D& conveyor) const;

        bool operator==(const ConveyorConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_CONVEYORCONFIG_HPP
