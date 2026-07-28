// Spline authoring, stored on the object itself.
//
// A spline is a Group whose DIRECT CHILDREN are its control points, in child
// order. Nothing else marks them: any untagged Object3D parented to a spline is
// a point, and its local position is that point's position in the spline's
// space. That choice is what makes the feature cheap — serialization,
// undo/redo, the transform gizmo, hierarchy display, renaming and deletion are
// the ones the editor already has, applied to ordinary scene nodes.
//
// The one exception is the GENERATED mesh: a spline whose `mesh` is not None
// carries a real Mesh child tagged `userData["splineDerived"] = "1"`. It is a
// document node like any other — saved, picked, materialled, physics-configured
// — so a scene renders and collides without the editor. Tagged children are not
// control points, which is why every count and index here goes through
// controlPoints()/pointIndexOf() rather than `children` directly.
//
// What makes a Group a spline is the presence of `object.userData["spline"]`,
// which also carries the curve's parameters. Mirrors PhysicsConfig and
// AnimationConfig: one flat `key=value;key=value` string, since userData
// round-trips scalars only (see ObjectExporter::writeUserData). Unknown keys are
// ignored on read, which keeps the format extensible.
//
// Unlike PhysicsConfig there is no "enabled" flag and write() never erases: the
// entry IS the spline, so removing it would stop the object being one.

#ifndef THREEPP_EDITOR_SPLINECONFIG_HPP
#define THREEPP_EDITOR_SPLINECONFIG_HPP

#include "threepp/math/Vector3.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class CatmullRomCurve3;
    class Object3D;

}// namespace threepp

namespace threepp::editor {

    struct SplineConfig {

        // three.js parameterisations, same names and same meaning.
        enum class Type {
            Centripetal,// default; avoids cusps and self-intersection
            Chordal,
            CatmullRom// the only one `tension` affects
        };

        // What the spline generates as real geometry, if anything.
        enum class MeshKind {
            None,// default; the spline is a path and nothing else
            Tube,// pipe / rail / cable, round cross-section
            Road // flat surface of constant width, level side to side
        };

        Type type = Type::Centripetal;
        bool closed = false;
        float tension = 0.5f;
        // Curve samples per segment. Drives the editor's overlay AND the
        // generated mesh — a tube's tessellation, and what a road's straights
        // and arcs are fitted from; nothing at runtime is obliged to use it.
        int samples = 24;

        MeshKind mesh = MeshKind::None;
        float radius = 0.25f;   // Tube
        int radialSegments = 8; // Tube
        float width = 4.f;      // Road
        float uvLength = 4.f;   // metres of curve per U tile, both kinds

        static constexpr const char* userDataKey = "spline";
        // Marks the generated mesh. Its presence is the whole tag; the value is
        // "1" because userData round-trips scalars only.
        static constexpr const char* derivedKey = "splineDerived";
        // Beyond this a segment costs more than it shows.
        static constexpr int maxSamples = 200;

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<SplineConfig> decode(const std::string& text);

        // nullopt when the object is not a spline.
        [[nodiscard]] static std::optional<SplineConfig> read(const Object3D& object);

        // Always writes the entry — see the header note.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        // Carrying the entry is the whole definition; the type is irrelevant.
        [[nodiscard]] static bool isSpline(const Object3D& object);
        // The spline `object` is a control point of, or nullptr. Only DIRECT
        // children qualify, so a mesh nested under a point is not one — and
        // neither is the generated mesh, which is tagged.
        [[nodiscard]] static Object3D* splineOf(const Object3D& object);

        // The generated-mesh tag. Answers "is this child derived state rather
        // than something the user authored".
        [[nodiscard]] static bool isDerived(const Object3D& object);
        static void markDerived(Object3D& object);
        // The spline's generated mesh, or nullptr. First tagged child wins; the
        // editor keeps there being exactly one.
        [[nodiscard]] static Object3D* derivedMesh(const Object3D& spline);

        // Direct children that are control points, in order — every child but
        // the generated mesh.
        [[nodiscard]] static std::vector<Object3D*> controlPointNodes(const Object3D& spline);
        // Their local positions, same order.
        [[nodiscard]] static std::vector<Vector3> controlPoints(const Object3D& spline);

        // Position of `child` among the control points, or the point count when
        // it is not one. NEVER interchangeable with childIndex(): the generated
        // mesh sits among the same children and is not a point.
        [[nodiscard]] static std::size_t pointIndexOf(const Object3D& spline, const Object3D& child);
        // The child index a new point must be inserted at to land at point
        // index `pointIndex`. Past the last point that is just after it, which
        // keeps the generated mesh last.
        [[nodiscard]] static std::size_t childSlotForPointIndex(const Object3D& spline,
                                                                std::size_t pointIndex);

        // The curve those points describe, in the spline's local space. nullptr
        // below two points, where there is no segment to sample.
        [[nodiscard]] std::shared_ptr<CatmullRomCurve3> curve(const Object3D& spline) const;

        // Curve divisions the samples-per-segment setting works out to for
        // `spline`. One formula, so the overlay and the generated mesh are
        // tessellated alike.
        [[nodiscard]] unsigned int divisions(const Object3D& spline) const;

        static const char* label(Type type);
        static const char* label(MeshKind kind);

        bool operator==(const SplineConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SPLINECONFIG_HPP
