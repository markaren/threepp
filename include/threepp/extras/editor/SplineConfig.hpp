// Spline authoring, stored on the object itself.
//
// A spline is a Group whose DIRECT CHILDREN are its control points, in child
// order. Nothing else marks them: any Object3D parented to a spline is a point,
// and its local position is that point's position in the spline's space. That
// choice is what makes the feature cheap — serialization, undo/redo, the
// transform gizmo, hierarchy display, renaming and deletion are the ones the
// editor already has, applied to ordinary scene nodes.
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

        Type type = Type::Centripetal;
        bool closed = false;
        float tension = 0.5f;
        // Curve samples per segment, for the editor's overlay only. Nothing at
        // runtime is obliged to use it.
        int samples = 24;

        static constexpr const char* userDataKey = "spline";

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
        // children qualify, so a mesh nested under a point is not one.
        [[nodiscard]] static Object3D* splineOf(const Object3D& object);

        // Local positions of every direct child, in child order.
        [[nodiscard]] static std::vector<Vector3> controlPoints(const Object3D& spline);

        // The curve those points describe, in the spline's local space. nullptr
        // below two points, where there is no segment to sample.
        [[nodiscard]] std::shared_ptr<CatmullRomCurve3> curve(const Object3D& spline) const;

        static const char* label(Type type);

        bool operator==(const SplineConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SPLINECONFIG_HPP
