// Per-mesh acoustic authoring: "sound has to get through THIS".
//
// One flat `key=value;…` string under userData["acousticSurface"], the
// PhysicsConfig/SplatSurfaceConfig format, so it travels through
// ObjectExporter/ObjectLoader and the play snapshot with no sidecar file.
//
// A mesh nobody flagged leaves NO entry (write() erases at enabled=false), which
// is what keeps a scene without acoustics byte-identical to one authored before
// this config existed — and what lets the play session decide "no acoustics
// here" by finding nothing rather than by inspecting every mesh's values.
//
// The two numbers are the two halves of threepp::AcousticSurface: what gets
// THROUGH the surface (occlusion) and what the surface eats when sound bounces
// off it (reverb). Deliberately not gated on THREEPP_WITH_AUDIO — like
// SoundConfig, the authoring is plain data and a build without an audio device
// must still load, show and save a document that has it.

#ifndef THREEPP_EDITOR_ACOUSTICSURFACECONFIG_HPP
#define THREEPP_EDITOR_ACOUSTICSURFACECONFIG_HPP

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct AcousticSurfaceConfig {

        // Off by default: registering every mesh in a scene would build a BVH
        // per mesh at Play for geometry nobody meant as a wall.
        bool enabled = false;

        // Fraction of sound energy that passes through. 0 = solid concrete,
        // ~0.6 = curtain or foliage, 1 = acoustically absent.
        float transmission = 0.f;

        // Fraction absorbed on reflection, the rest bounces. Drives the
        // reverb probe: concrete ~0.05, soft furnishing ~0.6.
        float absorption = 0.3f;

        static constexpr const char* userDataKey = "acousticSurface";

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<AcousticSurfaceConfig> decode(const std::string& text);

        // nullopt when the object carries no acoustic entry.
        [[nodiscard]] static std::optional<AcousticSurfaceConfig> read(const Object3D& object);

        // `enabled == false` removes the entry — see the header note.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        bool operator==(const AcousticSurfaceConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_ACOUSTICSURFACECONFIG_HPP
