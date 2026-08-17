// Per-object animation authoring, stored on the object itself.
//
// Mirrors PhysicsConfig: one flat `key=value;key=value` string in
// `object.userData["animation"]`, so it travels through
// ObjectExporter/ObjectLoader (and the play snapshot) with no sidecar file.
// Unknown keys are ignored on read, which keeps the format extensible.

#ifndef THREEPP_EDITOR_ANIMATIONCONFIG_HPP
#define THREEPP_EDITOR_ANIMATIONCONFIG_HPP

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct AnimationConfig {

        // Objects that carry clips animate during play unless switched off —
        // pressing play on an imported character should just make it move.
        bool autoplay = true;
        // Clip name; empty selects the object's first clip. Encoded with '='
        // and ';' stripped (they are the format's delimiters).
        std::string clip;
        bool loop = true;
        float speed = 1.f;

        static constexpr const char* userDataKey = "animation";

        [[nodiscard]] bool isDefault() const;

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<AnimationConfig> decode(const std::string& text);

        // nullopt when the object carries no animation entry; callers treat
        // that as the defaults above.
        [[nodiscard]] static std::optional<AnimationConfig> read(const Object3D& object);

        // Writes the entry; a default config removes it, so untouched objects
        // leave no trace in the saved file.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        bool operator==(const AnimationConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_ANIMATIONCONFIG_HPP
