// Per-object sound authoring, stored on the object itself.
//
// Mirrors AnimationConfig: one flat `key=value;key=value` string in
// `object.userData["sound"]`, so it travels through
// ObjectExporter/ObjectLoader (and the play snapshot) with no sidecar file.
// Unknown keys are ignored on read, which keeps the format extensible.
//
// The FILE is the exception, and gets a plain userData entry of its own:
//
//   userData["sound"]      positional=1;autoplay=1;loop=1;volume=1;...
//   userData["soundFile"]  C:/audio/rain.wav
//
// Same reason RobotConfig and ScriptConfig split theirs out — a Windows path
// contains the characters the flat format uses as delimiters, and escaping
// them would buy nothing.
//
// The path is stored as the file dialog hands it over (absolute, forward
// slashes), which is what the script and URDF references do. A RELATIVE path
// in a hand-edited document resolves against the document's own directory —
// see resolveFile() — so a scene and its audio can be moved together.

#ifndef THREEPP_EDITOR_SOUNDCONFIG_HPP
#define THREEPP_EDITOR_SOUNDCONFIG_HPP

#include <filesystem>
#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct SoundConfig {

        // Distance-attenuation curve. Mirrors PositionalAudio::DistanceModel
        // one for one; kept as its own enum so this header (and every build
        // without audio) stays free of the audio module.
        enum class DistanceModel {
            None,
            Inverse,
            Linear,
            Exponential
        };

        // Spatialized by default: a sound authored ON a scene node is almost
        // always meant to come FROM it. Off makes it ambience — music, room
        // tone — that plays at a constant level wherever the listener stands.
        bool positional = true;
        // Starts with Play unless switched off, the same bargain
        // AnimationConfig strikes: pressing play should just make things go.
        bool autoplay = true;
        bool loop = true;
        float volume = 1.f;
        // Pitch multiplier, 1 = as recorded.
        float rate = 1.f;

        // --- positional only ---------------------------------------------
        // Full volume within this distance.
        float minDistance = 1.f;
        // Where attenuation stops increasing (only the Linear model uses it as
        // a true cutoff). miniaudio's own default, i.e. "effectively unbounded".
        float maxDistance = 10000.f;
        float rolloff = 1.f;
        DistanceModel model = DistanceModel::Inverse;

        static constexpr const char* userDataKey = "sound";
        // The file, on its own plain key (see the header note).
        static constexpr const char* fileKey = "soundFile";

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<SoundConfig> decode(const std::string& text);

        // nullopt when the object carries neither entry. A node with only a
        // file reads as the defaults above, and one with only the parameter
        // string is a sound waiting for a file — both are authoring states the
        // inspector has to be able to show.
        [[nodiscard]] static std::optional<SoundConfig> read(const Object3D& object);

        // Whether this node is an authored sound at all — the predicate the
        // inspector section, the viewport marker and the play session share.
        [[nodiscard]] static bool isSound(const Object3D& object);

        [[nodiscard]] static std::string file(const Object3D& object);
        static void setFile(Object3D& object, const std::string& path);

        // Writes the parameter entry ALWAYS, defaults included: unlike
        // AnimationConfig (whose object is a sound source by virtue of its
        // clips) the entry is what makes a plain Object3D a sound at all, so
        // erasing it at defaults would delete the node's identity. Same rule
        // SplineConfig and TextConfig follow. The file is NOT touched here — it
        // has its own setter, and the two are edited independently.
        void write(Object3D& object) const;

        // Both entries, so "Remove sound" is one call.
        static void erase(Object3D& object);

        // Where `stored` actually points. Absolute paths are returned as they
        // are; a relative one is taken against `documentDir` (empty = the
        // process's working directory, which is what a bare filename means on
        // the command line too).
        [[nodiscard]] static std::filesystem::path resolveFile(const std::string& stored,
                                                               const std::filesystem::path& documentDir);

        [[nodiscard]] static const char* label(DistanceModel model);

        static constexpr DistanceModel models[] = {
                DistanceModel::None, DistanceModel::Inverse,
                DistanceModel::Linear, DistanceModel::Exponential};

        bool operator==(const SoundConfig&) const = default;
    };


    // The two userData entries as ONE value, so a single property command can
    // restore both. Without it "Remove sound" would be two undo steps with a
    // half-authored node in between (parameters gone, file still there), and
    // the inspector would have to decide which order to undo them in.
    struct SoundAuthoring {

        // nullopt when the object is not an authored sound at all.
        std::optional<SoundConfig> config;
        std::string file;

        [[nodiscard]] static SoundAuthoring read(const Object3D& object);
        void write(Object3D& object) const;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SOUNDCONFIG_HPP
