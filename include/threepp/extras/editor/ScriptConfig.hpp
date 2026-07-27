// Per-object Python script authoring, stored on the object itself.
//
// A script is one class with optional start(obj)/update(dt)/stop() methods — the
// MonoBehaviour shape. The editor instantiates it on Play and drops it on Stop.
// What the document has to remember is only two things: where the code is, and
// the values of that class's exposed parameters.
//
// The code comes in one of two forms, never both:
//
//   * a .py FILE, referenced by path — shared between objects and scenes, and
//     editable in whatever editor the user already has open;
//   * INLINE SOURCE, stored in the document itself — self-contained, travels
//     with the scene, and edited in the editor's own Script Editor window.
//
// Unlike PhysicsConfig and AnimationConfig this does NOT pack everything into
// one key=value string: a Windows path contains the characters that format uses
// as delimiters, and inline source contains newlines and anything else Python
// allows. RobotConfig hit the same wall, and the answer is the same — each gets
// its own plain userData entry:
//
//   userData["script"]        C:/projects/scripts/spinner.py
//   userData["scriptSource"]  class Spinner:\n    speed = 1.5\n ...
//   userData["scriptFields"]  speed=1.5;clockwise=1;label=spin
//
// "script" and "scriptSource" are mutually exclusive: assigning one clears the
// other, and a hand-edited document carrying both is read as inline (the
// document's own copy of the code wins over a path into someone else's disk)
// and self-heals the next time it is written.
//
// Field values are stored as text and typed on the way in, from the class
// attribute they correspond to (int/float/bool/str). That keeps the format
// human-readable and lets a build with no Python at all still round-trip a
// script's parameters instead of silently dropping them.
//
// Both names and values are sanitized of ';' and '=' when written, so a stray
// delimiter cannot corrupt the rest of the list.

#ifndef THREEPP_EDITOR_SCRIPTCONFIG_HPP
#define THREEPP_EDITOR_SCRIPTCONFIG_HPP

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct ScriptConfig {

        // One exposed class attribute. `value` is the authored override, in the
        // same text form the file stores.
        struct Field {
            std::string name;
            std::string value;

            bool operator==(const Field&) const = default;
        };

        // Source file, stored verbatim. Empty when the script is inline.
        std::string path;
        // Inline source, stored verbatim (newlines and all). Empty when the
        // script is a file.
        std::string source;
        // Insertion-ordered, so a saved document is byte-identical when nothing
        // changed and the inspector shows fields in the order the class declares
        // them.
        std::vector<Field> fields;

        static constexpr const char* pathKey = "script";
        static constexpr const char* sourceKey = "scriptSource";
        static constexpr const char* fieldsKey = "scriptFields";

        [[nodiscard]] bool empty() const { return path.empty() && source.empty(); }
        // Which of the two forms this is. Both false for an empty config.
        [[nodiscard]] bool isInline() const { return !source.empty(); }
        [[nodiscard]] bool isFile() const { return source.empty() && !path.empty(); }

        // The invariant, enforced rather than assumed: setting one form clears
        // the other. Assigning the members directly is fine for a fresh config;
        // these are for switching an existing one over.
        void setPath(std::string value);
        void setSource(std::string value);

        // --- value formatting ---------------------------------------------------
        // Locale-independent and trailing-zero trimmed, so an unchanged value
        // encodes byte-identically every time (the document format's contract).
        [[nodiscard]] static std::string toText(float value);
        [[nodiscard]] static std::string toText(int value);
        [[nodiscard]] static std::string toText(bool value);
        // Lenient: a value that does not parse comes back as the fallback rather
        // than throwing, because it arrives from a file the editor did not write.
        [[nodiscard]] static float toFloat(const std::string& text, float fallback = 0.f);
        [[nodiscard]] static int toInt(const std::string& text, int fallback = 0);
        [[nodiscard]] static bool toBool(const std::string& text, bool fallback = false);

        // Strips the format's delimiters from `text`.
        [[nodiscard]] static std::string sanitized(std::string text);

        // --- field access -------------------------------------------------------
        [[nodiscard]] std::optional<std::string> field(const std::string& name) const;
        // Replaces in place when present, appends otherwise — the order the
        // fields were first seen in is the order they stay in.
        void setField(const std::string& name, const std::string& value);
        void eraseField(const std::string& name);
        // Drops every field not in `names`, so a class that lost an attribute
        // does not leave the value behind in the document forever.
        void retainFields(const std::vector<std::string>& names);

        [[nodiscard]] std::string encodeFields() const;
        [[nodiscard]] static std::vector<Field> decodeFields(const std::string& text);

        // nullopt when the object carries no script at all. An object carrying
        // both keys reads as inline, with `path` empty.
        [[nodiscard]] static std::optional<ScriptConfig> read(const Object3D& object);

        // An empty config removes every entry, so clearing a script leaves no
        // trace in the saved file. Inline source wins over a path, and the
        // losing key is removed — a document only ever carries one form.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        bool operator==(const ScriptConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SCRIPTCONFIG_HPP
