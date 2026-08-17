// An inline Python script that AUTHORS scene content. Normally it belongs to the
// SCENE: a generated scene's rule is a property of that scene, and the obvious
// place to look for it is the root. Verified to survive the round trip — the
// root serialises as an object of type "Scene", so its userData (newlines and
// all) comes back from a save or a play/stop like any other node's.
//
// Nothing here is scene-specific though: read/write take any Object3D, so a
// Group can carry its own generator when a scene wants several independently
// re-runnable ones (props, then clutter on top). Same rules, narrower scope. The
// scene root is the default because one rule per scene is the common case, not
// because the others are excluded.
//
// The distinction from ScriptConfig is what the code is for, and it changes
// everything about when it runs. A ScriptConfig script is behaviour: a class
// with update(dt), instantiated on Play and dropped on Stop, never touching the
// document. A generator is authoring: plain top-level statements that build
// objects, run on demand in EDIT mode, and whose output is committed to the
// document as ordinary saveable content.
//
// So a generator has no update() contract and no class to find — it is a module
// body, executed once per Regenerate. What it creates becomes ONE tagged child
// of whatever carries the generator (see generatedKey) — a top-level group when
// that is the scene — replaced wholesale the next time it runs. That mirrors
// userData["splineDerived"]: the derived node is real scene content that
// serializes and can carry physics, and the thing that produces it is the
// authored state.
//
// Because the output is replaced wholesale, the SCRIPT is the source of truth for
// it: a material or physics config hand-edited onto generated content is lost on
// the next run. Set it in the script instead. The spline's tube sync can preserve
// a uuid and a material because it knows the output is one mesh; a generator's
// output is arbitrary, so it cannot.
//
// Storage follows ScriptConfig's reasoning exactly. Source contains newlines and
// arbitrary Python, so it cannot ride in the flat key=value format PhysicsConfig
// and AnimationConfig use; it gets its own plain userData entry:
//
//   userData["generatorSource"]  count = 400\nseed = 7\nfor i in range(count):...
//   userData["generatorFields"]  count=400;seed=7
//
// INLINE ONLY, deliberately. The whole point is that a generated scene is
// self-contained: the rule that built the content travels in the document, so
// opening the file on another machine shows both the result and the reason for
// it. A path into someone else's disk would give up exactly that. ScriptConfig
// supports both forms because behaviour scripts are shared between objects and
// scenes; an authoring rule belongs to the scene it authored.
//
// `fields` are the script's exposed parameters, in the same text-typed form
// ScriptConfig uses (and via the same static helpers, which is why this struct
// does not reimplement the encoding). A generator that declares `count = 400` at
// module level gets an inspector input for it, so re-running with a different
// count needs no code edit — a fixed config schema, derived from the script
// rather than designed up front.
//
// IMPORTANT — a generator is never run by opening a scene. Executing document
// content on load would make a scene file an executable one; ScriptHost holds
// the same line for behaviour scripts. Regeneration is always something the user
// asked for.

#ifndef THREEPP_EDITOR_GENERATORCONFIG_HPP
#define THREEPP_EDITOR_GENERATORCONFIG_HPP

#include "threepp/extras/editor/ScriptConfig.hpp"

#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct GeneratorConfig {

        // Inline source, verbatim (newlines and all).
        std::string source;
        // Exposed parameters, insertion-ordered so a saved document is
        // byte-identical when nothing changed.
        std::vector<ScriptConfig::Field> fields;

        static constexpr const char* sourceKey = "generatorSource";
        static constexpr const char* fieldsKey = "generatorFields";
        // Marks the child a generator produced. Read it to find the output to
        // replace; never write through it from anywhere else.
        static constexpr const char* generatedKey = "generated";

        [[nodiscard]] bool empty() const { return source.empty(); }

        // --- field access ---------------------------------------------------
        // Same semantics as ScriptConfig's, and the same encoding underneath.
        [[nodiscard]] std::optional<std::string> field(const std::string& name) const;
        void setField(const std::string& name, const std::string& value);
        void eraseField(const std::string& name);
        // Drops every field not in `names`, so a script that stopped exposing a
        // parameter does not leave its value in the document forever.
        void retainFields(const std::vector<std::string>& names);

        // --- document round trip --------------------------------------------
        // nullopt when the object carries no generator.
        [[nodiscard]] static std::optional<GeneratorConfig> read(const Object3D& object);
        [[nodiscard]] static bool isGenerator(const Object3D& object);

        // An empty config removes every entry, so clearing a generator leaves no
        // trace in the saved file.
        void write(Object3D& object) const;
        static void erase(Object3D& object);

        // The child this generator last produced, or nullptr. Re-found on every
        // call rather than cached: a regenerate replaces the node, and a scene
        // reload replaces the whole graph.
        [[nodiscard]] static Object3D* generatedChild(const Object3D& generator);

        bool operator==(const GeneratorConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_GENERATORCONFIG_HPP
