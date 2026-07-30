// Python scripting for the editor — the part the rest of the editor talks to.
//
// This header is deliberately free of pybind11 and Python headers: the
// inspector, EditorApp and the tests include it, and none of them should pay
// for (or fight with) the CPython headers. Everything Python-shaped lives
// behind ScriptHost.hpp inside this directory.
//
// The whole directory is compiled only when Python (Development.Embed) and
// pybind11 are available — see apps/editor/CMakeLists.txt and the
// THREEPP_EDITOR_WITH_PYTHON define. The editor builds and runs without it; the
// inspector then authors ScriptConfig without being able to discover fields.

#ifndef THREEPP_EDITOR_SCRIPTING_HPP
#define THREEPP_EDITOR_SCRIPTING_HPP

#include "threepp/extras/editor/PlaySession.hpp"

#include "threepp/core/Object3D.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace threepp::editor {

    // One exposed class attribute: a plain, non-underscore, non-callable
    // int/float/bool/str defined on the script class.
    struct ScriptField {

        enum class Type {
            Bool,
            Int,
            Float,
            String
        };

        std::string name;
        Type type = Type::Float;
        // The class attribute's own value, as text — what the inspector shows
        // for a field the document has no override for.
        std::string defaultValue;
    };

    namespace scripting {

        // Where threepp.editor.add() puts what a GENERATOR script builds, for the
        // duration of one run. Null outside a run, and add() then refuses rather
        // than guessing — an authoring call from a behaviour script during Play
        // has no business mutating the document.
        //
        // The sink is deliberately a node that is NOT yet in the document: the
        // script fills it off-graph, and the editor commits the finished subtree
        // as ONE undoable step afterwards. A script that raises halfway therefore
        // commits nothing at all, with no rollback machinery needed.
        //
        // One instance across every TU (inline function-local static), so the
        // binding in python/src and the editor app agree without a link-time
        // dependency between them.
        inline Object3D*& authoringSink() {

            static Object3D* sink = nullptr;
            return sink;
        }

        // The live scene the current generator is authoring INTO, for scripts that
        // place content relative to what is already there. Carried separately from
        // the sink precisely because the sink is detached during the run, so it
        // cannot be reached by walking up from it. Set and cleared together with
        // the sink.
        inline Object3D*& authoringScene() {

            static Object3D* scene = nullptr;
            return scene;
        }

        // Key state for a script that wants to be DRIVEN — threepp.editor.is_key_down.
        //
        // Installed by the editor app, which owns the window and the ImGui context and is
        // therefore the only thing that can say both "is this key held" and "is the user
        // actually typing into a text field right now". Unset (a headless test, a front end
        // with no window) means no input is available and the binding answers False rather
        // than guessing.
        //
        // Takes the key NAME, not a keycode: the app answers from ImGui's key state, whose
        // enum python/src has no business knowing about. Same inline function-local static
        // as authoringSink() — the binding lives in python/src and the app in apps/editor,
        // and neither links the other.
        inline std::function<bool(const std::string&)>& keyStateProvider() {

            static std::function<bool(const std::string&)> provider;
            return provider;
        }

        // Result of looking at a script without running any of its behaviour.
        struct Inspection {
            std::string className;
            std::vector<ScriptField> fields;
            // Non-empty when the file could not be loaded or no class could be
            // picked; `fields` is then empty and the inspector says so.
            std::string error;
        };

        // Starts the embedded interpreter if it is not running yet. Every entry
        // point below does this itself; it is exposed because "did Python come
        // up at all" is worth reporting once, from the console.
        bool ensureInterpreter(std::string* error = nullptr);
        [[nodiscard]] bool interpreterStarted();

        // Loads `path` fresh (previous module state is purged, so editing the
        // file and asking again picks up the change), finds the script class and
        // reports its exposed fields. Never throws.
        [[nodiscard]] Inspection inspect(const std::filesystem::path& path);

        // Same for inline source held in the document. `key` gives the
        // synthetic module its identity (pass the object uuid) and `label` is
        // what a traceback from this code will name it. Never throws.
        [[nodiscard]] Inspection inspectSource(const std::string& source, const std::string& key,
                                               const std::string& label);

        // Does `source` parse? Returns "" when it does, and the SyntaxError
        // with its line number when it does not. compile() only — nothing in
        // the script runs, which is what makes this safe to call from a text
        // editor's Apply button. Never throws.
        [[nodiscard]] std::string checkSyntax(const std::string& source, const std::string& label);

    }// namespace scripting


    // Runs every object's attached script for the duration of a Play.
    //
    // start()  compiles each script fresh — the referenced .py, or the inline
    //          source the document carries — instantiates its class, applies
    //          the authored field values and calls start(obj) with a handle to
    //          the object — always its CONCRETE type (Mesh, Robot, Light, ...),
    //          held by shared_ptr, so a script that stashes it outlives the
    //          session safely.
    // update() calls update(dt) on each live instance, GIL acquired once for the
    //          whole sweep.
    // stop()   calls stop() and drops every instance while holding the GIL.
    //
    // A script that raises is logged once, disabled for the rest of the session
    // and left behind; the others keep running and the editor keeps playing.
    class ScriptPlaySession: public PlaySession {

    public:
        ScriptPlaySession();
        ~ScriptPlaySession() override;

        void start(Scene& scene) override;
        void update(float dt) override;
        void stop() override;

        [[nodiscard]] std::string name() const override { return "ScriptPlaySession"; }

        // Console sink. Called on the main thread, from start/update/stop only.
        void setLogger(std::function<void(const std::string&)> logger);

        // Errors from the last session, keyed by object uuid. Kept after stop()
        // so the inspector can show what went wrong where.
        [[nodiscard]] std::string errorFor(const std::string& uuid) const;
        [[nodiscard]] std::size_t errorCount() const;
        void clearErrors();

        // How many script instances are currently live (diagnostics/tests).
        [[nodiscard]] std::size_t instanceCount() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SCRIPTING_HPP
