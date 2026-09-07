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

#include <cstdint>
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

        // The live scene a PLAY session is running — the other thing
        // threepp.editor.scene() can be answering for. Set and cleared by
        // ScriptPlaySession together with its resolver, so the two halves of
        // "what can a behaviour script reach" share one lifetime and one place
        // to go down.
        //
        // Deliberately a second static rather than reusing authoringScene():
        // that one means "what this generator is authoring INTO", a sentence
        // about a detached sink that has no meaning during Play, and one
        // pointer standing for two different questions is how a stale answer
        // gets handed to the wrong caller. Same inline function-local static as
        // its neighbours — the binding lives in python/src and the session in
        // apps/editor, and neither links the other.
        inline Object3D*& playScene() {

            static Object3D* scene = nullptr;
            return scene;
        }

        // Whose method is running right now: the uuid of the script instance
        // ScriptPlaySession is currently dispatching into, or empty outside any
        // dispatch.
        //
        // threepp.editor.start_coroutine is what needs it. A coroutine has to
        // BELONG to somebody — a raise inside one disables an instance, and an
        // instance's tasks die with it — and nothing in the call itself says
        // who: a generator is a plain Python object and the scheduler is one
        // shared list. So ownership is read off the dispatch that is running,
        // which the session sets around EVERY call into script code (start,
        // update, fixed_update, the collision, trigger and break callbacks,
        // stop) and which the coroutine pump sets around each task it steps.
        //
        // Empty means nobody is being dispatched, and start_coroutine RAISES
        // rather than guessing — the same refusal threepp.editor.add makes
        // outside a generator run. The consequence worth knowing: a coroutine
        // started inside a NEIGHBOUR's method, reached through
        // script_from_object, belongs to the instance that was DISPATCHED, not
        // to the one whose code happened to run.
        //
        // Same inline function-local static as its neighbours, and free of
        // Python for the same reason: the session sets it and the binding reads
        // it, and neither wants to include the other.
        inline std::string& dispatchingScript() {

            static std::string uuid;
            return uuid;
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

        // The clock a script reads — threepp.editor.time.
        //
        // There are TWO clocks in a play session and they do not agree. update(dt)
        // is handed the wall-clock delta of the last frame, because that is what a
        // frame callback is; fixed_update(dt) is handed the physics world's
        // constant substep. Under load they diverge for good: PhysxWorld::step
        // takes at most maxSubSteps substeps per call and then DISCARDS the
        // leftover accumulator to avoid the spiral of death, so a hitching frame
        // advances wall time in full and simulated time only partly. A script
        // integrating update()'s dt against anything physics-driven drifts exactly
        // when frames hitch, and until now had no way to even ask which clock it
        // was on.
        //
        // So both are published here, refreshed by ScriptPlaySession once per
        // frame before its sweep and again at each substep before fixed_update.
        // The sim half is READ FROM THE WORLD rather than accumulated separately:
        // simTime is the same double that stamps sensor samples, so a script
        // comparing its own clock against a sample's timestamp is comparing one
        // number against itself.
        //
        // Same inline function-local static as its neighbours — the binding lives
        // in python/src and the session in apps/editor, and neither links the
        // other.
        struct ScriptClock {

            // Wall-clock seconds the last frame took: exactly what update(dt) was
            // handed, published so the other methods can see it too.
            float frameDt = 0.f;
            // Wall seconds since the session started — the sum of the above.
            double wallTime = 0.0;
            // Simulated seconds since the session started: the physics world's own
            // clock, which advances only when substeps actually run.
            double simTime = 0.0;
            // The fixed substep, constant for the run. Equals frameDt in the
            // degraded case below, where there is no fixed clock to report.
            float simDt = 0.f;
            // Substeps completed since the session started. Stays 0 without a
            // world — no substep has run, and saying otherwise would invent one.
            std::uint64_t steps = 0;
            // True when the sim half above comes from a playing physics world.
            // False in a build or a scene with no world, where simTime degrades to
            // wallTime and simDt to frameDt: still a usable elapsed-time answer,
            // and this flag is how a script tells the difference rather than
            // discovering it in a plot.
            bool fixedClock = false;
            // False outside Play, where every field above is zero.
            bool active = false;
        };

        inline ScriptClock& scriptClock() {

            static ScriptClock clock;
            return clock;
        }

        // Debug draw — threepp.editor.draw_line and friends, on their way to the
        // viewport. A script computes vectors nobody can see (an altimeter ray, a
        // contact normal, a drive target); these land here as world-space line
        // segments, and the editor drains the list into one overlay LineSegments
        // each rendered frame. Immediate mode: drained is gone, and a script
        // that wants a line to persist draws it again next update() — which it
        // is called every frame to do.
        //
        // Plain floats on purpose. Everything else in this header is
        // Python-free so the inspector and the tests can include it; this stays
        // dumb enough to also be written from C++ (a play session that wants to
        // show its own working lands in the same batch).
        //
        // `active` is the whole gate, set by ScriptPlaySession together with its
        // resolver: draws outside a session are cheap no-ops rather than raises,
        // exactly as is_key_down answers False with no window — a script that
        // draws must still RUN in a headless pass, just unseen.
        //
        // The cap is a backstop for a script drawing per body per substep and
        // getting it wrong. Refused segments are counted in `dropped`, and the
        // drainer says so once, rather than the editor freezing under a million
        // lines nobody meant to ask for.
        struct DebugDrawList {

            struct Segment {
                float ax, ay, az;
                float bx, by, bz;
                float r, g, b;
            };

            static constexpr std::size_t cap = 100000;

            std::vector<Segment> segments;
            std::size_t dropped = 0;
            bool active = false;

            void push(float ax, float ay, float az, float bx, float by, float bz,
                      float r, float g, float b) {

                if (!active) return;
                if (segments.size() >= cap) {
                    ++dropped;
                    return;
                }
                segments.push_back({ax, ay, az, bx, by, bz, r, g, b});
            }

            void clear() {

                segments.clear();
                dropped = 0;
            }
        };

        inline DebugDrawList& debugDraw() {

            static DebugDrawList list;
            return list;
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
    // start()  runs in TWO PHASES. Phase 1 compiles each script fresh — the
    //          referenced .py, or the inline source the document carries —
    //          instantiates its class and applies the authored field values.
    //          Phase 2 then calls start(obj) on every instance, with a handle to
    //          the object — always its CONCRETE type (Mesh, Robot, Light, ...),
    //          held by shared_ptr, so a script that stashes it outlives the
    //          session safely.
    //
    //          The split is what makes the guarantee behind
    //          threepp.editor.script_from_object true: BY THE TIME ANY start()
    //          RUNS, EVERY SCRIPT INSTANCE EXISTS, fields applied. Interleaved,
    //          a script resolving a neighbour in its own start() would succeed
    //          or fail on scene order alone.
    // update() calls update(dt) on each live instance, GIL acquired once for the
    //          whole sweep.
    // stop()   calls stop() and drops every instance while holding the GIL.
    //
    // A script may also define on_collision_enter(contact) /
    // on_collision_exit(contact), fired when the body governing its object
    // starts and stops touching another body. start() enables PhysX contact
    // reporting on that body for it — nobody ticks a box — and the reports,
    // which arrive from inside the solver, are QUEUED there and delivered from
    // update()'s sweep, before update(dt). The queue is a list rather than a
    // state flag, so a touch that begins and ends inside one frame still
    // produces enter followed by exit rather than nothing at all.
    //
    // A script may also define on_trigger_enter(other) / on_trigger_exit(other),
    // the same machinery over PhysX's TRIGGER reports: fired when a body enters
    // and leaves a trigger volume, for the script on the VOLUME and for the
    // script on the entering BODY alike, each handed the other object. Nothing
    // is enabled on the actor for these — a volume is a trigger because the
    // document says so — so start() only registers the delivery.
    //
    // A script may also define fixed_update(dt), which does NOT run per frame:
    // start() registers ONE pre-substep callback with the PhysxWorld the physics
    // session is playing (only when some live instance defines the method), and
    // that callback sweeps them once per fixed substep with the world's constant
    // timestep. It is where forces, impulses and drive targets belong — writing
    // those from update(dt) makes a controller frame-rate dependent. Without a
    // playing physics world there is no fixed clock, so the method never fires
    // and start() says so once; a fabricated clock would be a lie about the only
    // thing the name promises.
    //
    // A script may also start COROUTINES — threepp.editor.start_coroutine —
    // generators that yield a condition and are resumed once per frame when it
    // is met. The scheduler lives in ScriptTasks.cpp; this session pumps it once
    // per update(), AFTER the update() sweep and inside the same GIL
    // acquisition, so an until() predicate reads a world that has been stepped
    // and then updated. Tasks belong to the instance that was being dispatched
    // when they were registered (see scripting::dispatchingScript), which is
    // what makes a raise inside one disable that instance whole, like any other
    // method. They are session state: dropped at stop(), rebuilt from nothing on
    // the next Play.
    //
    // Both clocks are published to threepp.editor.time for the duration (see
    // ScriptClock): the wall half advances once per frame from update(), the sim
    // half is read back from the playing world here and again before every
    // fixed_update. The session owns that lifetime exactly as it owns the
    // resolver and the draw list — zeroed on the way up, zeroed on the way down.
    //
    // A script that raises is logged once, disabled for the rest of the session
    // and left behind; the others keep running and the editor keeps playing.
    // That covers fixed_update too, and disables the instance whole.
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
