// threepp.editor, authoring half — what a GENERATOR script uses to put the
// content it builds into the document.
//
// Editor-only, like bind_editor_physics.cpp: the wheel has no document to author
// into, so these names exist only in the editor's embedded module. Must be
// registered AFTER init_editor, which creates the submodule.
//
// The contract is deliberately narrow. A generator does not get the document to
// mutate however it likes; it gets ONE verb, add(), which appends to the sink the
// editor opened for this run (see scripting::authoringSink). The editor then
// commits that finished subtree as a single undoable step. Consequences worth
// knowing:
//
//   * outside a generator run there is no sink, and add() raises rather than
//     silently doing nothing — a behaviour script calling it during Play is a
//     mistake, and the document is exactly what Play must not touch;
//   * a script that raises halfway has added nothing to the document, because
//     nothing was in the document yet;
//   * the objects are plain threepp objects, so everything else the module can
//     already build (geometries, materials, InstancedMesh, lights) composes
//     without any of it knowing about the editor.
//
// scene() and selected() are READ access, for scripts that place content
// relative to what is already there — scatter onto that terrain, one crate per
// marker. They hand back the live graph, so a script CAN reach in and mutate it
// directly; that is not authored content and will not be saved as such, and the
// docstrings say so rather than pretending the door is locked.
//
// scene() is the one name here that a BEHAVIOUR script uses too: during Play it
// answers with the scene the play session is running (scripting::playScene),
// which is how a script reaches a neighbour by name. It is the same verb asking
// the same question — "what is around me" — so it is one binding rather than
// two spellings of it, and the fallback below is the whole of the difference.

#include "bindings.hpp"

#include "Scripting.hpp"

#include "threepp/core/Object3D.hpp"

#include <pybind11/stl.h>

#include <string>

using namespace threepp;

namespace threepp_py {
namespace {

    // Same handle-not-typed-pointer rule the rest of threepp.editor follows:
    // threepp's virtual bases make a typed shared_ptr<Object3D> parameter
    // unusable, so every editor entry point takes py::handle and converts.
    std::shared_ptr<Object3D> sinkRelative(const py::handle& parent) {

        auto* sink = editor::scripting::authoringSink();
        if (!sink) {
            throw std::runtime_error(
                    "threepp.editor.add: nothing is being generated right now. This is for a "
                    "scene generator script, run from the editor's Generator section; a "
                    "behaviour script must not add to the document while playing.");
        }
        if (parent.is_none()) return {};
        auto object = threepp_py::as_object3d(parent);
        if (!object) {
            throw std::runtime_error("threepp.editor.add: `parent` is not an Object3D");
        }
        return object;
    }

}// namespace

    void init_editor_authoring(py::module_& m) {

        auto sub = m.def_submodule("editor");

        sub.def(
                "add", [](const py::handle& object, const py::handle& parent) -> py::object {
                    auto explicitParent = sinkRelative(parent);
                    auto child = as_object3d(object);
                    if (!child) {
                        throw std::runtime_error("threepp.editor.add: not an Object3D");
                    }
                    Object3D* target = explicitParent ? explicitParent.get()
                                                      : editor::scripting::authoringSink();
                    target->add(child);
                    // Hand the object straight back so a script can keep building
                    // on it: parent = editor.add(group).
                    return py::cast(child);
                },
                py::arg("object"), py::arg("parent") = py::none(),
                "Add `object` to what this generator is building, and return it. With no "
                "`parent` it goes at the generator's root; pass one of your own earlier adds "
                "to nest. Raises outside a generator run.");

        sub.def(
                "scene", []() -> py::object {
                    // A generator run first, because during one that IS the
                    // scene the caller means — and the two never overlap anyway:
                    // the editor does not generate into a document it is
                    // playing.
                    auto* scene = editor::scripting::authoringScene();
                    if (!scene) scene = editor::scripting::playScene();
                    if (!scene) {
                        throw std::runtime_error(
                                "threepp.editor.scene: no scene right now. This answers "
                                "during a generator run and from a behaviour script's "
                                "start/update/stop while playing — nowhere else.");
                    }
                    return py::cast(scene->shared_from_this());
                },
                "The live scene: what a generator is authoring into, or what a behaviour "
                "script is playing in.\n\n"
                "READ it to reach what you did not author — `scene.get_object_by_name(\"Ground\")`, "
                "`scene.children`. A generator places content relative to what already exists (a "
                "marker to put a crate on, a surface to scatter over); objects reached that way "
                "are NOT its output and are not replaced when it re-runs, only what you pass to "
                "add() is.\n\n"
                "During Play this answers from start() onwards, including update(), "
                "fixed_update() and the collision and trigger callbacks — so a script that needs "
                "the scene later does not have to stash it. Raises when nothing is generating "
                "and nothing is playing.");

        // Input, for a BEHAVIOUR script during Play — the other half of "a play session is a
        // thing you can drive". Poll it from update(); it never sticks.
        sub.def(
                "is_key_down", [](const std::string& key) {
                    auto& provider = editor::scripting::keyStateProvider();
                    // No window (headless, or a front end that installed no provider) reads as
                    // "nothing is held" rather than raising: a script that steers with the
                    // arrow keys should still RUN in a headless pass, just uncommanded.
                    return provider ? provider(key) : false;
                },
                py::arg("key"),
                "Poll whether a key is currently held — 'W', 'SPACE', 'UP', 'LEFT', 'KP8', the "
                "same names Canvas.is_key_down takes. Answers False while the user is typing "
                "into a field, so driving a robot cannot eat somebody's rename, and False in a "
                "build or a pass with no window. Query it every update() for continuous "
                "control; it never sticks.");
    }

}// namespace threepp_py
