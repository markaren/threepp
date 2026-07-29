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
                    auto* scene = editor::scripting::authoringScene();
                    if (!scene) {
                        throw std::runtime_error(
                                "threepp.editor.scene: only available while generating.");
                    }
                    return py::cast(scene->shared_from_this());
                },
                "The scene this generator is authoring into. READ it to place content "
                "relative to what already exists — a marker to put a crate on, a surface to "
                "scatter over. Objects reached this way are NOT this generator's output and "
                "are not replaced when it re-runs; only what you pass to add() is.");
    }

}// namespace threepp_py
