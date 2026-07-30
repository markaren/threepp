// The Python side of editor scripting: the embedded interpreter, the module the
// scripts import, the loader, and the object-handle factory.
//
// Internal to apps/editor/scripting — anything outside this directory should be
// including Scripting.hpp instead, which is free of Python headers.

#ifndef THREEPP_EDITOR_SCRIPTHOST_HPP
#define THREEPP_EDITOR_SCRIPTHOST_HPP

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include "threepp/math/Vector3.hpp"

#include <filesystem>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor::scripting {

    namespace py = pybind11;

    // What on_collision_enter / on_collision_exit are handed, bound as
    // `threepp.editor.Collision`.
    //
    // Deliberately small and already copied. The PhysX manifold it came from is
    // valid only for the duration of the contact callback, and this is delivered
    // a whole sweep later — so everything here is a value, taken at report time.
    // Built only at delivery, with the GIL held, which is the one moment `other`
    // (a concrete-typed object handle, or None) can exist at all.
    struct Collision {

        py::object other;
        // Zero on an exit event: a lost touch has no manifold left to read.
        Vector3 point;
        Vector3 normal;
        Vector3 impulse;
    };

    // Registers `Collision` into the threepp.editor submodule of `m`. Called
    // from the embedded module's body, after the submodule exists.
    void initCollision(py::module_& m);

    // Pulls the PYBIND11_EMBEDDED_MODULE translation unit into the link.
    //
    // The macro registers `threepp` as a built-in module from a static
    // initializer, and a static library drops any object file nothing refers to
    // — which is exactly what an initializer-only TU is. Calling this (from
    // ScriptHost.cpp, before Py_Initialize) is what keeps it.
    void registerEmbeddedModule();

    // A handle to `object` for a script, as its concrete leaf type.
    //
    // Mesh/Points/Line derive from Object3D VIRTUALLY, and pybind11 mishandles
    // every access that crosses that base (see the comment at the top of
    // python/src/bind_objects.cpp) — a handle cast as Object3D would let a
    // script corrupt the heap by assigning `name`. So dispatch on the concrete
    // type and cast that.
    //
    // Always shared_ptr-held (via shared_from_this): a script may keep the
    // handle past Stop, and the editor throws the whole scene away there. A
    // non-owning wrapper would be a dangling pointer with a virtual base, i.e.
    // a crash inside pybind11's own deregistration.
    //
    // GIL must be held.
    py::object handleFor(Object3D& object);

    // Runs an AUTHORING script: a module body, executed once for its side effects
    // on the document, with no class and no update() contract. Returns "" on
    // success, or a description of the failure (a Python traceback, or why the
    // interpreter is unavailable) — never throws, because the caller's job is to
    // put the reason in the console.
    //
    // Fresh globals per run, so a second Regenerate cannot behave differently
    // from the first by inheriting state. Acquires the GIL itself.
    std::string runAuthoringSource(const std::string& source, const std::string& label);

    // Loads `path` with fresh module state and returns the script class.
    // Throws py::error_already_set (a Python-level failure) or
    // std::runtime_error (no usable class). GIL must be held.
    py::object loadScriptClass(const std::filesystem::path& path, std::string& className);

    // Same, for source held in the document rather than in a file. `key` gives
    // the synthetic module its identity (the object uuid, so two objects never
    // share module state) and `label` is what tracebacks name the code after.
    // GIL must be held.
    py::object loadInlineScriptClass(const std::string& source, const std::string& key,
                                     const std::string& label, std::string& className);

    // "<inline:Box>" — what a traceback from inline source is filed under.
    [[nodiscard]] std::string inlineFilename(const std::string& label);

    // The loader/discovery helpers, as a Python module object. Built once per
    // interpreter. GIL must be held.
    py::object helpers();

    // Type/name/default triples for a script class's exposed attributes.
    // GIL must be held.
    py::object exposedFields(const py::object& cls);

    // Type + message + traceback, ready for the console. GIL must be held.
    std::string describeError(py::error_already_set& error);

}// namespace threepp::editor::scripting

#endif//THREEPP_EDITOR_SCRIPTHOST_HPP
