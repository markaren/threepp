// The `threepp` module the editor's embedded interpreter serves.
//
// Deliberately NOT the wheel-built threepp module. Importing that into an
// embedded interpreter would register a second, independent set of C++ types:
// the editor would hand a script an object whose type the module has never seen
// (or worse, one it has seen under a different registration), and pybind11 would
// either raise or hand back a wrapper around the wrong pointer.
//
// So the editor links the SAME per-area binding TUs the wheel is built from
// (python/src/bind_*.cpp) straight into itself and registers them through
// PYBIND11_EMBEDDED_MODULE. There is exactly one type registry, and it is this
// process's.
//
// Only the areas a scene script plausibly needs are linked — scene graph, math,
// geometry, curves, materials, animation, cameras, lights, robots. The renderer,
// loader, sensor and Vulkan areas are left out: they would drag windowing and
// device state into a script, and the editor already owns those.
//
// Physics is left out on the same grounds — a script has no business building a
// PhysxWorld — with one deliberate exception: threepp.editor's rigid_body /
// soft_body handles onto the bodies the play session is ALREADY simulating.
// That is the editor's own state, handed over read/write, not a second physics
// stack. It is compiled only where the PhysX SDK was found.

#include "ScriptHost.hpp"

#include "bindings.hpp"

#include "threepp/core/Object3D.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"

#include <memory>

using namespace threepp;

namespace tp = threepp_py;

PYBIND11_EMBEDDED_MODULE(threepp, m) {

    m.doc() = "threepp scene API, as the editor's script host exposes it.";

    // Same order dependency as python/src/bindings.cpp: math first, textures
    // before materials, core Object3D before anything deriving from it,
    // geometries + materials before objects.
    tp::init_math(m);
    tp::init_textures(m);
    tp::init_core(m);
    tp::init_geometries(m);
    tp::init_curves(m);// CatmullRomCurve3 — what an authored spline reads back as
    tp::init_editor(m);// threepp.editor — SplinePath, the no-boilerplate way to read one
    tp::init_editor_authoring(m);// threepp.editor.add — what a generator builds into
    tp::init_materials(m);
    tp::init_objects(m);
    tp::init_animation(m);
    tp::init_cameras(m);
    tp::init_lights(m);
    tp::init_robot(m);
#ifdef THREEPP_EDITOR_WITH_PHYSX
    // threepp.editor.rigid_body_from_object / soft_body_from_object. NOT the
    // general physics bindings — no PhysxWorld, no scene construction; just
    // handles onto the bodies the play session is already simulating. Must
    // follow init_editor, which owns the submodule.
    tp::init_editor_physics(m);
#endif
}

namespace threepp::editor::scripting {

    void registerEmbeddedModule() {

        // Nothing to do — being called is the point. The PYBIND11_EMBEDDED_MODULE
        // above registers itself from a static initializer, and this function is
        // the referenced symbol that stops the linker discarding the object file
        // that initializer lives in.
    }

    py::object handleFor(Object3D& object) {

        // Every node in an editor scene is owned by a shared_ptr (Object3D holds
        // its children as shared_ptr, and SceneDocument owns the root), so this
        // succeeds. A stack-allocated Object3D would throw bad_weak_ptr, which
        // the caller turns into a script error rather than a crash.
        const std::shared_ptr<Object3D> shared = object.shared_from_this();

        // Order matters: most-derived first, since as<>() is a dynamic_cast and
        // an InstancedMesh is also a Mesh.
        if (auto p = std::dynamic_pointer_cast<InstancedMesh>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<GrassMesh>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<Mesh>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<LineSegments>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<Line>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<Points>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<Sprite>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<Robot>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<Scene>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<Group>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<PerspectiveCamera>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<OrthographicCamera>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<AmbientLight>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<DirectionalLight>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<PointLight>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<SpotLight>(shared)) return py::cast(p);
        if (auto p = std::dynamic_pointer_cast<HemisphereLight>(shared)) return py::cast(p);

        // Plain Object3D (and anything whose leaf type is not bound): safe,
        // because it is not one of the virtual-base subclasses.
        return py::cast(shared);
    }

}// namespace threepp::editor::scripting
