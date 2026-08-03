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
// soft_body handles onto the bodies the play session is ALREADY simulating, and
// the matching read handles onto the sensors it is already running. That is the
// editor's own state, handed over, not a second physics or sensor stack. Both
// are compiled only where the PhysX SDK was found.
//
// threepp.editor.script_from_object is the same idea pointed at the session's
// own instances rather than at physics, and is therefore NOT gated on anything:
// scripts exist wherever this module does.

#include "ScriptHost.hpp"

#include "ScriptTasks.hpp"

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
#include <string>

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
    // threepp.editor.script_from_object — one script reaching another's live
    // instance. Registered unconditionally, unlike the handles below it: this
    // needs nothing but a running session.
    threepp::editor::scripting::initScriptLookup(m);
    // threepp.editor.start_coroutine and its wait/until/Task — a scheduler the
    // play session pumps once per frame. Registered unconditionally for the
    // same reason: a coroutine needs a running script and nothing else. It
    // execs its Python half into the submodule right here, so the names exist
    // from the first `import threepp` rather than from the first Play.
    threepp::editor::scripting::initTasks(m);
    tp::init_materials(m);
    tp::init_objects(m);
    tp::init_animation(m);
    tp::init_cameras(m);
    tp::init_lights(m);
    tp::init_robot(m);
#ifdef THREEPP_EDITOR_WITH_PHYSX
    // threepp.PhysxWorld and its RigidBody / Articulation / PhysxMaterial — the
    // SAME translation unit the wheel builds, so a script that wants to give a
    // spawned mesh a body writes what a standalone threepp program writes rather
    // than waiting for an editor-shaped verb to be invented for it.
    //
    // This used to be withheld, and the reason was sound: nothing may stand up a
    // SECOND PhysxWorld beside the one the play session is stepping (one
    // PxFoundation, one world). Withholding the whole TYPE was the blunt way to
    // enforce it — a script asking for threepp.PhysxWorld got a NameError that
    // says nothing. denyWorldConstruction() below replaces the constructor with
    // one that explains itself, and threepp.editor.world() hands back the world
    // that already exists. Before init_editor_physics, which binds its own
    // lifetime-checked threepp.editor.RigidBody: a DIFFERENT C++ type in a
    // different namespace, so the two registrations do not collide.
    tp::init_physx(m);
    threepp::editor::scripting::denyWorldConstruction(m);
    // threepp.editor.rigid_body_from_object / soft_body_from_object — handles
    // onto the bodies the play session is already simulating, lifetime-checked
    // against the session in a way a raw threepp.RigidBody is not. Must follow
    // init_editor, which owns the submodule.
    tp::init_editor_physics(m);
    // threepp.editor.imu_from_object and friends — the same idea for the
    // sensors the play session is running, which is how a script closes a loop
    // on NOISY measurements rather than on the simulation's ground truth.
    tp::init_editor_sensors(m);
    // threepp.editor.Collision — the payload of on_collision_enter /
    // on_collision_exit. Nothing produces one without a physics world, so like
    // the handles above the name is simply absent in a build without the SDK.
    threepp::editor::scripting::initCollision(m);
#endif
}

namespace threepp::editor::scripting {

    void initCollision(py::module_& m) {

        // The submodule init_editor already made — never a second one, or the
        // names would land somewhere no script imports.
        auto sub = m.attr("editor");

        py::class_<Collision>(
                sub, "Collision",
                "One touch, as on_collision_enter / on_collision_exit are handed it.\n\n"
                "A value copied out of the physics report that produced it, so keeping it "
                "is safe. `other` is the object on the far side of the touch, as its "
                "concrete type - or None when that body belongs to nothing the script can "
                "see. The contact geometry describes the ENTER only; an exit has no "
                "manifold left to read and carries zeros.")
                .def_property_readonly(
                        "other", [](const Collision& c) { return c.other; },
                        "The other object, as its concrete type (Mesh, Robot, ...), or None.")
                .def_readonly("point", &Collision::point,
                              "WORLD-SPACE point of the hardest-hit manifold point, at the "
                              "substep the touch began.")
                .def_readonly("normal", &Collision::normal,
                              "Unit contact normal at that point, pointing INTO this script's "
                              "body - the direction the other body is pushing it.")
                .def_readonly("impulse", &Collision::impulse,
                              "Total impulse over the manifold (N*s), same orientation. Divide "
                              "by the substep to read it as a force.")
                .def("__repr__", [](const Collision& c) {
                    std::string other = "None";
                    if (!c.other.is_none()) {
                        try {
                            other = py::cast<std::string>(
                                    c.other.attr("__class__").attr("__name__"));
                        } catch (const py::error_already_set&) {
                            other = "?";
                        }
                    }
                    return "<threepp.editor.Collision other=" + other + ">";
                });
    }

    void denyWorldConstruction(py::module_& m) {

        // Assigning over the pybind-installed __init__ leaves __new__ and every
        // bound method alone: the object is still allocated, and the raise comes
        // from initialisation — which is early enough that no PhysX call has
        // been made and nothing needs unwinding.
        auto cls = m.attr("PhysxWorld");
        cls.attr("__init__") = py::cpp_function(
                [](const py::args&, const py::kwargs&) {
                    throw std::runtime_error(
                            "threepp.PhysxWorld cannot be constructed inside the editor: the "
                            "play session owns the one world, and a second would bring up a "
                            "second PhysX foundation beside it. Use threepp.editor.world() to "
                            "get the world that is already running.");
                },
                py::is_method(cls), py::name("__init__"),
                py::doc("Unavailable in the editor - use threepp.editor.world()."));
    }

    void initScriptLookup(py::module_& m) {

        // def_submodule hands back the submodule init_editor already created
        // (PyImport_AddModule is idempotent), so this adds a name to it rather
        // than shadowing it — the same call bind_editor_authoring.cpp makes.
        auto sub = m.def_submodule("editor");

        sub.def(
                "script_from_object", [](const py::handle& h) -> py::object {
                    const auto& resolver = scriptResolver();
                    // Nothing playing: no session has installed one. None rather
                    // than a raise, because "the neighbour is not there" is a
                    // normal condition a script tests for, exactly as it is for
                    // rigid_body_from_object.
                    if (!resolver.lookup) return py::none();
                    const auto object = threepp_py::as_object3d(h);
                    if (!object) return py::none();
                    // The EXACT node, by uuid — no walk up the ancestry. A
                    // physics lookup walks because a collider governs a whole
                    // subtree; a script does not govern anything but the object
                    // it was authored on, and answering with a parent's script
                    // would invent a relationship nobody wrote down.
                    return resolver.lookup(object->uuid);
                },
                py::arg("object"),
                "The live script instance running on `object`, or None.\n\n"
                "The instance IS the API: call its methods and read or write its attributes "
                "to signal it. Returns None when nothing is playing, when `object` carries no "
                "script, or when that script's instance failed - a disabled script is dead to "
                "a lookup. The object must be the exact one the script is authored on; unlike "
                "rigid_body_from_object this does not walk up the scene graph.\n\n"
                "Every instance exists before any start() runs, so resolving a neighbour in "
                "start() works whatever order the scene is in. Do not keep the reference "
                "across Play sessions - the next Play builds new instances.");
    }

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
