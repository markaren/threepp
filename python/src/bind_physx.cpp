// Rigid-body physics — PhysX via threepp's PhysxWorld, exposed to Python.
//
// Compiled unconditionally; the body is active only when threepp can see the
// omniverse-physx-sdk (THREEPP_PY_HAS_PHYSX, defined by python/CMakeLists when
// find_package(unofficial-omniverse-physx-sdk) succeeds). Otherwise init_physx
// only sets HAS_PHYSX = False so Python can branch on it.
//
// Scope: rigid bodies — dynamic / static colliders from Box/Sphere/Capsule
// meshes, convex hulls, static triangle meshes (and whole subtrees), instanced
// bodies — plus a fixed-timestep step() that drives the bound visuals. Soft
// bodies and PxVehicle2 (which need the CUDA/GPU path) are intentionally left
// out of this first cut.
//
// Pointer-safety note: PhysxWorld::add(Mesh&) is given the *concrete* Mesh& by
// pybind (safe — the virtual Object3D base is never crossed here). The up-cast
// to Object3D& happens inside PhysxWorld::bind(), in pure C++, where the
// compiler adjusts the virtual base correctly — unlike pybind. The only place
// an Object3D is handed across the boundary is add_static_trimesh_tree(), which
// routes through as_object3d() (the established workaround).
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_PHYSX

#include <pybind11/stl.h>

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"

#include <PxPhysicsAPI.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;// the threepp_py::py alias isn't visible in the anon namespace
using namespace threepp;

namespace {

    // Thin Python-facing handle over a PhysX actor created through PhysxWorld.
    // The actor is owned by the world's PxScene, so a RigidBody is only valid
    // while its PhysxWorld lives — enforced with keep_alive on the adders.
    class RigidBody {
    public:
        explicit RigidBody(::physx::PxRigidActor* a) : actor_(a) {}

        Vector3 position() const { return fromPxVec3(actor_->getGlobalPose().p); }
        Quaternion quaternion() const { return fromPxQuat(actor_->getGlobalPose().q); }
        void setPose(const Vector3& p, const Quaternion& q) { actor_->setGlobalPose(toPxTransform(p, q)); }
        bool isDynamic() const { return actor_->is<::physx::PxRigidDynamic>() != nullptr; }

        void setLinearVelocity(const Vector3& v) { dyn()->setLinearVelocity(toPxVec3(v)); }
        Vector3 linearVelocity() const { return fromPxVec3(dyn()->getLinearVelocity()); }
        void setAngularVelocity(const Vector3& v) { dyn()->setAngularVelocity(toPxVec3(v)); }
        Vector3 angularVelocity() const { return fromPxVec3(dyn()->getAngularVelocity()); }
        void addForce(const Vector3& v) { dyn()->addForce(toPxVec3(v), ::physx::PxForceMode::eFORCE); }
        void addImpulse(const Vector3& v) { dyn()->addForce(toPxVec3(v), ::physx::PxForceMode::eIMPULSE); }
        void setLinearDamping(float d) { dyn()->setLinearDamping(d); }
        void setAngularDamping(float d) { dyn()->setAngularDamping(d); }
        float mass() const { return dyn()->getMass(); }
        void wakeUp() { dyn()->wakeUp(); }
        void setKinematic(bool k) { dyn()->setRigidBodyFlag(::physx::PxRigidBodyFlag::eKINEMATIC, k); }
        void setKinematicTarget(const Vector3& p, const Quaternion& q) { dyn()->setKinematicTarget(toPxTransform(p, q)); }

    private:
        ::physx::PxRigidDynamic* dyn() const {
            auto* d = actor_->is<::physx::PxRigidDynamic>();
            if (!d) throw std::runtime_error("RigidBody: this operation needs a dynamic body (this one is static)");
            return d;
        }
        ::physx::PxRigidActor* actor_;
    };

}// namespace

namespace threepp_py {

    void init_physx(py::module_& m) {

        py::class_<RigidBody>(m, "RigidBody",
                              "Handle to a PhysX actor created via PhysxWorld. Valid only while its world "
                              "is alive. Velocity/force/kinematic operations require a dynamic body.")
                .def_property_readonly("is_dynamic", &RigidBody::isDynamic)
                .def_property_readonly("position", &RigidBody::position)
                .def_property_readonly("quaternion", &RigidBody::quaternion)
                .def("set_pose", &RigidBody::setPose, py::arg("position"), py::arg("quaternion") = Quaternion())
                .def_property("linear_velocity", &RigidBody::linearVelocity, &RigidBody::setLinearVelocity)
                .def_property("angular_velocity", &RigidBody::angularVelocity, &RigidBody::setAngularVelocity)
                .def("set_linear_velocity", &RigidBody::setLinearVelocity, py::arg("v"))
                .def("set_angular_velocity", &RigidBody::setAngularVelocity, py::arg("v"))
                .def("add_force", &RigidBody::addForce, py::arg("force"),
                     "Apply a continuous force (N), consumed by the next step().")
                .def("add_impulse", &RigidBody::addImpulse, py::arg("impulse"),
                     "Apply an instantaneous impulse (kg·m/s).")
                .def("set_linear_damping", &RigidBody::setLinearDamping, py::arg("d"))
                .def("set_angular_damping", &RigidBody::setAngularDamping, py::arg("d"))
                .def_property_readonly("mass", &RigidBody::mass)
                .def("wake_up", &RigidBody::wakeUp)
                .def("set_kinematic", &RigidBody::setKinematic, py::arg("kinematic"),
                     "Toggle kinematic mode: the body is driven by set_kinematic_target and ignores forces/gravity.")
                .def("set_kinematic_target", &RigidBody::setKinematicTarget,
                     py::arg("position"), py::arg("quaternion") = Quaternion());

        py::class_<PhysxWorld>(m, "PhysxWorld",
                               "A PhysX rigid-body world wired to the threepp scene graph. Add meshes as "
                               "bodies, then call step(dt) each frame; every bound mesh's position/quaternion "
                               "follows the simulation. Pure CPU — no canvas or renderer required.")
                .def(py::init([](const Vector3& gravity, float fixed_timestep, int max_substeps,
                                 unsigned num_threads, bool gpu_dynamics) {
                         PhysxWorld::Settings s;
                         s.gravity = gravity;
                         s.fixedTimestep = fixed_timestep;
                         s.maxSubSteps = max_substeps;
                         s.numThreads = num_threads;
                         s.enableGpuDynamics = gpu_dynamics;
                         return std::make_unique<PhysxWorld>(s);
                     }),
                     py::arg("gravity") = Vector3(0, -9.81f, 0),
                     py::arg("fixed_timestep") = 1.f / 60.f,
                     py::arg("max_substeps") = 4,
                     py::arg("num_threads") = 2,
                     py::arg("gpu_dynamics") = false,
                     "gpu_dynamics requires a CUDA GPU and is only needed for soft bodies (not yet exposed).")
                .def("step", &PhysxWorld::step, py::arg("dt"),
                     "Advance the simulation by dt seconds (variable-rate caller, fixed-rate physics). "
                     "After it returns, every bound mesh's transform reflects the new state.")
                .def("set_gravity", &PhysxWorld::setGravity, py::arg("gravity"))
                .def("add", [](PhysxWorld& w, Mesh& mesh, float density) { return RigidBody(w.add(mesh, density)); },
                     py::arg("mesh"), py::arg("density") = 1000.f,
                     py::keep_alive<1, 2>(), py::keep_alive<0, 1>(),
                     "Add a dynamic body whose shape is inferred from the mesh's Box/Sphere/Capsule "
                     "geometry; the mesh is bound so it follows the sim. Returns a RigidBody.")
                .def("add_static", [](PhysxWorld& w, Mesh& mesh) { return RigidBody(w.addStatic(mesh)); },
                     py::arg("mesh"), py::keep_alive<0, 1>(),
                     "Add a static collider inferred from the mesh's Box/Sphere/Capsule geometry.")
                .def("add_dynamic_convex",
                     [](PhysxWorld& w, Mesh& mesh, float density) {
                         auto* a = w.addDynamicConvex(mesh, density);
                         if (!a) throw std::runtime_error("add_dynamic_convex: mesh has no usable position geometry");
                         return RigidBody(a);
                     },
                     py::arg("mesh"), py::arg("density") = 1000.f,
                     py::keep_alive<1, 2>(), py::keep_alive<0, 1>(),
                     "Add a dynamic body as the convex hull of the mesh's vertices (arbitrary shapes).")
                .def("add_static_trimesh",
                     [](PhysxWorld& w, Mesh& mesh) {
                         auto* a = w.addStaticTrimesh(mesh);
                         if (!a) throw std::runtime_error("add_static_trimesh: mesh has no triangle geometry");
                         return RigidBody(a);
                     },
                     py::arg("mesh"), py::keep_alive<0, 1>(),
                     "Add a static collider matching the mesh triangles exactly (static/kinematic only).")
                .def("add_static_trimesh_tree",
                     [](PhysxWorld& w, const py::handle& root) {
                         auto obj = as_object3d(root);
                         std::vector<RigidBody> out;
                         for (auto* a : w.addStaticTrimeshTree(*obj)) out.emplace_back(a);
                         return out;
                     },
                     py::arg("root"),
                     // No keep_alive on the returned list (a Python list can't be a weakref
                     // target); the caller is expected to hold the world for its lifetime.
                     "Add every Mesh under `root` as its own static trimesh collider — e.g. turn an "
                     "imported glTF environment straight into collision geometry. Returns a list.")
                .def("add_instanced",
                     [](PhysxWorld& w, InstancedMesh& mesh, float density) {
                         std::vector<RigidBody> out;
                         for (auto* a : w.add(mesh, density)) out.emplace_back(a);
                         return out;
                     },
                     py::arg("mesh"), py::arg("density") = 1000.f, py::keep_alive<1, 2>(),
                     // keep_alive<1,2> (world keeps the mesh) only; no keep_alive on the list
                     // return — hold the world yourself while stepping.
                     "Add one dynamic body per instance of an InstancedMesh. Returns a list of RigidBody.")
                .def("on_pre_substep",
                     [](PhysxWorld& w, py::function cb) {
                         w.onPreSubstep([cb](float dt) { py::gil_scoped_acquire g; cb(dt); });
                     },
                     py::arg("callback"), "Register callback(dt) fired before each fixed substep.")
                .def("on_post_substep",
                     [](PhysxWorld& w, py::function cb) {
                         w.onPostSubstep([cb](float dt) { py::gil_scoped_acquire g; cb(dt); });
                     },
                     py::arg("callback"), "Register callback(dt) fired after each fixed substep.");

        m.attr("HAS_PHYSX") = true;
    }

}// namespace threepp_py

#else// THREEPP_PY_HAS_PHYSX not defined — no PhysX in this build

namespace threepp_py {

    void init_physx(py::module_& m) {
        m.attr("HAS_PHYSX") = false;// Python can check availability: threepp.HAS_PHYSX
    }

}// namespace threepp_py

#endif
