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

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/PhysxGpuBatch.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/physx/UrdfArticulation.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"

#include <PxPhysicsAPI.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;// the threepp_py::py alias isn't visible in the anon namespace
using namespace threepp;
using namespace ::physx;// bare PhysX type names in the binding code (was provided by a now-removed
                        // using-directive inside the old in-file Articulation block)

namespace {

// Thin handle over a PxMaterial (surface friction + restitution). The PxMaterial is owned by the
// PxPhysics inside PhysxWorld and released with it, so this just holds the pointer (keep_alive ties
// it to the world). Properties are runtime-mutable so a domain-randomization loop can re-roll the
// friction/restitution of a per-env material in place each reset. Default restitution 0 (a bouncing
// foot is wrong for locomotion); the global defaultMaterial stays 0.2 for back-compat.
class PhysxMaterial {
public:
    explicit PhysxMaterial(::physx::PxMaterial* m) : mat_(m) {}
    ::physx::PxMaterial* raw() const { return mat_; }
    float staticFriction() const { return mat_->getStaticFriction(); }
    void setStaticFriction(float v) { mat_->setStaticFriction(v); }
    float dynamicFriction() const { return mat_->getDynamicFriction(); }
    void setDynamicFriction(float v) { mat_->setDynamicFriction(v); }
    float restitution() const { return mat_->getRestitution(); }
    void setRestitution(float v) { mat_->setRestitution(v); }

private:
    ::physx::PxMaterial* mat_;
};

// "average"|"min"|"multiply"|"max" -> PxCombineMode (how two contacting materials' coefficients mix).
inline ::physx::PxCombineMode::Enum combineModeFromString(const std::string& s) {
    using ::physx::PxCombineMode;
    if (s == "min") return PxCombineMode::eMIN;
    if (s == "multiply") return PxCombineMode::eMULTIPLY;
    if (s == "max") return PxCombineMode::eMAX;
    return PxCombineMode::eAVERAGE;
}

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

        py::class_<ArticulationLink>(m, "ArticulationLink",
                                     "A link of an Articulation plus its inbound revolute joint (the root has "
                                     "none). Valid while its Articulation/world live.")
                .def_property_readonly("is_root", &ArticulationLink::isRoot)
                .def_property_readonly("position", &ArticulationLink::position)
                .def_property_readonly("quaternion", &ArticulationLink::quaternion)
                .def_property_readonly("joint_position", &ArticulationLink::jointPosition, "Joint angle (radians).")
                .def_property_readonly("joint_velocity", &ArticulationLink::jointVelocity, "Joint angular velocity (rad/s).")
                .def("add_force", &ArticulationLink::addForce, py::arg("force"), "Apply an external force (N) to this link.")
                .def("add_impulse", &ArticulationLink::addImpulse, py::arg("impulse"), "Apply an external impulse (kg·m/s) — e.g. a random shove.")
                .def("set_drive_target", &ArticulationLink::setDriveTarget, py::arg("target"),
                     "Set the PD drive's target angle (radians).")
                .def("set_drive_velocity", &ArticulationLink::setDriveVelocity, py::arg("velocity"));

        py::class_<Articulation>(m, "Articulation",
                                 "A reduced-coordinate articulation (robot): a tree of links joined by "
                                 "motorized revolute joints. Build with add_link (root first), then "
                                 "finalize(); stepping the world drives the bound meshes.")
                .def("add_link",
                     [](Articulation& a, Mesh& mesh, const py::object& parent, float density,
                        const std::array<float, 3>& axis, const std::array<float, 3>& anchor,
                        const py::object& lower, const py::object& upper,
                        float stiffness, float damping, float max_force, float drive_target,
                        const std::string& joint_type, float joint_friction, const py::object& material) {
                         ArticulationLink* p = parent.is_none() ? nullptr : parent.cast<ArticulationLink*>();
                         const bool limited = !lower.is_none() && !upper.is_none();
                         const float lo = limited ? lower.cast<float>() : 0.f;
                         const float hi = limited ? upper.cast<float>() : 0.f;
                         ::physx::PxMaterial* mat = material.is_none() ? nullptr : material.cast<PhysxMaterial*>()->raw();
                         return a.addLink(p, mesh, density, axis, anchor, limited, lo, hi,
                                          stiffness, damping, max_force, drive_target, joint_type, joint_friction, mat);
                     },
                     py::arg("mesh"), py::arg("parent") = py::none(), py::arg("density") = 1000.f,
                     py::arg("axis") = std::array<float, 3>{0.f, 0.f, 1.f},
                     py::arg("anchor") = std::array<float, 3>{0.f, 0.f, 0.f},
                     py::arg("lower") = py::none(), py::arg("upper") = py::none(),
                     py::arg("stiffness") = 0.f, py::arg("damping") = 0.f,
                     py::arg("max_force") = 1e6f, py::arg("drive_target") = 0.f,
                     py::arg("joint_type") = "revolute", py::arg("joint_friction") = 0.0f,
                     py::arg("material") = py::none(),
                     py::keep_alive<1, 2>(), py::keep_alive<0, 1>(),
                     "Add a link. parent=None → the fixed/free root; otherwise attach an inbound joint at "
                     "world-space `anchor` along world-space `axis`. joint_type='revolute' (hinge about axis) "
                     "or 'prismatic' (slider along axis). lower/upper set the joint limits (radians for "
                     "revolute, metres for prismatic; omit both for a free axis); stiffness/damping/max_force "
                     "configure the PD drive (stiffness>0 motorizes it; leave 0 for a passive/force-controlled "
                     "joint). Shape is inferred from the mesh (Box/Sphere/Capsule). `material` (from "
                     "world.create_material) overrides the contact friction/restitution for this link's "
                     "shape — e.g. a grippy, restitution-0 foot, or a per-env material for friction "
                     "domain randomization; default uses the world's shared material. Returns an ArticulationLink.")
                .def("finalize", &Articulation::finalize,
                     "Add the finished articulation to the scene. No links may be added afterwards.")
                .def("reset", &Articulation::reset, py::arg("position"),
                     "Episode reset: teleport the root upright to `position` with zero velocity and "
                     "zero all joint positions/velocities (back to the neutral build pose).")
                .def("set_joint_positions",
                     [](Articulation& a, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                         a.setJointPositions(arr.data(), static_cast<std::size_t>(arr.size()));
                     },
                     py::arg("positions"),
                     "Set all joint positions (DOF order) and zero velocities — e.g. place a cart-pole "
                     "hanging straight down for a swing-up demo.")
                .def("joint_positions",
                     [](const Articulation& a) {
                         const auto v = a.jointPositions();
                         return py::array_t<float>(static_cast<py::ssize_t>(v.size()), v.data());
                     },
                     "All revolute joint angles (radians) as one numpy array, in add_link order.")
                .def("joint_velocities",
                     [](const Articulation& a) {
                         const auto v = a.jointVelocities();
                         return py::array_t<float>(static_cast<py::ssize_t>(v.size()), v.data());
                     },
                     "All revolute joint angular velocities (rad/s) as one numpy array.")
                .def("set_drive_targets",
                     [](Articulation& a, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                         a.setDriveTargets(arr.data(), static_cast<std::size_t>(arr.size()));
                     },
                     py::arg("targets"),
                     "Set every joint's PD drive target from one numpy array — the batched hot path "
                     "for vectorized stepping (one call instead of one per joint).")
                .def("root_state",
                     [](const Articulation& a) {
                         const auto s = a.rootState();
                         return py::array_t<float>(7, s.data());
                     },
                     "Root link world pose as numpy [px,py,pz, qx,qy,qz,qw] in one call.")
                .def("root_velocity",
                     [](const Articulation& a) {
                         const auto s = a.rootVelocity();
                         return py::array_t<float>(6, s.data());
                     },
                     "Root link world-frame velocity as numpy [vx,vy,vz, wx,wy,wz] — the base "
                     "linear + angular velocity a locomotion observation needs.")
                .def("dof_order",
                     [](const Articulation& a) {
                         const auto v = a.dofOrder();
                         return py::array_t<int>(static_cast<py::ssize_t>(v.size()), v.data());
                     },
                     "Per add-order joint, its low-level DOF slot in the direct-GPU joint buffers "
                     "(PhysX cache order != add-order). Use to map a GPU-trained policy back to the "
                     "CPU getters: obs_gpu[dof_order[i]] = cpu[i]; cpu_target[i] = gpu_target[dof_order[i]].");

        py::class_<PhysxMaterial>(m, "PhysxMaterial",
                                  "A contact material (surface friction + restitution). Create via "
                                  "world.create_material(...), pass to add_link/add/add_static. The "
                                  "static_friction / dynamic_friction / restitution properties are "
                                  "mutable at runtime — re-roll them each reset for per-env friction "
                                  "domain randomization (a key sim-to-real robustness lever).")
                .def_property("static_friction", &PhysxMaterial::staticFriction, &PhysxMaterial::setStaticFriction)
                .def_property("dynamic_friction", &PhysxMaterial::dynamicFriction, &PhysxMaterial::setDynamicFriction)
                .def_property("restitution", &PhysxMaterial::restitution, &PhysxMaterial::setRestitution)
                .def("set",
                     [](PhysxMaterial& m, float static_friction, float dynamic_friction, float restitution) {
                         m.setStaticFriction(static_friction);
                         m.setDynamicFriction(dynamic_friction);
                         m.setRestitution(restitution);
                     },
                     py::arg("static_friction"), py::arg("dynamic_friction"), py::arg("restitution"),
                     "Set all three coefficients at once (the domain-randomization hot path).");

        py::class_<PhysxWorld>(m, "PhysxWorld",
                               "A PhysX rigid-body world wired to the threepp scene graph. Add meshes as "
                               "bodies, then call step(dt) each frame; every bound mesh's position/quaternion "
                               "follows the simulation. Pure CPU — no canvas or renderer required.")
                .def(py::init([](const Vector3& gravity, float fixed_timestep, int max_substeps,
                                 unsigned num_threads, bool gpu_dynamics, bool direct_gpu,
                                 bool tgs_pcm, std::uintptr_t cuda_context) {
                         PhysxWorld::Settings s;
                         s.gravity = gravity;
                         s.fixedTimestep = fixed_timestep;
                         s.maxSubSteps = max_substeps;
                         s.numThreads = num_threads;
                         s.enableGpuDynamics = gpu_dynamics;
                         s.enableDirectGpu = direct_gpu;
                         s.enableTgsPcm = tgs_pcm;
                         s.cudaContext = reinterpret_cast<CUcontext>(cuda_context);
                         return std::make_unique<PhysxWorld>(s);
                     }),
                     py::arg("gravity") = Vector3(0, -9.81f, 0),
                     py::arg("fixed_timestep") = 1.f / 60.f,
                     py::arg("max_substeps") = 4,
                     py::arg("num_threads") = 2,
                     py::arg("gpu_dynamics") = false,
                     py::arg("direct_gpu") = false,
                     py::arg("tgs_pcm") = false,
                     py::arg("cuda_context") = 0,
                     "gpu_dynamics requires a CUDA GPU (needed for soft bodies). direct_gpu also "
                     "enables the PhysX direct-GPU API for batched GPU-resident articulation state "
                     "I/O (PhysxGpuBatch) — the basis for GPU vectorized RL. Under direct_gpu the "
                     "per-actor CPU getters and the binding-sync step() are NOT valid. tgs_pcm makes "
                     "a CPU world use the TGS solver + PCM + stabilization (the GPU path always does) "
                     "so its contact model MATCHES a GPU-trained policy for sim-to-sim deploy. "
                     "cuda_context "
                     "(an existing CUcontext as an int, e.g. torch's primary context) makes PhysX "
                     "share that context instead of creating its own — required to mix PhysX GPU work "
                     "with the framework's cuBLAS/cuDNN on the same device.")
                .def("step", &PhysxWorld::step, py::arg("dt"),
                     "Advance the simulation by dt seconds (variable-rate caller, fixed-rate physics). "
                     "After it returns, every bound mesh's transform reflects the new state.")
                .def("set_gravity", &PhysxWorld::setGravity, py::arg("gravity"))
                .def("add", [](PhysxWorld& w, Mesh& mesh, float density, const py::object& material) {
                         ::physx::PxMaterial* mat = material.is_none() ? nullptr : material.cast<PhysxMaterial*>()->raw();
                         return RigidBody(w.add(mesh, density, mat));
                     },
                     py::arg("mesh"), py::arg("density") = 1000.f, py::arg("material") = py::none(),
                     py::keep_alive<1, 2>(), py::keep_alive<0, 1>(),
                     "Add a dynamic body whose shape is inferred from the mesh's Box/Sphere/Capsule "
                     "geometry; the mesh is bound so it follows the sim. `material` (from create_material) "
                     "overrides the contact friction/restitution. Returns a RigidBody.")
                .def("add_static", [](PhysxWorld& w, Mesh& mesh, const py::object& material) {
                         ::physx::PxMaterial* mat = material.is_none() ? nullptr : material.cast<PhysxMaterial*>()->raw();
                         return RigidBody(w.addStatic(mesh, mat));
                     },
                     py::arg("mesh"), py::arg("material") = py::none(), py::keep_alive<0, 1>(),
                     "Add a static collider inferred from the mesh's Box/Sphere/Capsule geometry. "
                     "`material` (from create_material) sets its friction/restitution — e.g. a grippy floor.")
                .def("create_material",
                     [](PhysxWorld& w, float static_friction, float dynamic_friction, float restitution,
                        const std::string& friction_combine, const std::string& restitution_combine) {
                         ::physx::PxMaterial* m = w.physics().createMaterial(static_friction, dynamic_friction, restitution);
                         if (!m) throw std::runtime_error("create_material: PxPhysics::createMaterial failed");
                         m->setFrictionCombineMode(combineModeFromString(friction_combine));
                         m->setRestitutionCombineMode(combineModeFromString(restitution_combine));
                         return std::make_unique<PhysxMaterial>(m);
                     },
                     py::arg("static_friction") = 0.5f, py::arg("dynamic_friction") = 0.5f,
                     py::arg("restitution") = 0.0f, py::arg("friction_combine") = "average",
                     py::arg("restitution_combine") = "average", py::keep_alive<0, 1>(),
                     "Create a contact material. Defaults: friction 0.5/0.5, restitution 0 (no bounce — "
                     "right for feet/locomotion, unlike the world's shared 0.2 default). combine modes "
                     "('average'|'min'|'multiply'|'max') control how two contacting materials' coefficients "
                     "mix — use 'min' so a clean material governs a contact against a different one. The "
                     "returned PhysxMaterial is mutable (per-env friction randomization). Keeps the world alive.")
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
                     py::arg("callback"), "Register callback(dt) fired after each fixed substep.")
                .def("create_articulation",
                     [](PhysxWorld& w, bool fixed_base, int solver_position_iterations, bool disable_self_collision) {
                         return std::make_unique<Articulation>(w, fixed_base, solver_position_iterations, disable_self_collision);
                     },
                     py::arg("fixed_base") = false, py::arg("solver_position_iterations") = 8,
                     py::arg("disable_self_collision") = false, py::keep_alive<0, 1>(),
                     "Create a reduced-coordinate articulation (robot). fixed_base pins the root to the "
                     "world (use for arms; leave false for free-floating bodies like a walking robot). "
                     "Add links, then call finalize().")
                .def("load_articulation",
                     [](PhysxWorld& w, const std::string& path, bool fixed_base,
                        const std::array<float, 3>& base_position, float default_density,
                        float stiffness, float damping, float max_force, bool self_collision,
                        int solver_position_iterations, bool render_visuals) {
                         URDFArticulationOptions opts;
                         opts.fixedBase = fixed_base;
                         opts.basePosition = Vector3(base_position[0], base_position[1], base_position[2]);
                         opts.defaultDensity = default_density;
                         opts.stiffness = stiffness;
                         opts.damping = damping;
                         opts.maxForce = max_force;
                         opts.selfCollision = self_collision;
                         opts.solverPositionIterations = solver_position_iterations;
                         opts.renderVisuals = render_visuals;
                         auto r = loadArticulation(w, std::filesystem::path(path), opts);
                         if (!r.articulation) throw std::runtime_error("load_articulation: could not read URDF: " + path);
                         return std::make_tuple(std::move(r.articulation), std::move(r.meshes), std::move(r.jointNames));
                     },
                     py::arg("path"), py::arg("fixed_base") = false,
                     py::arg("base_position") = std::array<float, 3>{0.f, 0.f, 0.f},
                     py::arg("default_density") = 1000.f, py::arg("stiffness") = 0.f, py::arg("damping") = 0.f,
                     py::arg("max_force") = 1e6f, py::arg("self_collision") = false,
                     py::arg("solver_position_iterations") = 12, py::arg("render_visuals") = true,
                     // No keep_alive: the result is a tuple (can't be a weakref nurse). The returned
                     // articulation holds a PhysxWorld& — the caller must keep the world alive (urdf.py does).
                     "Import a URDF/xacro as a finalized Articulation (one shared parser with the C++ "
                     "URDFLoader — xacro supported). Returns (articulation, meshes, joint_names): the "
                     "collider meshes are bound to the sim (add them to a scene to render), joint_names "
                     "lists the actuated joints in drive-target order. Collision is primitive/bbox, mass "
                     "from <inertial> (else default_density x volume); fixed joints are collapsed. "
                     "stiffness/damping/max_force set a PD drive on every joint.");

        // GPU-resident batched articulation state I/O (the direct-GPU API). Build one
        // over many identical finalized articulations in a PhysxWorld(direct_gpu=True),
        // then drive the whole swarm with read_*/write_* + step — no CPU readback.
        using Read = threepp::PhysxGpuBatch::Read;
        using Write = threepp::PhysxGpuBatch::Write;
        auto reshape = [](std::vector<float>&& flat, py::ssize_t n, py::ssize_t block) {
            py::array_t<float> a({n, block});
            std::memcpy(a.mutable_data(), flat.data(), flat.size() * sizeof(float));
            return a;
        };
        // Validate a torch CUDA tensor for the zero-copy path and return its device pointer.
        // Without this the boundary would take a bare int and a wrong shape / dtype / device /
        // non-contiguous / freed tensor would silently corrupt GPU memory instead of raising.
        auto cudaPtr = [](const py::object& t, std::int64_t expectFloats, const char* what) -> CUdeviceptr {
            if (!t.attr("is_cuda").cast<bool>())
                throw std::runtime_error(std::string(what) + ": expected a CUDA tensor");
            if (!t.attr("is_contiguous")().cast<bool>())
                throw std::runtime_error(std::string(what) + ": tensor must be contiguous");
            if (!t.attr("is_floating_point")().cast<bool>() || t.attr("element_size")().cast<int>() != 4)
                throw std::runtime_error(std::string(what) + ": tensor must be float32");
            const auto nfl = t.attr("numel")().cast<std::int64_t>();
            if (nfl != expectFloats)
                throw std::runtime_error(std::string(what) + ": tensor has " + std::to_string(nfl) +
                                         " floats, expected " + std::to_string(expectFloats));
            return static_cast<CUdeviceptr>(t.attr("data_ptr")().cast<std::uintptr_t>());
        };
        // Validate a 32-bit (int32/uint32) CUDA index tensor; returns its ptr and sets outN.
        auto cudaIdx = [](const py::object& t, std::int64_t& outN, const char* what) -> CUdeviceptr {
            if (!t.attr("is_cuda").cast<bool>())
                throw std::runtime_error(std::string(what) + ": index tensor must be CUDA");
            if (!t.attr("is_contiguous")().cast<bool>())
                throw std::runtime_error(std::string(what) + ": index tensor must be contiguous");
            if (t.attr("element_size")().cast<int>() != 4)
                throw std::runtime_error(std::string(what) + ": index tensor must be 32-bit (int32/uint32)");
            outN = t.attr("numel")().cast<std::int64_t>();
            return static_cast<CUdeviceptr>(t.attr("data_ptr")().cast<std::uintptr_t>());
        };
        py::class_<threepp::PhysxGpuBatch>(m, "PhysxGpuBatch",
                "Batched GPU-resident state I/O over many reduced-coordinate articulations in one "
                "direct-GPU scene. The read_*/write_* methods take a torch CUDA tensor (validated for "
                "cuda/float32/contiguous/correct-size) and move ALL robots' state in one call with no "
                "CPU readback; *_host variants stage through numpy for debugging. All articulations in a "
                "batch must share a DOF count. Requires PhysxWorld(direct_gpu=True) and finalized articulations.")
                .def(py::init([](PhysxWorld& world, const py::iterable& arts) {
                         std::vector<PxArticulationReducedCoordinate*> raw;
                         for (auto h : arts) {
                             auto* a = h.cast<Articulation*>();
                             if (!a->finalized()) throw std::runtime_error("PhysxGpuBatch: articulation not finalized()");
                             raw.push_back(a->rawArt());
                         }
                         return std::make_unique<threepp::PhysxGpuBatch>(world, std::move(raw));
                     }),
                     py::arg("world"), py::arg("articulations"), py::keep_alive<1, 2>(),
                     "world must be created with direct_gpu=True and outlive this batch.")
                .def_property_readonly("count", [](threepp::PhysxGpuBatch& b) { return b.count(); })
                .def_property_readonly("max_dofs", [](threepp::PhysxGpuBatch& b) { return b.maxDofs(); })
                .def_property_readonly("max_links", [](threepp::PhysxGpuBatch& b) { return b.maxLinks(); })
                .def("step", &threepp::PhysxGpuBatch::step, py::arg("dt"),
                     "Advance every articulation one substep on the GPU (no binding sync).")
                // --- zero-copy path: pass the torch CUDA tensor (validated: cuda/float32/
                //     contiguous/correct-numel) — NOT a raw .data_ptr() ---
                .def("read_joint_pos", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.read(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "read_joint_pos"), Read::eJOINT_POSITION); },
                     py::arg("tensor"), "Fill the [n, max_dofs] float32 cuda tensor with joint positions.")
                .def("read_joint_vel", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.read(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "read_joint_vel"), Read::eJOINT_VELOCITY); },
                     py::arg("tensor"), "Fill the [n, max_dofs] float32 cuda tensor with joint velocities.")
                .def("read_root_pose", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.read(cudaPtr(t, std::int64_t(b.count()) * 7, "read_root_pose"), Read::eROOT_GLOBAL_POSE); },
                     py::arg("tensor"), "Fill the [n, 7] float32 cuda tensor with root pose [qx,qy,qz,qw,px,py,pz].")
                .def("read_root_linvel", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.read(cudaPtr(t, std::int64_t(b.count()) * 3, "read_root_linvel"), Read::eROOT_LINEAR_VELOCITY); },
                     py::arg("tensor"), "Fill the [n, 3] float32 cuda tensor with root linear velocity.")
                .def("read_root_angvel", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.read(cudaPtr(t, std::int64_t(b.count()) * 3, "read_root_angvel"), Read::eROOT_ANGULAR_VELOCITY); },
                     py::arg("tensor"), "Fill the [n, 3] float32 cuda tensor with root angular velocity.")
                // --- per-link reads (link 0 = root, then links in add_link order): foot
                //     kinematics for clearance/slip rewards. Buffer is [n, max_links * block]. ---
                .def("read_link_pose", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.read(cudaPtr(t, std::int64_t(b.count()) * b.maxLinks() * 7, "read_link_pose"), Read::eLINK_GLOBAL_POSE); },
                     py::arg("tensor"), "Fill the [n, max_links*7] float32 cuda tensor with per-link poses [qx,qy,qz,qw,px,py,pz].")
                .def("read_link_linvel", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.read(cudaPtr(t, std::int64_t(b.count()) * b.maxLinks() * 3, "read_link_linvel"), Read::eLINK_LINEAR_VELOCITY); },
                     py::arg("tensor"), "Fill the [n, max_links*3] float32 cuda tensor with per-link linear velocities (world frame).")
                .def("read_link_angvel", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.read(cudaPtr(t, std::int64_t(b.count()) * b.maxLinks() * 3, "read_link_angvel"), Read::eLINK_ANGULAR_VELOCITY); },
                     py::arg("tensor"), "Fill the [n, max_links*3] float32 cuda tensor with per-link angular velocities (world frame).")
                .def("write_joint_target_pos", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "write_joint_target_pos"), Write::eJOINT_TARGET_POSITION); },
                     py::arg("tensor"), "Set all joints' PD position targets from the [n, max_dofs] float32 cuda tensor.")
                .def("write_joint_force", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "write_joint_force"), Write::eJOINT_FORCE); },
                     py::arg("tensor"),
                     "Apply per-DOF joint forces/torques (effort control) from the [n, max_dofs] float32 cuda "
                     "tensor. Re-apply each step (forces don't persist). Use for force-controlled joints.")
                .def("write_joint_target_vel", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "write_joint_target_vel"), Write::eJOINT_TARGET_VELOCITY); },
                     py::arg("tensor"), "Set all joints' PD velocity targets from the [n, max_dofs] float32 cuda tensor.")
                .def("write_joint_pos", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "write_joint_pos"), Write::eJOINT_POSITION); },
                     py::arg("tensor"), "Overwrite ALL joints' positions from the [n, max_dofs] float32 cuda tensor (full-batch reset).")
                .def("write_joint_vel", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "write_joint_vel"), Write::eJOINT_VELOCITY); },
                     py::arg("tensor"), "Overwrite ALL joints' velocities from the [n, max_dofs] float32 cuda tensor (full-batch reset).")
                // --- subset reset (done envs only): nb is derived from the index tensor ---
                .def("write_subset_joint_pos", [cudaPtr, cudaIdx](threepp::PhysxGpuBatch& b, const py::object& src, const py::object& idx) {
                         std::int64_t nb = 0; auto ip = cudaIdx(idx, nb, "write_subset_joint_pos.indices");
                         b.writeSubset(cudaPtr(src, nb * b.maxDofs(), "write_subset_joint_pos.src"), ip, Write::eJOINT_POSITION, static_cast<::physx::PxU32>(nb)); },
                     py::arg("src"), py::arg("indices"))
                .def("write_subset_joint_vel", [cudaPtr, cudaIdx](threepp::PhysxGpuBatch& b, const py::object& src, const py::object& idx) {
                         std::int64_t nb = 0; auto ip = cudaIdx(idx, nb, "write_subset_joint_vel.indices");
                         b.writeSubset(cudaPtr(src, nb * b.maxDofs(), "write_subset_joint_vel.src"), ip, Write::eJOINT_VELOCITY, static_cast<::physx::PxU32>(nb)); },
                     py::arg("src"), py::arg("indices"))
                .def("write_subset_root_pose", [cudaPtr, cudaIdx](threepp::PhysxGpuBatch& b, const py::object& src, const py::object& idx) {
                         std::int64_t nb = 0; auto ip = cudaIdx(idx, nb, "write_subset_root_pose.indices");
                         b.writeSubset(cudaPtr(src, nb * 7, "write_subset_root_pose.src"), ip, Write::eROOT_GLOBAL_POSE, static_cast<::physx::PxU32>(nb)); },
                     py::arg("src"), py::arg("indices"))
                .def("write_subset_root_linvel", [cudaPtr, cudaIdx](threepp::PhysxGpuBatch& b, const py::object& src, const py::object& idx) {
                         std::int64_t nb = 0; auto ip = cudaIdx(idx, nb, "write_subset_root_linvel.indices");
                         b.writeSubset(cudaPtr(src, nb * 3, "write_subset_root_linvel.src"), ip, Write::eROOT_LINEAR_VELOCITY, static_cast<::physx::PxU32>(nb)); },
                     py::arg("src"), py::arg("indices"))
                .def("write_subset_root_angvel", [cudaPtr, cudaIdx](threepp::PhysxGpuBatch& b, const py::object& src, const py::object& idx) {
                         std::int64_t nb = 0; auto ip = cudaIdx(idx, nb, "write_subset_root_angvel.indices");
                         b.writeSubset(cudaPtr(src, nb * 3, "write_subset_root_angvel.src"), ip, Write::eROOT_ANGULAR_VELOCITY, static_cast<::physx::PxU32>(nb)); },
                     py::arg("src"), py::arg("indices"))
                .def("gpu_indices", [](threepp::PhysxGpuBatch& b) {
                         auto idx = b.gpuIndicesHost();
                         py::array_t<std::uint32_t> a(static_cast<py::ssize_t>(idx.size()));
                         std::memcpy(a.mutable_data(), idx.data(), idx.size() * sizeof(std::uint32_t));
                         return a; },
                     "The K articulation GPU indices as a uint32 numpy array (upload once to build "
                     "subset-index buffers for resets).")
                // --- host-staged debug readers (return numpy [n, block]) ---
                .def("read_joint_pos_host", [reshape](threepp::PhysxGpuBatch& b) {
                         return reshape(b.readHost(Read::eJOINT_POSITION), b.count(), b.maxDofs()); })
                .def("read_joint_vel_host", [reshape](threepp::PhysxGpuBatch& b) {
                         return reshape(b.readHost(Read::eJOINT_VELOCITY), b.count(), b.maxDofs()); })
                .def("read_root_pose_host", [reshape](threepp::PhysxGpuBatch& b) {
                         return reshape(b.readHost(Read::eROOT_GLOBAL_POSE), b.count(), 7); })
                .def("read_root_linvel_host", [reshape](threepp::PhysxGpuBatch& b) {
                         return reshape(b.readHost(Read::eROOT_LINEAR_VELOCITY), b.count(), 3); })
                .def("read_root_angvel_host", [reshape](threepp::PhysxGpuBatch& b) {
                         return reshape(b.readHost(Read::eROOT_ANGULAR_VELOCITY), b.count(), 3); })
                .def("read_link_pose_host", [reshape](threepp::PhysxGpuBatch& b) {
                         return reshape(b.readHost(Read::eLINK_GLOBAL_POSE), b.count(), b.maxLinks() * 7); })
                .def("read_link_linvel_host", [reshape](threepp::PhysxGpuBatch& b) {
                         return reshape(b.readHost(Read::eLINK_LINEAR_VELOCITY), b.count(), b.maxLinks() * 3); })
                .def("read_link_angvel_host", [reshape](threepp::PhysxGpuBatch& b) {
                         return reshape(b.readHost(Read::eLINK_ANGULAR_VELOCITY), b.count(), b.maxLinks() * 3); });

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
