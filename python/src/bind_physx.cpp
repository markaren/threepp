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
// bodies (CUDA path) came later; so did the PxVehicle2 direct-drive vehicle at
// the bottom of the file.
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
#include "threepp/extras/physx/ArticulationTendon.hpp"
#include "threepp/extras/physx/Joint.hpp"
#include "threepp/extras/physx/PhysxGpuBatch.hpp"
#include "threepp/extras/physx/TendonCable.hpp"
#include "threepp/extras/physx/PhysxSoftBody.hpp"
#include "threepp/extras/physx/PhysxVehicle.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/physx/UrdfArticulation.hpp"
#include "threepp/extras/sensors/Imu.hpp"
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

// Wheel index guard for the PhysxVehicle readouts. The C++ getters index a
// std::array<..., 4> with no bounds check, so an out-of-range index from Python
// would read past it instead of raising — check here, at the boundary.
inline int wheelIndex(int i) {
    if (i < 0 || i > 3) {
        throw std::out_of_range(
                "wheel index must be 0..3 (0=front-right, 1=front-left, 2=rear-right, 3=rear-left)");
    }
    return i;
}

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

        ::physx::PxRigidActor* raw() const { return actor_; }     // for PhysxWorld::removeActor
        void invalidate() { actor_ = nullptr; }                   // after world.remove: handle no longer usable

    private:
        ::physx::PxRigidDynamic* dyn() const {
            auto* d = actor_->is<::physx::PxRigidDynamic>();
            if (!d) throw std::runtime_error("RigidBody: this operation needs a dynamic body (this one is static)");
            return d;
        }
        ::physx::PxRigidActor* actor_;
    };


// Thin handle over a threepp::SoftBody (a PxDeformableVolume). The body is owned by
// the PhysxWorld's softBodies_ vector and dies with the world (or with
// remove_soft_body), so this only holds pointers — keep_alive on the adder ties the
// handle's lifetime to the world's, exactly as RigidBody does.
//
// The two readers below deliberately do NOT go through SoftBody's pinned
// positionsInvMass_ mirror: that buffer is refreshed inside world.step() and is
// private. A direct device->host copy off the volume's own position buffer is the
// same few kilobytes and is valid at any point between steps.
class PySoftBody {
public:
    PySoftBody(threepp::SoftBody* sb, ::physx::PxCudaContextManager* cuda) : sb_(sb), cuda_(cuda) {}

    threepp::SoftBody* raw() const { return sb_; }
    void invalidate() { sb_ = nullptr; }

    // Current COLLISION-mesh vertex positions, world space, as (N, 3) float32.
    py::array_t<float> simPositions() const {
        auto* vol = live()->actor();
        const auto n = static_cast<::physx::PxU32>(vol->getCollisionMesh()->getNbVertices());
        std::vector<::physx::PxVec4> host(n);
        {
            ::physx::PxScopedCudaLock _lock(*cuda_);
            cuda_->getCudaContext()->memcpyDtoH(
                    host.data(),
                    reinterpret_cast<CUdeviceptr>(vol->getPositionInvMassBufferD()),
                    static_cast<size_t>(n) * sizeof(::physx::PxVec4));
        }
        py::array_t<float> out({static_cast<py::ssize_t>(n), py::ssize_t(3)});
        float* dst = out.mutable_data();
        for (::physx::PxU32 i = 0; i < n; ++i) {
            dst[i * 3 + 0] = host[i].x;
            dst[i * 3 + 1] = host[i].y;
            dst[i * 3 + 2] = host[i].z;
        }
        return out;
    }

    // The cooked CONFORMING collision tet mesh: (rest vertices (V,3) float32,
    // tets (T,4) int32). Rest vertices are the cook's own local space — which is
    // the template mesh's local space, and differs from sim_positions() by the
    // spawn's rigid transform only (placeAndUpload applies scale 1). Every
    // rest-relative quality metric (volume ratio, edge stretch) is therefore valid
    // against sim_positions() as-is.
    py::tuple tetMesh() const {
        using namespace ::physx;
        auto* tm = live()->actor()->getCollisionMesh();
        const PxU32 nv = tm->getNbVertices();
        const PxU32 nt = tm->getNbTetrahedrons();
        const PxVec3* v = tm->getVertices();

        py::array_t<float> verts({static_cast<py::ssize_t>(nv), py::ssize_t(3)});
        float* vd = verts.mutable_data();
        for (PxU32 i = 0; i < nv; ++i) {
            vd[i * 3 + 0] = v[i].x;
            vd[i * 3 + 1] = v[i].y;
            vd[i * 3 + 2] = v[i].z;
        }

        py::array_t<std::int32_t> tets({static_cast<py::ssize_t>(nt), py::ssize_t(4)});
        std::int32_t* td = tets.mutable_data();
        const bool narrow = tm->getTetrahedronMeshFlags().isSet(PxTetrahedronMeshFlag::e16_BIT_INDICES);
        if (narrow) {
            const auto* src = static_cast<const PxU16*>(tm->getTetrahedrons());
            for (PxU32 i = 0; i < nt * 4; ++i) td[i] = static_cast<std::int32_t>(src[i]);
        } else {
            const auto* src = static_cast<const PxU32*>(tm->getTetrahedrons());
            for (PxU32 i = 0; i < nt * 4; ++i) td[i] = static_cast<std::int32_t>(src[i]);
        }
        return py::make_tuple(verts, tets);
    }

    int nbCollisionVertices() const {
        return static_cast<int>(live()->actor()->getCollisionMesh()->getNbVertices());
    }
    int nbTets() const {
        return static_cast<int>(live()->actor()->getCollisionMesh()->getNbTetrahedrons());
    }
    void setRecomputeNormals(bool v) { live()->setRecomputeNormals(v); }
    void enableGpuSkinning() { live()->enableGpuSkinning(); }

private:
    threepp::SoftBody* live() const {
        if (!sb_) throw std::runtime_error("SoftBody: handle was invalidated by remove_soft_body");
        return sb_;
    }
    threepp::SoftBody* sb_;
    ::physx::PxCudaContextManager* cuda_;
};

// Handle over a PxDeformableVolumeMaterial (owned by PxPhysics, released with the
// world). Separate from PhysxMaterial: PhysX keeps rigid and deformable materials
// in different type hierarchies and they are not interchangeable.
class PhysxSoftBodyMaterial {
public:
    explicit PhysxSoftBodyMaterial(::physx::PxDeformableVolumeMaterial* m) : mat_(m) {}
    ::physx::PxDeformableVolumeMaterial* raw() const { return mat_; }
    float youngs() const { return mat_->getYoungsModulus(); }
    void setYoungs(float v) { mat_->setYoungsModulus(v); }
    float poissons() const { return mat_->getPoissons(); }
    void setPoissons(float v) { mat_->setPoissons(v); }
    float dynamicFriction() const { return mat_->getDynamicFriction(); }
    void setDynamicFriction(float v) { mat_->setDynamicFriction(v); }
    float damping() const { return mat_->getDamping(); }
    void setDamping(float v) { mat_->setDamping(v); }

private:
    ::physx::PxDeformableVolumeMaterial* mat_;
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
                .def("add_torque", &ArticulationLink::addTorque, py::arg("torque"),
                     "Apply an external torque (N·m) about this link's centre of mass.")
                .def("add_force_at_pos", &ArticulationLink::addForceAtPos, py::arg("force"), py::arg("world_pos"),
                     "Apply an external force (N) at a WORLD point instead of at the centre of mass. add_force "
                     "alone acts through the COM and so produces zero torque about this link's own joint — it "
                     "cannot drive the articulation the way an offset load does. This adds the moment "
                     "(world_pos - centre_of_mass) × force, which is what a cable over a pulley, a fingertip "
                     "contact or a thruster on a boom actually applies. Note the arm is measured from the CENTRE "
                     "OF MASS, not the link origin or the joint anchor.")
                .def("world_point", &ArticulationLink::worldPoint, py::arg("local_offset"),
                     "World position of a point given in this link's ACTOR frame — the same frame a spatial "
                     "tendon attachment's relative_offset uses.")
                .def("add_impulse", &ArticulationLink::addImpulse, py::arg("impulse"),
                     "Apply an external impulse (kg·m/s) — e.g. a random shove. PhysX takes no impulse on an "
                     "articulation link, so this goes in as the force that carries the same momentum through one "
                     "substep (force = impulse / fixed_timestep), consumed by the next step().")
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
                .def("reset", &Articulation::reset, py::arg("position"), py::arg("quaternion") = Quaternion(),
                     "Episode reset: teleport the root to `position` with optional `quaternion` orientation "
                     "(default upright/identity), zero velocity, and zero all joint positions/velocities.")
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
                .def("link",
                     [](Articulation& a, const py::object& key) -> ArticulationLink {
                         if (py::isinstance<py::str>(key)) {
                             const auto name = key.cast<std::string>();
                             const auto* l = a.findLink(name);
                             if (!l) throw py::key_error("Articulation.link: no link named '" + name +
                                                         "' (see link_names)");
                             return *l;// a copy: the C++ pointer is only stable until the next add_link
                         }
                         const auto& ls = a.links();
                         const auto n = static_cast<std::ptrdiff_t>(ls.size());
                         auto i = key.cast<std::ptrdiff_t>();
                         if (i < 0) i += n;// Python-style negative indexing
                         if (i < 0 || i >= n) throw py::index_error("Articulation.link: index out of range");
                         return ls[static_cast<std::size_t>(i)];
                     },
                     py::arg("key"), py::keep_alive<0, 1>(),
                     "The ArticulationLink at add-order index `key` (int, 0 = root, negatives count "
                     "from the end) or registered under name `key` (str). For a URDF-loaded "
                     "articulation every URDF link name resolves — a link attached by a FIXED joint "
                     "maps to the link it was welded into, so a tool frame like a hand TCP still "
                     "finds its body. Use it to add_force/add_impulse on one specific link (e.g. "
                     "load the tool link with a catch impulse for two-way coupling) or to read its "
                     "pose. The handle keeps the articulation alive; the world must outlive both.")
                .def_property_readonly("links",
                     [](const Articulation& a) { return a.links(); },
                     // No keep_alive: a Python list can't be a weakref nurse (same rule as
                     // add_static_trimesh_tree) — hold the articulation/world yourself.
                     "Every link as a list of ArticulationLink, in add_link order (root first). "
                     "Handles are valid only while the articulation and its world live — hold "
                     "both; prefer link(key) for a single handle, which keeps the articulation "
                     "alive itself.")
                .def_property_readonly("link_names", &Articulation::linkNames,
                     "All names link(name) resolves, in registration order. Populated by "
                     "load_articulation with every URDF link name (root and fixed-collapsed "
                     "children included); empty for a hand-built articulation.")
                .def("dof_order",
                     [](const Articulation& a) {
                         const auto v = a.dofOrder();
                         return py::array_t<int>(static_cast<py::ssize_t>(v.size()), v.data());
                     },
                     "Per add-order joint, its low-level DOF slot in the direct-GPU joint buffers "
                     "(PhysX cache order != add-order). Use to map a GPU-trained policy back to the "
                     "CPU getters: obs_gpu[dof_order[i]] = cpu[i]; cpu_target[i] = gpu_target[dof_order[i]].");

        // --- Tendons -----------------------------------------------------------
        // Both tendon types are owned by their articulation ("When an articulation is
        // released, its attached tendons are automatically released"), so every handle
        // here is non-owning and every constructor carries keep_alive<1,2> to stop the
        // articulation being collected out from under it.
        py::class_<SpatialAttachment>(m, "SpatialAttachment",
                                      "One attachment point of a SpatialTendon, pinned at an offset in a link's "
                                      "actor frame. rest_length and the limits are LEAF-ONLY — PhysX silently "
                                      "ignores them on an interior attachment, so setting one here raises "
                                      "instead of leaving a mis-built routing looking configured.")
                .def_property_readonly("is_leaf", &SpatialAttachment::isLeaf)
                .def_property("coefficient", &SpatialAttachment::coefficient, &SpatialAttachment::setCoefficient,
                              "Scale on this segment's contribution to the accumulated tendon length.")
                .def_property("relative_offset", &SpatialAttachment::relativeOffset, &SpatialAttachment::setRelativeOffset,
                              "The attachment point in its link's ACTOR frame.")
                .def_property("rest_length", &SpatialAttachment::restLength, &SpatialAttachment::setRestLength,
                              "Rest length of the bilateral spring for the sub-tendon ending here (leaf only).")
                .def("set_taut_length", &SpatialAttachment::setTautLength, py::arg("length"),
                     "Configure this leaf as a PULL-ONLY cable of the given taut length: low limit -inf (a cable "
                     "never pushes), high limit `length`. Force appears only once the tendon is longer than this, "
                     "and is exactly zero when slack. Pair with stiffness=0 and limit_stiffness=k on the tendon.")
                .def("set_limits", &SpatialAttachment::setLimits, py::arg("low"), py::arg("high"))
                .def_property_readonly("limits", &SpatialAttachment::limits);

        py::class_<SpatialTendon>(m, "SpatialTendon",
                                  "A geometric tendon: a tree of attachment points whose coefficient-weighted "
                                  "segment lengths sum to the tendon length. MUST be built before "
                                  "Articulation.finalize() — PhysX forbids creating one on a scene-resident "
                                  "articulation.\n\n"
                                  "CAVEAT: a sub-tendon applies force only at its leaf and root links, along "
                                  "the root-to-leaf CHORD; interior attachments set the length but exert no "
                                  "force. So a single multi-attachment tendon does NOT reproduce the transverse "
                                  "pulley reaction a real routed cable applies, and its moment arms distal of "
                                  "the first joint are fabricated. For mechanically real routing use one "
                                  "two-attachment tendon per segment, where the chord IS the segment.")
                .def(py::init<Articulation&>(), py::arg("articulation"), py::keep_alive<1, 2>())
                .def("add_attachment",
                     [](SpatialTendon& t, const ArticulationLink& link, const Vector3& local_offset,
                        const py::object& parent, float coefficient) {
                         const SpatialAttachment* p = parent.is_none() ? nullptr : parent.cast<const SpatialAttachment*>();
                         return t.addAttachment(p, link, local_offset, coefficient);
                     },
                     py::arg("link"), py::arg("local_offset"), py::arg("parent") = py::none(),
                     py::arg("coefficient") = 1.f,
                     "Attach to `link` at `local_offset` in its ACTOR frame. parent=None makes it the root.")
                .def_property("stiffness", &SpatialTendon::stiffness, &SpatialTendon::setStiffness,
                              "Bilateral spring on the length. Leave 0 for a cable: a nonzero stiffness makes "
                              "the tendon PUSH when it is shorter than its rest length.")
                .def_property("damping", &SpatialTendon::damping, &SpatialTendon::setDamping,
                              "Documented as acting on both the spring and the limits, so it is not known to be "
                              "one-sided. Leave 0 for a cable and damp at the joint instead.")
                .def_property("limit_stiffness", &SpatialTendon::limitStiffness, &SpatialTendon::setLimitStiffness,
                              "Axial stiffness of the cable in the pull-only construction.")
                .def_property("offset", &SpatialTendon::offset,
                              [](SpatialTendon& t, float o) { t.setOffset(o); },
                              "The actuator: added to the accumulated length, so raising it makes the tendon act "
                              "shorter, i.e. pull.")
                .def_property_readonly("num_attachments", &SpatialTendon::numAttachments);

        py::class_<FixedTendon::TendonJoint>(m, "TendonJoint", "One joint DOF bound into a FixedTendon.")
                .def("set_coefficient",
                     [](FixedTendon::TendonJoint& j, float c, float recip) {
                         j.setCoefficient(::physx::PxArticulationAxis::eTWIST, c, recip);
                     },
                     py::arg("coefficient"), py::arg("recip_coefficient"),
                     "Re-scale this DOF's contribution. Whether this takes effect on a scene-resident "
                     "articulation is undocumented — verify with a measured torque change, not a return code.");

        py::class_<FixedTendon>(m, "FixedTendon",
                                "A joint-space tendon: length is the linear combination sum(c_i * q_i) of the "
                                "joint positions it spans, so a spring on that length couples those joints. No "
                                "geometry, so its 'moment arms' are the coefficients — prescribed rather than "
                                "emergent, which is the trade against SpatialTendon. The joints must be directly "
                                "connected in the articulation. Build before Articulation.finalize().")
                .def(py::init<Articulation&>(), py::arg("articulation"), py::keep_alive<1, 2>())
                .def("add_joint",
                     [](FixedTendon& t, const ArticulationLink& link, float coefficient,
                        const py::object& recip_coefficient, const py::object& parent) {
                         const FixedTendon::TendonJoint* p =
                                 parent.is_none() ? nullptr : parent.cast<const FixedTendon::TendonJoint*>();
                         const float recip = recip_coefficient.is_none() ? coefficient
                                                                        : recip_coefficient.cast<float>();
                         return t.addJoint(p, link, coefficient, recip);
                     },
                     py::arg("link"), py::arg("coefficient"), py::arg("recip_coefficient") = py::none(),
                     py::arg("parent") = py::none(),
                     "Bind `link`'s INBOUND joint into the tendon. `coefficient` is c_i in L = sum(c_i q_i) — "
                     "dimensionally a moment arm for a revolute DOF. `recip_coefficient` scales the response "
                     "applied back to this DOF; the SDK calls 1/coefficient 'commonly expected', but power "
                     "balance (F·L̇ = sum(tau_i·q̇_i) with L = sum(c_i q_i)) gives tau_i = F·c_i, so the "
                     "energetically consistent multiplier is c_i. Defaults to `coefficient` for that reason.")
                .def_property("stiffness", &FixedTendon::stiffness, &FixedTendon::setStiffness)
                .def_property("damping", &FixedTendon::damping, &FixedTendon::setDamping)
                .def_property("limit_stiffness", &FixedTendon::limitStiffness, &FixedTendon::setLimitStiffness)
                .def_property("rest_length", &FixedTendon::restLength, &FixedTendon::setRestLength)
                .def_property("offset", &FixedTendon::offset, [](FixedTendon& t, float o) { t.setOffset(o); })
                .def_property_readonly("limits", &FixedTendon::limits)
                .def("set_limits", &FixedTendon::setLimits, py::arg("low"), py::arg("high"))
                .def("open_limits", &FixedTendon::openLimits,
                     "Set limits wide enough never to clamp. The SDK's DEFAULT limit parameters are "
                     "(+FLT_MAX, -FLT_MAX) — the header itself calls that 'an invalid configuration that can "
                     "only work if stiffness is zero'. A spring-driven fixed tendon left at the default is a "
                     "silent no-op: it produces no force at all, and nothing warns.")
                .def_property_readonly("num_joints", &FixedTendon::numJoints);

        py::class_<TendonCable> cable(m, "TendonCable",
                                      "A tendon that behaves like a CABLE: routed over via points, pull-only, "
                                      "with a real tension number and optional routing friction.\n\n"
                                      "Neither PhysX tendon is a routed cable, measured on a two-link finger "
                                      "(python/examples/tendon_probe.py): a spatial tendon's interior "
                                      "attachments set the length but exert no force, so its generalized force "
                                      "matches the gradient taken with the via point FROZEN to 0.07 deg and "
                                      "sits 21.06 deg from a real cable; a fixed tendon has no geometry at "
                                      "all. This applies the true frictionless-pulley force at every via "
                                      "point, so torque = -T dL/dq exactly -- measured at 0.02% of magnitude "
                                      "and 0.001 deg of direction against the analytic gradient.\n\n"
                                      "Runs on the CPU physics path only: PhysX rejects link forces under "
                                      "direct-GPU, so batched GPU RL would need the SDK tendons instead.");
        py::enum_<TendonCable::Mode>(cable, "Mode")
                .value("TENSION", TendonCable::Mode::Tension,
                       "The command IS the cable tension (N) - an ideal motor with a torque loop closed "
                       "around it. Cannot go unstable: there is no stiffness to explode.")
                .value("LENGTH", TendonCable::Mode::Length,
                       "The command is the SPOOLED length (m); tension follows from how far the route is "
                       "stretched past it, T = k*(L - L_cmd) + c*Ldot, floored at zero. A real "
                       "series-elastic drivetrain - cable stretch is a property of actual tendons.");
        cable.def(py::init<PhysxWorld&, TendonCable::Mode>(), py::arg("world"),
                  py::arg("mode") = TendonCable::Mode::Tension, py::keep_alive<1, 2>())
                .def("add_via_point", &TendonCable::addViaPoint, py::arg("link"), py::arg("local_offset"),
                     "Add a via point on `link` at `local_offset` in the link's ACTOR frame, in order from "
                     "the actuator end to the insertion. First and last are the anchor and the insertion; "
                     "everything between is a pulley.")
                .def("set_tension", &TendonCable::setTension, py::arg("tension"),
                     "TENSION mode: cable tension in newtons. Negative is clamped to zero rather than "
                     "rejected - a cable asked to push simply goes slack.")
                .def("set_spool_length", &TendonCable::setSpoolLength, py::arg("length"),
                     "LENGTH mode: the spooled length in metres. Pull the cable by REDUCING it.")
                .def("set_stiffness", &TendonCable::setStiffness, py::arg("k"))
                .def("set_damping", &TendonCable::setDamping, py::arg("c"))
                .def("set_friction", &TendonCable::setFriction, py::arg("mu"),
                     "Capstan friction at the pulleys. A cable wrapping through angle theta comes out "
                     "carrying T*exp(-mu*theta) - which is exactly why tendon hands are hard to control "
                     "precisely: the tension reaching the fingertip is not the tension the motor applied, "
                     "and the shortfall depends on posture. 0 (default) is the ideal cable; sheathed "
                     "tendons are typically 0.1-0.4.")
                .def_property_readonly("length", &TendonCable::length,
                                       "Total routed length (m), from the live link poses.")
                .def_property_readonly("tension", &TendonCable::tension,
                                       "Tension at the ACTUATOR end (N) - the value actually applied last "
                                       "substep, not a reconstruction. PhysX exposes no tendon force readback "
                                       "at all, so with an SDK tendon this could only ever be a prediction.")
                .def_property_readonly("tip_tension", &TendonCable::tipTension,
                                       "Tension at the INSERTION end (N). Equals tension when friction is 0; "
                                       "the ratio is what the routing swallowed.")
                .def_property_readonly("num_nodes", &TendonCable::numNodes)
                .def_property_readonly("path",
                     [](const TendonCable& c) {
                         const auto pts = c.path();
                         py::array_t<float> out({static_cast<py::ssize_t>(pts.size()), py::ssize_t(3)});
                         auto m = out.mutable_unchecked<2>();
                         for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(pts.size()); ++i) {
                             m(i, 0) = pts[i].x; m(i, 1) = pts[i].y; m(i, 2) = pts[i].z;
                         }
                         return out;
                     },
                     "The resolved cable path in world space as an (N, 3) array, actuator end "
                     "first, wrap arcs included. Draw it: a cable on the wrong side of a joint, "
                     "or cutting a corner it should wrap, is obvious on sight and nearly "
                     "invisible in a column of moment arms.")
                .def_property_readonly("num_path_points", &TendonCable::numPathPoints,
                     "Points in the RESOLVED path, wrap arcs included, so it grows as a joint flexes.")
                .def("add_wrap", &TendonCable::addWrap, py::arg("link"), py::arg("local_centre"),
                     py::arg("local_axis"), py::arg("radius"), py::arg("side_hint"), py::arg("sheathed") = false,
                     "Between the previous and next via point, run the cable around a cylinder of "
                     "`radius` centred at `local_centre` on `link` with its axis along `local_axis` "
                     "(both in the link ACTOR frame). Put the cylinder on a joint, co-axial with the "
                     "hinge, and the radius IS the tendon standoff. A via point is welded to its link, "
                     "so a cable strung across a flexing joint chords over it and the moment REVERSES "
                     "once the distal point swings behind the axis (measured on this hand: an FDP MCP "
                     "arm running +9.98 mm extended to -5.03 mm at 89 deg). A real tendon slides along "
                     "the pulley instead; the wrap models that, and the arm then holds at the radius "
                     "through the whole range.");

        // One maximal-coordinate constraint between two RigidBodies (or one
        // body and the world). The joint MUST die before its world (its
        // destructor releases the PxJoint), which the keep_alive on the
        // constructor guarantees for the GC path.
        py::class_<Joint> joint(m, "Joint",
                                "A joint between two rigid bodies, or one body and the world "
                                "(pass None for that side). fixed/revolute/prismatic/spherical "
                                "ride one configured PxD6Joint; distance is a tether. The frame "
                                "is world-space: anchor at `position`, hinge/slide axis along "
                                "the frame's local X (`rotation` aims it). Valid only while its "
                                "world lives.");

        py::enum_<Joint::Type>(joint, "Type")
                .value("FIXED", Joint::Type::Fixed)
                .value("REVOLUTE", Joint::Type::Revolute)
                .value("PRISMATIC", Joint::Type::Prismatic)
                .value("SPHERICAL", Joint::Type::Spherical)
                .value("DISTANCE", Joint::Type::Distance);

        py::class_<Joint::Params>(joint, "Params",
                                  "Everything a joint is configured with. Angles in radians, "
                                  "lengths in metres. The drive is force-mode PD: `target` acts "
                                  "through stiffness, `velocity` through damping. break_force / "
                                  "break_torque of 0 = unbreakable.")
                .def(py::init<>())
                .def_readwrite("type", &Joint::Params::type)
                .def_readwrite("limited", &Joint::Params::limited)
                .def_readwrite("lower", &Joint::Params::lower)
                .def_readwrite("upper", &Joint::Params::upper)
                .def_readwrite("cone_y", &Joint::Params::coneY)
                .def_readwrite("cone_z", &Joint::Params::coneZ)
                .def_readwrite("stiffness", &Joint::Params::stiffness)
                .def_readwrite("damping", &Joint::Params::damping)
                .def_readwrite("max_force", &Joint::Params::maxForce)
                .def_readwrite("target", &Joint::Params::target)
                .def_readwrite("velocity", &Joint::Params::velocity)
                .def_readwrite("break_force", &Joint::Params::breakForce)
                .def_readwrite("break_torque", &Joint::Params::breakTorque)
                .def_readwrite("collide", &Joint::Params::collide);

        joint.def(py::init([](PhysxWorld& world, const py::object& bodyA, const py::object& bodyB,
                              const Vector3& position, const Quaternion& rotation,
                              const Joint::Params& params) {
                      // RigidBody keeps its actor private; unwrapping happens
                      // here, C++-side, exactly as PhysxWorld.remove does.
                      auto* a = bodyA.is_none() ? nullptr : bodyA.cast<RigidBody*>()->raw();
                      auto* b = bodyB.is_none() ? nullptr : bodyB.cast<RigidBody*>()->raw();
                      return std::make_unique<Joint>(world, a, b,
                                                     toPxTransform(position, rotation), params);
                  }),
                  py::arg("world"), py::arg("body_a"), py::arg("body_b"),
                  py::arg("position"), py::arg("rotation") = Quaternion(),
                  py::arg("params") = Joint::Params{},
                  py::keep_alive<1, 2>(),// the world outlives the joint
                  "Create a joint in `world` between body_a and body_b (either may be None, "
                  "meaning the world itself — not both). Default params are a fixed weld.")
                .def_property_readonly("type", &Joint::type)
                .def_property_readonly("position", &Joint::position,
                                       "The joint coordinate: radians (revolute / a spherical's "
                                       "twist), metres (prismatic), anchor distance (distance).")
                .def_property_readonly("velocity", &Joint::velocity,
                                       "Its rate: rad/s or m/s, same convention as position.")
                .def_property_readonly("broken", &Joint::broken,
                                       "True once the solver exceeded the break threshold; the "
                                       "constraint never comes back.")
                .def("set_drive_target", &Joint::setDriveTarget, py::arg("value"),
                     "PD setpoint along the motion axis (radians / metres). Acts through "
                     "stiffness — inert at zero stiffness.")
                .def("set_drive_velocity", &Joint::setDriveVelocity, py::arg("value"),
                     "Velocity setpoint (rad/s or m/s). Acts through damping.")
                .def(
                        "reaction",
                        [](const Joint& j) {
                            Vector3 force, torque;
                            j.reactionForce(force, torque);
                            return py::make_tuple(force, torque);
                        },
                        "(force N, torque N*m) the solver applied to hold the constraint on "
                        "the last step, world axes. Zero once broken - the failure load is "
                        "break_wrench().")
                .def(
                        "break_wrench",
                        [](const Joint& j) {
                            Vector3 force, torque;
                            j.breakWrench(force, torque);
                            return py::make_tuple(force, torque);
                        },
                        "(force N, torque N*m) the solver applied on the step that BROKE the "
                        "joint - the true failure load, past the break threshold. Zero until "
                        "broken.");

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

        py::class_<PhysxSoftBodyMaterial>(m, "PhysxSoftBodyMaterial",
                                          "A deformable-volume material: Young's modulus (Pa), Poisson's ratio and "
                                          "surface friction. Create via world.create_soft_body_material(...) and pass "
                                          "the SAME handle to every add_soft_body that shares it — PhysX keeps one "
                                          "PxMaterial per call otherwise. Not interchangeable with PhysxMaterial "
                                          "(rigid bodies use a different PhysX type).")
                .def_property("young", &PhysxSoftBodyMaterial::youngs, &PhysxSoftBodyMaterial::setYoungs)
                .def_property("poisson", &PhysxSoftBodyMaterial::poissons, &PhysxSoftBodyMaterial::setPoissons)
                .def_property("dynamic_friction", &PhysxSoftBodyMaterial::dynamicFriction,
                              &PhysxSoftBodyMaterial::setDynamicFriction)
                .def_property("damping", &PhysxSoftBodyMaterial::damping, &PhysxSoftBodyMaterial::setDamping);

        py::class_<PySoftBody>(m, "SoftBody",
                               "Handle to a PhysX deformable volume created via world.add_soft_body. Valid only "
                               "while its world lives (and until remove_soft_body). The simulation runs on two "
                               "meshes: a CONFORMING collision tet mesh (what tet_mesh/sim_positions report, and "
                               "what contact is resolved against) and a voxelised simulation mesh the solver "
                               "integrates; voxel_resolution sizes the latter.")
                .def("sim_positions", &PySoftBody::simPositions,
                     "Current collision-mesh vertex positions as an (N, 3) float32 array, world space. One "
                     "device->host copy per call — read it once per frame, not once per fish per query.")
                .def("tet_mesh", &PySoftBody::tetMesh,
                     "((V, 3) float32 rest vertices, (T, 4) int32 tets) of the cooked CONFORMING collision "
                     "mesh. Rest vertices are in the template mesh's own local space, so they differ from "
                     "sim_positions() by the spawn transform only — rest-relative metrics (volume ratio, "
                     "edge stretch) compare directly. Feed the pair to another solver to run PhysX's "
                     "tetrahedralisation elsewhere.")
                .def_property_readonly("num_vertices", &PySoftBody::nbCollisionVertices,
                                       "Collision-mesh vertex count (the length of sim_positions()).")
                .def_property_readonly("num_tets", &PySoftBody::nbTets, "Collision-mesh tetrahedron count.")
                .def("set_recompute_normals", &PySoftBody::setRecomputeNormals, py::arg("enabled"),
                     "Recompute the visual geometry's vertex normals each step (default on). Turn it off "
                     "when something else owns the normals — it is a full pass over the visual mesh.")
                .def("enable_gpu_skinning", &PySoftBody::enableGpuSkinning,
                     "Blend the visual mesh in the vertex shader from a small per-body tet texture instead of "
                     "CPU-skinning and re-uploading the full-resolution mesh every step. Call once, right "
                     "after add_soft_body. Also the cheap option when the visual mesh is NOT drawn at all "
                     "(an external skinner owns the render surface): it reduces the per-step cost to one "
                     "few-hundred-texel texture write.");

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
                // GIL released for the whole step: PhysX simulate+fetch plus the
                // transform sync are pure C++, and the substep callbacks below
                // re-acquire per call — so torch inference or a dataset writer
                // on another Python thread keeps running while physics steps.
                .def("step", &PhysxWorld::step, py::arg("dt"),
                     py::call_guard<py::gil_scoped_release>(),
                     "Advance the simulation by dt seconds (variable-rate caller, fixed-rate physics). "
                     "After it returns, every bound mesh's transform reflects the new state. "
                     "Releases the GIL while stepping.")
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
                .def("remove",
                     [](PhysxWorld& w, RigidBody& body) {
                         w.removeActor(body.raw());
                         body.invalidate();   // the handle is now dead — reusing it is undefined
                     },
                     py::arg("body"),
                     "Remove a body (from add / add_static / add_dynamic_convex / add_static_trimesh) from the "
                     "world and release it — e.g. to rebuild geometry without recreating the world. Any mesh "
                     "binding is dropped; the RigidBody handle is INVALID afterwards (don't reuse it).")
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
                .def("create_soft_body_material",
                     [](PhysxWorld& w, float young, float poisson, float friction, const py::object& damping) {
                         auto* m = w.createSoftBodyMaterial(young, poisson, friction);
                         if (!m) throw std::runtime_error("create_soft_body_material: createDeformableVolumeMaterial failed");
                         if (!damping.is_none()) m->setDamping(damping.cast<float>());
                         return std::make_unique<PhysxSoftBodyMaterial>(m);
                     },
                     py::arg("young") = 1e6f, py::arg("poisson") = 0.45f,
                     py::arg("friction") = 0.5f, py::arg("damping") = py::none(),
                     py::keep_alive<0, 1>(),
                     "Create a deformable-volume material (Young's modulus Pa, Poisson's ratio, surface "
                     "friction). Requires gpu_dynamics=True. Create ONE and share it across every "
                     "add_soft_body that uses the same flesh — each call allocates a PxMaterial that lives "
                     "until the world dies.")
                .def("add_soft_body",
                     [](PhysxWorld& w, Mesh& mesh, const py::object& material, int voxel_resolution,
                        unsigned solver_iterations, bool self_collision, const std::string& cache_key, float mass) {
                         ::physx::PxDeformableVolumeMaterial* mat =
                                 material.is_none() ? nullptr : material.cast<PhysxSoftBodyMaterial*>()->raw();
                         auto* sb = w.addSoftBody(mesh, mat, voxel_resolution, solver_iterations,
                                                  self_collision, cache_key, mass);
                         return PySoftBody(sb, w.cudaContextManager());
                     },
                     py::arg("mesh"), py::arg("material") = py::none(), py::arg("voxel_resolution") = 10,
                     py::arg("solver_iterations") = 20, py::arg("self_collision") = false,
                     py::arg("cache_key") = std::string(), py::arg("mass") = 0.f,
                     py::keep_alive<1, 2>(), py::keep_alive<0, 1>(),
                     "Cook `mesh` into a deformable volume and add it. Requires PhysxWorld(gpu_dynamics=True). "
                     "The mesh's world matrix is baked into the cooked geometry and its local transform reset, "
                     "so place the mesh first, then add it. voxel_resolution sets the SIMULATION mesh detail "
                     "(higher = finer and slower); the collision mesh is conforming and follows the surface. "
                     "cache_key reuses the (expensive) cook and per-vertex binding across every body built from "
                     "the same source geometry at the same voxel_resolution — pass the model's filename to pay "
                     "the cook once per species. mass in kg; 0 keeps the unit-density mass from the tet volume. "
                     "Returns a SoftBody handle; the world owns the body.")
                .def("remove_soft_body",
                     [](PhysxWorld& w, PySoftBody& body) {
                         w.removeSoftBody(body.raw());
                         body.invalidate();// the handle is now dead — reusing it raises
                     },
                     py::arg("body"),
                     "Destroy a soft body: releases the PhysX actor and its GPU/pinned buffers, and (for "
                     "bodies added from a Mesh) detaches that mesh from its parent. The SoftBody handle is "
                     "INVALID afterwards.")
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
                .def("add_static_heightfield",
                     [](PhysxWorld& w,
                        py::array_t<float, py::array::c_style | py::array::forcecast> heights,
                        float cell, const Vector3& origin, float thickness) {
                         if (heights.ndim() != 2) {
                             throw std::runtime_error(
                                     "add_static_heightfield: heights must be a 2-D (ny, nx) array, got " +
                                     std::to_string(heights.ndim()) + " dimensions");
                         }
                         const int ny = static_cast<int>(heights.shape(0));
                         const int nx = static_cast<int>(heights.shape(1));
                         if (nx < 2 || ny < 2) {
                             throw std::runtime_error(
                                     "add_static_heightfield: need at least 2x2 samples, got (" +
                                     std::to_string(ny) + ", " + std::to_string(nx) + ")");
                         }
                         auto* a = w.addStaticHeightField(heights.data(), nx, ny, cell,
                                                          origin, thickness, nullptr);
                         if (!a) {
                             throw std::runtime_error(
                                     "add_static_heightfield: degenerate field (cell and thickness must "
                                     "be > 0, and at least one sample must be finite)");
                         }
                         return RigidBody(a);
                     },
                     py::arg("heights"), py::arg("cell"), py::arg("origin"),
                     py::arg("thickness") = 0.5f, py::keep_alive<0, 1>(),
                     "Add a 2.5D height field as a static collider, in a Z-UP world.\n\n"
                     "`heights` is a 2-D float32/float64 array of shape (ny, nx): ROWS ARE Y. "
                     "heights[iy, ix] is the surface z at world "
                     "(origin.x + ix*cell, origin.y + iy*cell), with `cell` the sample spacing in "
                     "metres, the same in x and y. A (nx, ny) grid indexed the other way round must "
                     "be passed transposed. Heights are quantised to int16 against the field's own "
                     "z range (0.6 mm for a 20 m range), and a non-finite sample is pinned to the "
                     "field's floor rather than poisoning the whole grid.\n\n"
                     "Use this instead of add_static_trimesh wherever the collider is terrain: a "
                     "height field cannot represent a HOLE or a near-vertical SPIKE, which is what "
                     "a marching-cubes bake of a scan is full of, and it costs ~4 bytes a sample "
                     "against ~100 for the same surface as triangles.\n\n"
                     "`thickness` is how deep below the surface the field is meant to stay solid. "
                     "PhysX 5 has no such knob (PxHeightFieldDesc::thickness was a PhysX 3 field) "
                     "and a height field is a surface, not a volume: measured, a 4 cm ball at "
                     "6 m/s tunnels through it exactly as it tunnels through the same surface as a "
                     "trimesh. Nothing tunnels while the body's diameter exceeds its per-substep "
                     "travel. The argument is accepted and validated (> 0) but PhysX cannot honour "
                     "it.\n\n"
                     "Returns a RigidBody; the world owns the actor.")
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
                         return w.onPreSubstep([cb](float dt) { py::gil_scoped_acquire g; cb(dt); });
                     },
                     py::arg("callback"),
                     "Register callback(dt) fired before each fixed substep. Returns a "
                     "handle for remove_substep_callback().")
                .def("on_post_substep",
                     [](PhysxWorld& w, py::function cb) {
                         return w.onPostSubstep([cb](float dt) { py::gil_scoped_acquire g; cb(dt); });
                     },
                     py::arg("callback"),
                     "Register callback(dt) fired after each fixed substep. Returns a "
                     "handle for remove_substep_callback().")
                .def("remove_substep_callback", &PhysxWorld::removeSubstepCallback,
                     py::arg("handle"),
                     "Unregister a pre/post substep callback by its handle. A stale or "
                     "already-removed handle is a no-op.")
                .def("register_sensor",
                     [](PhysxWorld& w, Sensor& s) { w.registerSensor(&s); },
                     py::arg("sensor"), py::keep_alive<1, 2>(),
                     "Register a sensor (Imu, JointEncoder, ContactSensor, ...) to be sampled from the step loop once per "
                     "fixed substep, the instant body states are fresh. Call AFTER adding the body "
                     "the sensor is attached to; raises if the attachment has no managed rigid body. "
                     "The world keeps the sensor alive.")
                .def("unregister_sensor",
                     [](PhysxWorld& w, Sensor& s) { w.unregisterSensor(&s); },
                     py::arg("sensor"), "Stop sampling a previously registered sensor.")
                .def_property_readonly("sim_time", &PhysxWorld::simTime,
                                       "Accumulated fixed-substep simulation time (s) — the clock stamped "
                                       "onto sensor samples.")
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
                        int solver_position_iterations, bool render_visuals, float scale,
                        const std::map<std::string, std::string>& args) {
                         URDFArticulationOptions opts;
                         opts.args = args;
                         opts.fixedBase = fixed_base;
                         opts.basePosition = Vector3(base_position[0], base_position[1], base_position[2]);
                         opts.defaultDensity = default_density;
                         opts.stiffness = stiffness;
                         opts.damping = damping;
                         opts.maxForce = max_force;
                         opts.selfCollision = self_collision;
                         opts.solverPositionIterations = solver_position_iterations;
                         opts.renderVisuals = render_visuals;
                         opts.scale = scale;
                         auto r = loadArticulation(w, std::filesystem::path(path), opts);
                         if (!r.articulation) {
                             // r.error is the URDFLoader's own account, copied out
                             // before the loader inside loadArticulation died with
                             // it. Without it this said only "could not read",
                             // about a file the parser could describe in detail.
                             throw std::runtime_error("load_articulation: could not read '" + path +
                                                      "'" + (r.error.empty() ? "" : " - " + r.error));
                         }
                         return std::make_tuple(std::move(r.articulation), std::move(r.meshes), std::move(r.jointNames));
                     },
                     py::arg("path"), py::arg("fixed_base") = false,
                     py::arg("base_position") = std::array<float, 3>{0.f, 0.f, 0.f},
                     py::arg("default_density") = 1000.f, py::arg("stiffness") = 0.f, py::arg("damping") = 0.f,
                     py::arg("max_force") = 1e6f, py::arg("self_collision") = false,
                     py::arg("solver_position_iterations") = 12, py::arg("render_visuals") = true,
                     py::arg("scale") = 1.f,
                     py::arg("args") = std::map<std::string, std::string>{},
                     // No keep_alive: the result is a tuple (can't be a weakref nurse). The returned
                     // articulation holds a PhysxWorld& — the caller must keep the world alive (urdf.py does).
                     "Import a URDF/xacro as a finalized Articulation (one shared parser with the C++ "
                     "URDFLoader — xacro supported). Returns (articulation, meshes, joint_names): the "
                     "collider meshes are bound to the sim (add them to a scene to render), joint_names "
                     "lists the actuated joints in drive-target order. Per-link handles are on the "
                     "articulation itself: articulation.link('tool_link_name') resolves every URDF link "
                     "name to its ArticulationLink (for add_force on a tool link, per-link poses...). "
                     "Collision is primitive/bbox, mass "
                     "from <inertial> (else default_density x volume); fixed joints are collapsed. "
                     "stiffness/damping/max_force set a PD drive on every joint. scale reinterprets the "
                     "file's length units (a millimetre URDF in a metre world is 0.001) - shapes, joint "
                     "frames and prismatic limits are built scaled, masses stay as authored, and a "
                     "prismatic DOF then reads and drives in the SCALED units. `args` are xacro "
                     "argument overrides, the same name:=value pairs the xacro CLI takes - a "
                     "parameterised description built without them expands to the FILE's defaults, "
                     "which for many robots names config paths that do not exist.");

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
                     py::call_guard<py::gil_scoped_release>(),
                     "Advance every articulation one substep on the GPU (no binding sync). "
                     "Releases the GIL while stepping.")
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
                .def("write_link_force", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxLinks() * 3, "write_link_force"), Write::eLINK_FORCE); },
                     py::arg("tensor"),
                     "Apply an external force (N) to every link, from the [n, max_links, 3] float32 cuda "
                     "tensor, in WORLD coordinates at each link's centre of mass. This is the only way to "
                     "push a batched robot: ArticulationLink.add_force is a CPU-path call and PhysX rejects "
                     "it outright under direct-GPU. Forces are consumed by the next step and cleared, so "
                     "re-apply every substep you want them to act on — a random shove is one substep of "
                     "impulse/dt on the base link.")
                .def("write_link_torque", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxLinks() * 3, "write_link_torque"), Write::eLINK_TORQUE); },
                     py::arg("tensor"),
                     "Apply an external torque (N*m) to every link, from the [n, max_links, 3] float32 cuda "
                     "tensor, in WORLD coordinates. Cleared after each step, like write_link_force.")
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

        // ---- PxVehicle2 direct-drive vehicle ----
        //
        // The C++ PhysxVehicle, unchanged: PhysX owns the suspension, the tire
        // model, the sticky tires, the substepping and the load transfer. The
        // vehicle steps itself from a PhysxWorld substep callback, so the only
        // thing Python drives is commands in / telemetry out — plus the road
        // override, which lets a ground model Python owns (deformable terrain,
        // a terramechanics soil model) stand in for the scene query under a wheel.
        //
        // Constructor defaults are the Range Rover Evoque tuning from the C++
        // demo (examples/projects/Vehicle/main.cpp): 4WD, sticky tires, direct
        // drive. They are known-fun; change the dimensions to suit another body
        // and leave the dynamics alone unless you are ready to re-tune.
        //
        // Wheel indices everywhere: 0 = front-right, 1 = front-left,
        // 2 = rear-right, 3 = rear-left. Frame: +Z forward, +X right, +Y up.
        py::class_<PhysxVehicle> vehicle(
                m, "PhysxVehicle",
                "A drivable 4-wheel vehicle (PxVehicle2 direct drive) in a PhysxWorld. Feed it "
                "throttle/brake/steer each frame and copy its chassis + wheel poses onto your "
                "visuals; it advances itself inside world.step(dt). Valid only while its world "
                "lives. Wheel indices: 0=front-right, 1=front-left, 2=rear-right, 3=rear-left.");

        py::enum_<PhysxVehicle::Gear>(vehicle, "Gear")
                .value("REVERSE", PhysxVehicle::Gear::Reverse)
                .value("NEUTRAL", PhysxVehicle::Gear::Neutral)
                .value("FORWARD", PhysxVehicle::Gear::Forward);

        vehicle.def(py::init([](PhysxWorld& world,
                                float chassis_width, float chassis_height, float chassis_length,
                                float chassis_mass, float wheelbase, float track_width,
                                float wheel_radius, float wheel_half_width, float wheel_mass,
                                const std::array<bool, 4>& driven_wheels,
                                float max_throttle_torque, float max_brake_torque, float max_steer_angle,
                                float tire_friction, float longitudinal_stiffness, float lateral_stiffness,
                                float suspension_travel, float suspension_stiffness, float suspension_damping,
                                float suspension_attachment_y, float wheel_damping_rate,
                                const Vector3& position, const Quaternion& rotation) {
                        PhysxVehicle::Settings s;
                        s.chassisWidth = chassis_width;
                        s.chassisHeight = chassis_height;
                        s.chassisLength = chassis_length;
                        s.chassisMass = chassis_mass;
                        s.wheelbase = wheelbase;
                        s.trackWidth = track_width;
                        s.wheelRadius = wheel_radius;
                        s.wheelHalfWidth = wheel_half_width;
                        s.wheelMass = wheel_mass;
                        s.drivenWheels = driven_wheels;
                        s.maxThrottleTorque = max_throttle_torque;
                        s.maxBrakeTorque = max_brake_torque;
                        s.maxSteerAngleRad = max_steer_angle;
                        s.tireFriction = tire_friction;
                        s.longitudinalStiffness = longitudinal_stiffness;
                        s.lateralStiffness = lateral_stiffness;
                        s.suspensionTravelDist = suspension_travel;
                        s.suspensionStiffness = suspension_stiffness;
                        s.suspensionDamping = suspension_damping;
                        s.suspensionAttachmentY = suspension_attachment_y;
                        s.wheelDampingRate = wheel_damping_rate;
                        s.spawnPosition = position;
                        s.spawnRotation = rotation;
                        return std::make_unique<PhysxVehicle>(world, s);
                    }),
                    py::arg("world"),
                    py::arg("chassis_width") = 1.95f, py::arg("chassis_height") = 1.4f,
                    py::arg("chassis_length") = 4.4f, py::arg("chassis_mass") = 1500.f,
                    py::arg("wheelbase") = 2.66f, py::arg("track_width") = 1.65f,
                    py::arg("wheel_radius") = 0.4f, py::arg("wheel_half_width") = 0.15f,
                    py::arg("wheel_mass") = 25.f,
                    py::arg("driven_wheels") = std::array<bool, 4>{true, true, true, true},
                    py::arg("max_throttle_torque") = 1500.f, py::arg("max_brake_torque") = 5000.f,
                    py::arg("max_steer_angle") = 0.6f,
                    py::arg("tire_friction") = 2.f,
                    py::arg("longitudinal_stiffness") = 100'000.f, py::arg("lateral_stiffness") = 80'000.f,
                    py::arg("suspension_travel") = 0.3f, py::arg("suspension_stiffness") = 35'000.f,
                    py::arg("suspension_damping") = 4500.f, py::arg("suspension_attachment_y") = -0.4f,
                    py::arg("wheel_damping_rate") = 1.5f,
                    py::arg("position") = Vector3(0, 1.2f, 0), py::arg("rotation") = Quaternion(),
                    py::keep_alive<1, 2>(),// the world outlives the vehicle (it steps it)
                    "Spawn a vehicle in `world`. Defaults are the Range Rover Evoque tuning of the "
                    "C++ demo: 4WD direct drive, tire_friction 2.0 (dry asphalt). Dimensions are the "
                    "chassis box PhysX simulates — match them to whatever body you draw on top. "
                    "driven_wheels selects which wheels take throttle torque, in wheel-index order.")
                // -- Inputs --
                .def("set_throttle", &PhysxVehicle::setThrottle, py::arg("value"),
                     "Throttle, 0..1. Direct drive: torque straight to the driven wheels.")
                .def("set_brake", &PhysxVehicle::setBrake, py::arg("value"), "Brake, 0..1 (all four wheels).")
                .def("set_steer", &PhysxVehicle::setSteer, py::arg("value"),
                     "Steer, -1..1 (front wheels; 1 = max_steer_angle to the right).")
                .def_property("gear", &PhysxVehicle::gear, &PhysxVehicle::setGear,
                              "Gear.FORWARD / Gear.NEUTRAL / Gear.REVERSE. Direct drive has no gearbox — "
                              "this only picks the sign of the drive torque.")
                .def("respawn",
                     [](PhysxVehicle& v, const Vector3& position, const Quaternion& rotation) {
                         auto* actor = v.chassisActor();
                         actor->setGlobalPose(toPxTransform(position, rotation));
                         actor->setLinearVelocity(::physx::PxVec3(0.f));
                         actor->setAngularVelocity(::physx::PxVec3(0.f));
                         actor->wakeUp();
                     },
                     py::arg("position"), py::arg("rotation") = Quaternion(),
                     "Teleport the chassis and kill its velocities (the wheels keep spinning down "
                     "on their own). The suspension re-settles over the next few steps.")
                .def("add_force_at_pos",
                     [](PhysxVehicle& v, const Vector3& force, const Vector3& world_pos) {
                         ::physx::PxRigidBodyExt::addForceAtPos(
                                 *v.chassisActor(), toPxVec3(force), toPxVec3(world_pos),
                                 ::physx::PxForceMode::eFORCE);
                     },
                     py::arg("force"), py::arg("world_pos"),
                     "Apply a continuous force (N) to the chassis at a world-space point — the way to "
                     "add something PhysX's vehicle knows nothing about, e.g. the bulldozing drag of a "
                     "wheel ploughing through soil. Consumed by the next step().")
                // -- Chassis readouts --
                .def_property_readonly("position",
                                       [](const PhysxVehicle& v) { return fromPxVec3(v.chassisPose().p); },
                                       "Chassis center, world space. Copy onto your visual each frame.")
                .def_property_readonly("quaternion",
                                       [](const PhysxVehicle& v) { return fromPxQuat(v.chassisPose().q); })
                .def_property_readonly("forward_speed", &PhysxVehicle::forwardSpeed,
                                       "Speed along the chassis forward axis (m/s); negative in reverse.")
                // -- Per-wheel readouts --
                .def("wheel_local_pose",
                     [](const PhysxVehicle& v, int i) {
                         const auto p = v.wheelLocalPose(wheelIndex(i));
                         return py::make_tuple(fromPxVec3(p.p), fromPxQuat(p.q));
                     },
                     py::arg("wheel"),
                     "(position, quaternion) of the wheel in CHASSIS space — steer, suspension travel "
                     "and spin included. Put your wheel visual under the chassis group and assign both.")
                .def("wheel_angular_speed",
                     [](const PhysxVehicle& v, int i) { return v.wheelAngularSpeed(wheelIndex(i)); },
                     py::arg("wheel"), "Wheel spin rate (rad/s).")
                .def("wheel_rotation_angle",
                     [](const PhysxVehicle& v, int i) { return v.wheelRotationAngle(wheelIndex(i)); },
                     py::arg("wheel"), "Wheel spin angle (radians, wrapped to ±2π).")
                .def("tire_longitudinal_slip",
                     [](const PhysxVehicle& v, int i) { return v.tireLongitudinalSlip(wheelIndex(i)); },
                     py::arg("wheel"),
                     "Longitudinal slip ratio: 0 = pure rolling, +1 = the wheel spinning up under a "
                     "stationary car, -1 = locked. The wheelspin readout.")
                .def("tire_lateral_slip",
                     [](const PhysxVehicle& v, int i) { return v.tireLateralSlip(wheelIndex(i)); },
                     py::arg("wheel"), "Lateral slip (≈ tan of the slip angle; 0.1 ≈ 6° of drift).")
                .def("suspension_jounce",
                     [](const PhysxVehicle& v, int i) { return v.suspensionJounce(wheelIndex(i)); },
                     py::arg("wheel"),
                     "Suspension compression (m): 0 = full droop, suspension_travel = bottomed out.")
                .def("suspension_jounce_speed",
                     [](const PhysxVehicle& v, int i) { return v.suspensionJounceSpeed(wheelIndex(i)); },
                     py::arg("wheel"), "Compression rate (m/s) — spikes on landings and curb strikes.")
                .def("suspension_force",
                     [](const PhysxVehicle& v, int i) { return v.suspensionForce(wheelIndex(i)); },
                     py::arg("wheel"),
                     "Wheel load (N): the magnitude of the suspension force this wheel puts into the "
                     "chassis. On level ground the four sum to the chassis weight and redistribute "
                     "under braking/cornering — this is the W a soil model wants for sinkage and grip.")
                .def("wheel_grounded",
                     [](const PhysxVehicle& v, int i) { return v.wheelGrounded(wheelIndex(i)); },
                     py::arg("wheel"), "True while this wheel has ground within suspension reach.")
                // -- Road override --
                .def("set_road_override",
                     [](PhysxVehicle& v, int i, float height, float mu, float vRoad) {
                         v.setRoadOverride(wheelIndex(i), height, mu, vRoad);
                     },
                     py::arg("wheel"), py::arg("height"), py::arg("mu"), py::arg("v_road") = 0.f,
                     "Hand this wheel's suspension a road of your own instead of what the PhysX scene "
                     "query found: a horizontal plane at world y=`height` with friction `mu`. Set it "
                     "per wheel, per frame, from whatever ground model you own — e.g. terrain grade "
                     "minus the soil's equilibrium sinkage, so the wheel rides IN the ground by "
                     "exactly the sinkage the load dictates, with mu from Mohr-Coulomb rather than "
                     "the tire_friction ceiling. On a road PROFILE also pass `v_road`, the surface's "
                     "vertical velocity under the wheel (v * slope, m/s, +up) — it reaches the tire "
                     "slip terms. NOTE (probe-verified, PhysX 5.x): the suspension damper IGNORES it "
                     "and still measures against a static plane, so suspension_force() reads biased "
                     "low by damping*v_road on grades while the limit constraint quietly carries the "
                     "difference; see the header for why compensating externally makes it worse. "
                     "Everything else about the vehicle is untouched.")
                .def("clear_road_override",
                     [](PhysxVehicle& v, int i) { v.clearRoadOverride(wheelIndex(i)); },
                     py::arg("wheel"), "Give this wheel back to the scene query (the rigid fallback).")
                .def("road_override_active",
                     [](const PhysxVehicle& v, int i) { return v.roadOverrideActive(wheelIndex(i)); },
                     py::arg("wheel"));

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
