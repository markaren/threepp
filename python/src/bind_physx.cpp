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

#include <array>
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

    using namespace ::physx;

    // Shortest-arc quaternion mapping unit vector `from` onto unit vector `to`.
    // Used to orient a revolute joint frame so its X axis (the eTWIST axis) lies
    // along the requested world hinge axis.
    inline PxQuat shortestArc(PxVec3 from, PxVec3 to) {
        from.normalize();
        to.normalize();
        const float d = from.dot(to);
        if (d >= 1.f - 1e-6f) return PxQuat(PxIdentity);
        if (d <= -1.f + 1e-6f) {
            PxVec3 axis = PxVec3(1, 0, 0).cross(from);
            if (axis.magnitudeSquared() < 1e-6f) axis = PxVec3(0, 1, 0).cross(from);
            axis.normalize();
            return PxQuat(PxPi, axis);
        }
        const PxVec3 c = from.cross(to);
        PxQuat q(c.x, c.y, c.z, 1.f + d);
        q.normalize();
        return q;
    }

    // Box/Sphere/Capsule collider inferred from a mesh geometry (mirrors
    // PhysxWorld's private inferShape). localPose corrects the capsule axis
    // (threepp capsule is Y-aligned; PhysX capsule is X-aligned).
    struct LinkShape {
        PxGeometryHolder geom;
        PxTransform localPose{PxIdentity};
        bool valid = true;
    };
    inline LinkShape inferLinkShape(const BufferGeometry& g) {
        LinkShape s;
        if (auto* b = dynamic_cast<const BoxGeometry*>(&g)) {
            s.geom = PxBoxGeometry(b->width * 0.5f, b->height * 0.5f, b->depth * 0.5f);
        } else if (auto* sp = dynamic_cast<const SphereGeometry*>(&g)) {
            s.geom = PxSphereGeometry(sp->radius);
        } else if (auto* c = dynamic_cast<const CapsuleGeometry*>(&g)) {
            s.geom = PxCapsuleGeometry(c->radius, c->length * 0.5f);
            s.localPose = PxTransform(PxQuat(-PxHalfPi, PxVec3(0, 0, 1)));
        } else {
            s.valid = false;
        }
        return s;
    }

    // Handle to one articulation link + its inbound revolute joint (null for the
    // root). Valid only while its Articulation (and world) live.
    class ArticulationLink {
    public:
        ArticulationLink(PxArticulationLink* link, PxArticulationJointReducedCoordinate* joint, PxTransform creationPose)
            : link_(link), joint_(joint), creationPose_(creationPose) {}

        bool is_root() const { return joint_ == nullptr; }
        Vector3 position() const { return fromPxVec3(link_->getGlobalPose().p); }
        Quaternion quaternion() const { return fromPxQuat(link_->getGlobalPose().q); }

        // External force/impulse on this link (a PxArticulationLink is a PxRigidBody).
        // Use for perturbations — e.g. random shoves to train push recovery.
        void add_force(const Vector3& v) { link_->addForce(toPxVec3(v), PxForceMode::eFORCE); }
        void add_impulse(const Vector3& v) { link_->addForce(toPxVec3(v), PxForceMode::eIMPULSE); }

        void set_drive_target(float t) { joint()->setDriveTarget(PxArticulationAxis::eTWIST, t); }
        void set_drive_velocity(float v) { joint()->setDriveVelocity(PxArticulationAxis::eTWIST, v); }
        float joint_position() const { return joint()->getJointPosition(PxArticulationAxis::eTWIST); }
        float joint_velocity() const { return joint()->getJointVelocity(PxArticulationAxis::eTWIST); }

        PxArticulationLink* raw() const { return link_; }
        PxTransform creationPose() const { return creationPose_; }

    private:
        PxArticulationJointReducedCoordinate* joint() const {
            if (!joint_) throw std::runtime_error("ArticulationLink: the root link has no joint");
            return joint_;
        }
        PxArticulationLink* link_;
        PxArticulationJointReducedCoordinate* joint_;
        PxTransform creationPose_;
    };

    // Reduced-coordinate articulation builder (a robot). Add links (root first,
    // then children with an inbound revolute joint), then finalize() to add it to
    // the world's scene. world.step() drives the bound visual meshes.
    class Articulation {
    public:
        Articulation(PhysxWorld& world, bool fixedBase, int solverPositionIters, bool disableSelfCollision)
            : world_(world) {
            art_ = world_.physics().createArticulationReducedCoordinate();
            if (!art_) throw std::runtime_error("createArticulationReducedCoordinate failed");
            art_->setArticulationFlag(PxArticulationFlag::eFIX_BASE, fixedBase);
            if (disableSelfCollision) art_->setArticulationFlag(PxArticulationFlag::eDISABLE_SELF_COLLISION, true);
            if (solverPositionIters > 0) art_->setSolverIterationCounts(static_cast<PxU32>(solverPositionIters), 1);
        }
        ~Articulation() {
            if (cache_) cache_->release();
            // If it was never added to a scene we still own it; once finalized the
            // scene owns it and releases it on world teardown (the world outlives us).
            if (art_ && !finalized_) art_->release();
        }

        // Episode reset: teleport the root to `pos` upright with zero velocity and
        // zero every joint position/velocity (back to the neutral build pose). The
        // bound visuals snap to the new state on the next world.step().
        void reset(const Vector3& pos) {
            if (!finalized_) throw std::runtime_error("Articulation.reset: finalize() first");
            art_->setRootGlobalPose(PxTransform(toPxVec3(pos), PxQuat(PxIdentity)), false);
            if (!cache_) cache_ = art_->createCache();
            art_->zeroCache(*cache_);
            art_->applyCache(*cache_,
                             PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY |
                                     PxArticulationCacheFlag::eROOT_VELOCITIES,
                             true);
        }
        Articulation(const Articulation&) = delete;
        Articulation& operator=(const Articulation&) = delete;

        ArticulationLink add_link(ArticulationLink* parent, Mesh& mesh, float density,
                                  const std::array<float, 3>& axis, const std::array<float, 3>& anchor,
                                  bool limited, float lower, float upper,
                                  float stiffness, float damping, float maxForce, float driveTarget) {
            if (finalized_) throw std::runtime_error("Articulation.add_link: already finalized (no links after finalize)");
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("Articulation.add_link: mesh has no geometry");
            const auto shape = inferLinkShape(*g);
            if (!shape.valid) throw std::runtime_error("Articulation.add_link: unsupported geometry (use Box/Sphere/Capsule)");

            mesh.updateMatrixWorld();
            Vector3 pos, scl;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scl);
            const PxTransform linkPose(toPxVec3(pos), toPxQuat(rot));

            PxArticulationLink* parentLink = parent ? parent->raw() : nullptr;
            PxArticulationLink* link = art_->createLink(parentLink, linkPose);
            if (!link) throw std::runtime_error("Articulation.add_link: createLink failed");

            PxShape* s = world_.physics().createShape(shape.geom.any(), world_.defaultMaterial(), true);
            s->setLocalPose(shape.localPose);
            link->attachShape(*s);
            s->release();
            PxRigidBodyExt::updateMassAndInertia(*link, density);

            PxArticulationJointReducedCoordinate* joint = nullptr;
            if (parentLink) {
                joint = link->getInboundJoint();
                joint->setJointType(PxArticulationJointType::eREVOLUTE);
                // Joint frame in world space: origin at the hinge anchor, X axis
                // along the hinge axis (eTWIST rotates about X).
                const PxTransform jointWorld(toPxVec3(Vector3(anchor[0], anchor[1], anchor[2])),
                                             shortestArc(PxVec3(1, 0, 0), toPxVec3(Vector3(axis[0], axis[1], axis[2]))));
                joint->setParentPose(parent->creationPose().getInverse() * jointWorld);
                joint->setChildPose(linkPose.getInverse() * jointWorld);
                joint->setMotion(PxArticulationAxis::eTWIST,
                                 limited ? PxArticulationMotion::eLIMITED : PxArticulationMotion::eFREE);
                if (limited) joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(lower, upper));
                if (stiffness > 0.f || damping > 0.f) {
                    joint->setDriveParams(PxArticulationAxis::eTWIST,
                                          PxArticulationDrive(stiffness, damping, maxForce, PxArticulationDriveType::eFORCE));
                    joint->setDriveTarget(PxArticulationAxis::eTWIST, driveTarget, false);// autowake=false: pre-scene
                }
            }
            // A PxArticulationLink is a PxRigidActor, so the rigid-body bind path
            // syncs the visual mesh to the simulated link pose.
            world_.bind(mesh, *link);
            return ArticulationLink(link, joint, linkPose);
        }

        void finalize() {
            if (finalized_) return;
            world_.scene().addArticulation(*art_);
            finalized_ = true;
        }

    private:
        PhysxWorld& world_;
        PxArticulationReducedCoordinate* art_ = nullptr;
        PxArticulationCache* cache_ = nullptr;
        bool finalized_ = false;
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
                .def_property_readonly("is_root", &ArticulationLink::is_root)
                .def_property_readonly("position", &ArticulationLink::position)
                .def_property_readonly("quaternion", &ArticulationLink::quaternion)
                .def_property_readonly("joint_position", &ArticulationLink::joint_position, "Joint angle (radians).")
                .def_property_readonly("joint_velocity", &ArticulationLink::joint_velocity, "Joint angular velocity (rad/s).")
                .def("add_force", &ArticulationLink::add_force, py::arg("force"), "Apply an external force (N) to this link.")
                .def("add_impulse", &ArticulationLink::add_impulse, py::arg("impulse"), "Apply an external impulse (kg·m/s) — e.g. a random shove.")
                .def("set_drive_target", &ArticulationLink::set_drive_target, py::arg("target"),
                     "Set the PD drive's target angle (radians).")
                .def("set_drive_velocity", &ArticulationLink::set_drive_velocity, py::arg("velocity"));

        py::class_<Articulation>(m, "Articulation",
                                 "A reduced-coordinate articulation (robot): a tree of links joined by "
                                 "motorized revolute joints. Build with add_link (root first), then "
                                 "finalize(); stepping the world drives the bound meshes.")
                .def("add_link",
                     [](Articulation& a, Mesh& mesh, const py::object& parent, float density,
                        const std::array<float, 3>& axis, const std::array<float, 3>& anchor,
                        const py::object& lower, const py::object& upper,
                        float stiffness, float damping, float max_force, float drive_target) {
                         ArticulationLink* p = parent.is_none() ? nullptr : parent.cast<ArticulationLink*>();
                         const bool limited = !lower.is_none() && !upper.is_none();
                         const float lo = limited ? lower.cast<float>() : 0.f;
                         const float hi = limited ? upper.cast<float>() : 0.f;
                         return a.add_link(p, mesh, density, axis, anchor, limited, lo, hi,
                                           stiffness, damping, max_force, drive_target);
                     },
                     py::arg("mesh"), py::arg("parent") = py::none(), py::arg("density") = 1000.f,
                     py::arg("axis") = std::array<float, 3>{0.f, 0.f, 1.f},
                     py::arg("anchor") = std::array<float, 3>{0.f, 0.f, 0.f},
                     py::arg("lower") = py::none(), py::arg("upper") = py::none(),
                     py::arg("stiffness") = 0.f, py::arg("damping") = 0.f,
                     py::arg("max_force") = 1e6f, py::arg("drive_target") = 0.f,
                     py::keep_alive<1, 2>(), py::keep_alive<0, 1>(),
                     "Add a link. parent=None → the fixed/free root; otherwise attach a revolute inbound "
                     "joint at world-space `anchor` about world-space `axis`. lower/upper (radians) set the "
                     "joint limits (omit both for a free axis); stiffness/damping/max_force configure the PD "
                     "position drive (stiffness>0 motorizes it). Shape is inferred from the mesh "
                     "(Box/Sphere/Capsule). Returns an ArticulationLink.")
                .def("finalize", &Articulation::finalize,
                     "Add the finished articulation to the scene. No links may be added afterwards.")
                .def("reset", &Articulation::reset, py::arg("position"),
                     "Episode reset: teleport the root upright to `position` with zero velocity and "
                     "zero all joint positions/velocities (back to the neutral build pose).");

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
                     py::arg("callback"), "Register callback(dt) fired after each fixed substep.")
                .def("create_articulation",
                     [](PhysxWorld& w, bool fixed_base, int solver_position_iterations, bool disable_self_collision) {
                         return std::make_unique<Articulation>(w, fixed_base, solver_position_iterations, disable_self_collision);
                     },
                     py::arg("fixed_base") = false, py::arg("solver_position_iterations") = 8,
                     py::arg("disable_self_collision") = false, py::keep_alive<0, 1>(),
                     "Create a reduced-coordinate articulation (robot). fixed_base pins the root to the "
                     "world (use for arms; leave false for free-floating bodies like a walking robot). "
                     "Add links, then call finalize().");

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
