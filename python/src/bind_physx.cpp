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
#include "threepp/extras/physx/PhysxGpuBatch.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"

#include <PxPhysicsAPI.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
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
        ArticulationLink(PxArticulationLink* link, PxArticulationJointReducedCoordinate* joint,
                         PxTransform creationPose, PxArticulationAxis::Enum axis = PxArticulationAxis::eTWIST)
            : link_(link), joint_(joint), creationPose_(creationPose), axis_(axis) {}

        bool is_root() const { return joint_ == nullptr; }
        Vector3 position() const { return fromPxVec3(link_->getGlobalPose().p); }
        Quaternion quaternion() const { return fromPxQuat(link_->getGlobalPose().q); }

        // External force/impulse on this link (a PxArticulationLink is a PxRigidBody).
        // Use for perturbations — e.g. random shoves to train push recovery, or to
        // force-drive a cart link in a cart-pole on the CPU deployment path.
        void add_force(const Vector3& v) { link_->addForce(toPxVec3(v), PxForceMode::eFORCE); }
        void add_impulse(const Vector3& v) { link_->addForce(toPxVec3(v), PxForceMode::eIMPULSE); }

        // Operate on this joint's actual motion axis (eTWIST for revolute, eX for
        // prismatic) so the accessors are correct for both joint types.
        void set_drive_target(float t) { joint()->setDriveTarget(axis_, t); }
        void set_drive_velocity(float v) { joint()->setDriveVelocity(axis_, v); }
        float joint_position() const { return joint()->getJointPosition(axis_); }
        float joint_velocity() const { return joint()->getJointVelocity(axis_); }

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
        PxArticulationAxis::Enum axis_;
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

        // CPU state I/O (per-joint getters/setters, the cache) is NOT valid once the world
        // runs in direct-GPU mode — that state isn't synced. Fail loudly instead of silently
        // returning stale data; use PhysxGpuBatch for state I/O under direct_gpu.
        void cpuOnly(const char* what) const {
            if (world_.directGpuEnabled())
                throw std::runtime_error(std::string("Articulation.") + what +
                                         ": not valid under direct_gpu — use PhysxGpuBatch for state I/O");
        }

        // Episode reset: teleport the root to `pos` upright with zero velocity and
        // zero every joint position/velocity (back to the neutral build pose). The
        // bound visuals snap to the new state on the next world.step().
        void reset(const Vector3& pos) {
            if (!finalized_) throw std::runtime_error("Articulation.reset: finalize() first");
            cpuOnly("reset");
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
                                  float stiffness, float damping, float maxForce, float driveTarget,
                                  const std::string& jointType, float jointFriction) {
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
            if (!parentLink) rootLink_ = link;// the root link (for batched root_state)

            PxShape* s = world_.physics().createShape(shape.geom.any(), world_.defaultMaterial(), true);
            s->setLocalPose(shape.localPose);
            link->attachShape(*s);
            s->release();
            PxRigidBodyExt::updateMassAndInertia(*link, density);

            // Revolute (hinge, rotation about the axis) or prismatic (slider,
            // translation along the axis). For revolute the motion DOF is eTWIST
            // (rotation about joint-frame X); for prismatic it's eX (translation
            // along joint-frame X) — same frame, different DOF.
            const bool prismatic = (jointType == "prismatic");
            const PxArticulationAxis::Enum ax = prismatic ? PxArticulationAxis::eX : PxArticulationAxis::eTWIST;

            PxArticulationJointReducedCoordinate* joint = nullptr;
            if (parentLink) {
                joint = link->getInboundJoint();
                joint->setJointType(prismatic ? PxArticulationJointType::ePRISMATIC
                                              : PxArticulationJointType::eREVOLUTE);
                // Joint frame in world space: origin at the anchor, X axis along the
                // hinge/slide axis (eTWIST rotates about X / eX translates along X).
                const PxTransform jointWorld(toPxVec3(Vector3(anchor[0], anchor[1], anchor[2])),
                                             shortestArc(PxVec3(1, 0, 0), toPxVec3(Vector3(axis[0], axis[1], axis[2]))));
                joint->setParentPose(parent->creationPose().getInverse() * jointWorld);
                joint->setChildPose(linkPose.getInverse() * jointWorld);
                joint->setMotion(ax, limited ? PxArticulationMotion::eLIMITED : PxArticulationMotion::eFREE);
                // Joint friction holds a joint static until the applied torque exceeds it —
                // which silently fakes balancing tasks (a near-upright pole's tiny gravity
                // torque never overcomes the default friction, so it never falls). Default to
                // frictionless so "free" joints are genuinely free.
                joint->setFrictionCoefficient(jointFriction);
                joint->setArmature(ax, 0.0f);
                if (limited) joint->setLimitParams(ax, PxArticulationLimit(lower, upper));
                if (stiffness > 0.f || damping > 0.f) {
                    joint->setDriveParams(ax,
                                          PxArticulationDrive(stiffness, damping, maxForce, PxArticulationDriveType::eFORCE));
                    joint->setDriveTarget(ax, driveTarget, false);// autowake=false: pre-scene
                }
                joints_.push_back(joint);
            }
            // A PxArticulationLink is a PxRigidActor, so the rigid-body bind path
            // syncs the visual mesh to the simulated link pose.
            world_.bind(mesh, *link);
            return ArticulationLink(link, joint, linkPose, ax);
        }

        void finalize() {
            if (finalized_) return;
            world_.scene().addArticulation(*art_);
            finalized_ = true;
        }

        // Set all joint positions (DOF order) and zero velocities, via the cache.
        // Use to place an articulation in a chosen configuration — e.g. start a
        // cart-pole hanging straight down for a swing-up demo.
        void set_joint_positions(const py::array_t<float>& pos) {
            if (!finalized_) throw std::runtime_error("Articulation.set_joint_positions: finalize() first");
            cpuOnly("set_joint_positions");
            if (!cache_) cache_ = art_->createCache();
            art_->zeroCache(*cache_);
            auto r = pos.unchecked<1>();
            const PxU32 dof = art_->getDofs();
            const PxU32 n = std::min<PxU32>(dof, static_cast<PxU32>(r.shape(0)));
            for (PxU32 i = 0; i < n; ++i) cache_->jointPosition[i] = r(i);
            art_->applyCache(*cache_, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
        }

        // Underlying PhysX articulation — used to assemble a PhysxGpuBatch over many
        // identical robots for GPU-resident vectorized stepping.
        PxArticulationReducedCoordinate* raw_art() const { return art_; }
        bool finalized() const { return finalized_; }

        // Batched joint I/O — one call reads/writes every revolute joint (in
        // add_link order), instead of a Python call per joint. This is the hot
        // path for vectorized RL: it collapses ~36 pybind calls per robot per
        // step to 3, clawing back the cost of the Python bridge.
        py::array_t<float> joint_positions() const {
            cpuOnly("joint_positions");
            py::array_t<float> a(static_cast<py::ssize_t>(joints_.size()));
            float* p = a.mutable_data();
            for (size_t i = 0; i < joints_.size(); ++i)
                p[i] = joints_[i]->getJointPosition(PxArticulationAxis::eTWIST);
            return a;
        }
        py::array_t<float> joint_velocities() const {
            cpuOnly("joint_velocities");
            py::array_t<float> a(static_cast<py::ssize_t>(joints_.size()));
            float* p = a.mutable_data();
            for (size_t i = 0; i < joints_.size(); ++i)
                p[i] = joints_[i]->getJointVelocity(PxArticulationAxis::eTWIST);
            return a;
        }
        void set_drive_targets(const py::array_t<float>& arr) {
            cpuOnly("set_drive_targets");
            auto r = arr.unchecked<1>();
            const size_t n = std::min<size_t>(joints_.size(), static_cast<size_t>(r.shape(0)));
            // autowake=true on the last write: a policy that drives a settled robot
            // (e.g. after a reset/settle) must wake it, or the targets are ignored and
            // the articulation stays frozen at its rest pose.
            for (size_t i = 0; i < n; ++i)
                joints_[i]->setDriveTarget(PxArticulationAxis::eTWIST, r(i), i + 1 == n);
        }

        // For each joint (in add-order, the order joint_positions()/set_drive_targets use),
        // its low-level DOF slot — the index it occupies in the direct-GPU joint buffers
        // (PhysxGpuBatch), which follow PhysX's cache layout, NOT add-order. Use this to
        // reconcile a GPU-trained policy (GPU-DOF order) with the CPU getters (add-order):
        // obs_gpu[dof_order[i]] = cpu_value[i];  cpu_target[i] = gpu_target[dof_order[i]].
        py::array_t<int> dof_order() const {
            const size_t n = joints_.size();
            // GPU/cache DOF order follows ascending low-level link index, one DOF per
            // revolute link. So a joint's GPU slot = the rank of its child link's index.
            std::vector<PxU32> linkIdx(n);
            for (size_t i = 0; i < n; ++i)
                linkIdx[i] = joints_[i]->getChildArticulationLink().getLinkIndex();
            py::array_t<int> a(static_cast<py::ssize_t>(n));
            int* p = a.mutable_data();
            for (size_t i = 0; i < n; ++i) {
                int rank = 0;
                for (size_t j = 0; j < n; ++j)
                    if (linkIdx[j] < linkIdx[i]) ++rank;
                p[i] = rank;
            }
            return a;
        }

        // Root link world pose as one array [px,py,pz, qx,qy,qz,qw] — one call
        // instead of reading position + quaternion + their 7 components separately.
        py::array_t<float> root_state() const {
            cpuOnly("root_state");
            py::array_t<float> a(7);
            float* p = a.mutable_data();
            const PxTransform t = rootLink_ ? rootLink_->getGlobalPose() : PxTransform(PxIdentity);
            p[0] = t.p.x; p[1] = t.p.y; p[2] = t.p.z;
            p[3] = t.q.x; p[4] = t.q.y; p[5] = t.q.z; p[6] = t.q.w;
            return a;
        }

    private:
        PhysxWorld& world_;
        PxArticulationReducedCoordinate* art_ = nullptr;
        PxArticulationCache* cache_ = nullptr;
        PxArticulationLink* rootLink_ = nullptr;
        std::vector<PxArticulationJointReducedCoordinate*> joints_;// non-root joints, add order
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
                        float stiffness, float damping, float max_force, float drive_target,
                        const std::string& joint_type, float joint_friction) {
                         ArticulationLink* p = parent.is_none() ? nullptr : parent.cast<ArticulationLink*>();
                         const bool limited = !lower.is_none() && !upper.is_none();
                         const float lo = limited ? lower.cast<float>() : 0.f;
                         const float hi = limited ? upper.cast<float>() : 0.f;
                         return a.add_link(p, mesh, density, axis, anchor, limited, lo, hi,
                                           stiffness, damping, max_force, drive_target, joint_type, joint_friction);
                     },
                     py::arg("mesh"), py::arg("parent") = py::none(), py::arg("density") = 1000.f,
                     py::arg("axis") = std::array<float, 3>{0.f, 0.f, 1.f},
                     py::arg("anchor") = std::array<float, 3>{0.f, 0.f, 0.f},
                     py::arg("lower") = py::none(), py::arg("upper") = py::none(),
                     py::arg("stiffness") = 0.f, py::arg("damping") = 0.f,
                     py::arg("max_force") = 1e6f, py::arg("drive_target") = 0.f,
                     py::arg("joint_type") = "revolute", py::arg("joint_friction") = 0.0f,
                     py::keep_alive<1, 2>(), py::keep_alive<0, 1>(),
                     "Add a link. parent=None → the fixed/free root; otherwise attach an inbound joint at "
                     "world-space `anchor` along world-space `axis`. joint_type='revolute' (hinge about axis) "
                     "or 'prismatic' (slider along axis). lower/upper set the joint limits (radians for "
                     "revolute, metres for prismatic; omit both for a free axis); stiffness/damping/max_force "
                     "configure the PD drive (stiffness>0 motorizes it; leave 0 for a passive/force-controlled "
                     "joint). Shape is inferred from the mesh (Box/Sphere/Capsule). Returns an ArticulationLink.")
                .def("finalize", &Articulation::finalize,
                     "Add the finished articulation to the scene. No links may be added afterwards.")
                .def("reset", &Articulation::reset, py::arg("position"),
                     "Episode reset: teleport the root upright to `position` with zero velocity and "
                     "zero all joint positions/velocities (back to the neutral build pose).")
                .def("set_joint_positions", &Articulation::set_joint_positions, py::arg("positions"),
                     "Set all joint positions (DOF order) and zero velocities — e.g. place a cart-pole "
                     "hanging straight down for a swing-up demo.")
                .def("joint_positions", &Articulation::joint_positions,
                     "All revolute joint angles (radians) as one numpy array, in add_link order.")
                .def("joint_velocities", &Articulation::joint_velocities,
                     "All revolute joint angular velocities (rad/s) as one numpy array.")
                .def("set_drive_targets", &Articulation::set_drive_targets, py::arg("targets"),
                     "Set every joint's PD drive target from one numpy array — the batched hot path "
                     "for vectorized stepping (one call instead of one per joint).")
                .def("root_state", &Articulation::root_state,
                     "Root link world pose as numpy [px,py,pz, qx,qy,qz,qw] in one call.")
                .def("dof_order", &Articulation::dof_order,
                     "Per add-order joint, its low-level DOF slot in the direct-GPU joint buffers "
                     "(PhysX cache order != add-order). Use to map a GPU-trained policy back to the "
                     "CPU getters: obs_gpu[dof_order[i]] = cpu[i]; cpu_target[i] = gpu_target[dof_order[i]].");

        py::class_<PhysxWorld>(m, "PhysxWorld",
                               "A PhysX rigid-body world wired to the threepp scene graph. Add meshes as "
                               "bodies, then call step(dt) each frame; every bound mesh's position/quaternion "
                               "follows the simulation. Pure CPU — no canvas or renderer required.")
                .def(py::init([](const Vector3& gravity, float fixed_timestep, int max_substeps,
                                 unsigned num_threads, bool gpu_dynamics, bool direct_gpu,
                                 std::uintptr_t cuda_context) {
                         PhysxWorld::Settings s;
                         s.gravity = gravity;
                         s.fixedTimestep = fixed_timestep;
                         s.maxSubSteps = max_substeps;
                         s.numThreads = num_threads;
                         s.enableGpuDynamics = gpu_dynamics;
                         s.enableDirectGpu = direct_gpu;
                         s.cudaContext = reinterpret_cast<CUcontext>(cuda_context);
                         return std::make_unique<PhysxWorld>(s);
                     }),
                     py::arg("gravity") = Vector3(0, -9.81f, 0),
                     py::arg("fixed_timestep") = 1.f / 60.f,
                     py::arg("max_substeps") = 4,
                     py::arg("num_threads") = 2,
                     py::arg("gpu_dynamics") = false,
                     py::arg("direct_gpu") = false,
                     py::arg("cuda_context") = 0,
                     "gpu_dynamics requires a CUDA GPU (needed for soft bodies). direct_gpu also "
                     "enables the PhysX direct-GPU API for batched GPU-resident articulation state "
                     "I/O (PhysxGpuBatch) — the basis for GPU vectorized RL. Under direct_gpu the "
                     "per-actor CPU getters and the binding-sync step() are NOT valid. cuda_context "
                     "(an existing CUcontext as an int, e.g. torch's primary context) makes PhysX "
                     "share that context instead of creating its own — required to mix PhysX GPU work "
                     "with the framework's cuBLAS/cuDNN on the same device.")
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
                             raw.push_back(a->raw_art());
                         }
                         return std::make_unique<threepp::PhysxGpuBatch>(world, std::move(raw));
                     }),
                     py::arg("world"), py::arg("articulations"), py::keep_alive<1, 2>(),
                     "world must be created with direct_gpu=True and outlive this batch.")
                .def_property_readonly("count", [](threepp::PhysxGpuBatch& b) { return b.count(); })
                .def_property_readonly("max_dofs", [](threepp::PhysxGpuBatch& b) { return b.maxDofs(); })
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
                .def("write_joint_target_pos", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "write_joint_target_pos"), Write::eJOINT_TARGET_POSITION); },
                     py::arg("tensor"), "Set all joints' PD position targets from the [n, max_dofs] float32 cuda tensor.")
                .def("write_joint_force", [cudaPtr](threepp::PhysxGpuBatch& b, const py::object& t) {
                         b.write(cudaPtr(t, std::int64_t(b.count()) * b.maxDofs(), "write_joint_force"), Write::eJOINT_FORCE); },
                     py::arg("tensor"),
                     "Apply per-DOF joint forces/torques (effort control) from the [n, max_dofs] float32 cuda "
                     "tensor. Re-apply each step (forces don't persist). Use for force-controlled joints.")
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
                         return reshape(b.readHost(Read::eROOT_ANGULAR_VELOCITY), b.count(), 3); });

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
