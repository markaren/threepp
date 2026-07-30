// threepp.editor, physics half — the runtime face of the editor's PhysicsConfig.
//
// rigid_body_from_object(obj) / soft_body_from_object(obj) hand a script the
// body PhysX is actually simulating for that object, so a MonoBehaviour-style
// script can push, steer and read it instead of fighting the simulation by
// writing transforms it will overwrite next step.
// articulation_from_object(obj) is the same idea for a robot the session
// simulates as a reduced-coordinate articulation: joint state in, drive
// targets out — the surface a policy deployment loop needs, which is exactly
// what a script standing in for one wants.
// raycast(origin, direction, ...) is the other direction entirely: not a handle
// onto one authored body but a QUESTION put to the whole playing world — what is
// under my feet, what is in front of me, what am I aiming at. The ground check
// every character controller opens with.
//
// Three things about the contract, stated once:
//
//   * A body exists only DURING Play. An authored PhysicsConfig is just
//     userData until the physics session builds actors from it, so these
//     functions return None outside Play (and for an object with no physics).
//     That is the difference from spline_from_object, which reads authoring
//     data and works any time.
//   * A handle is TIED to the play session that produced it. Stop releases
//     every actor, so a handle kept across a stop/play raises rather than
//     dereferencing freed memory. Ask again after each start.
//   * The lookup walks UP the scene graph, like PhysxWorld::findActor: a
//     script on a child of a physics object still finds the body governing it.
//
// This file is compiled only where the PhysX SDK was found, and registers into
// the same threepp.editor submodule bind_editor.cpp creates — so in a build
// without PhysX the names simply are not there, rather than existing and
// always failing. It is also EDITOR-only: unlike most of python/src it is not
// one of the wheel's translation units, only one of threepp_editor_scripting's
// — which is what lets it reach the script host's handleFor below.

#include "bindings.hpp"

// The articulation handle answers in lists (joint names, positions, targets),
// which the RigidBody/SoftBody halves never needed.
#include <pybind11/stl.h>

// handleFor: a raycast hands back an object the script never passed IN, so it
// has to be cast to its concrete leaf type here (see the header's comment on
// why casting one as Object3D across the virtual base is a heap bug waiting to
// happen). The only such factory in the process, shared with the collision
// payload, and in this same static library.
#include "ScriptHost.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // bindings.hpp puts the alias inside threepp_py; RaycastHit holds a Python
    // object and lives out here, with the other handle types.
    namespace py = pybind11;

    Vector3 toVector3(const ::physx::PxVec3& v) { return {v.x, v.y, v.z}; }

    // Everything a handle needs from the session that created it. Held as a
    // weak token rather than a pointer to the session, so a handle cannot keep
    // a stopped session's world alive and cannot silently read a dead one.
    struct Lifetime {

        std::weak_ptr<const void> token;

        [[nodiscard]] bool alive() const { return !token.expired(); }

        void require(const char* what) const {

            if (!alive()) {
                throw std::runtime_error(std::string("this ") + what +
                                         " belongs to a play session that has stopped - "
                                         "ask for it again after Play starts");
            }
        }
    };


    // A PxRigidActor, as a script sees it. Static bodies expose only what makes
    // sense for one; asking a static body for its velocity is a mistake worth
    // reporting rather than answering with a zero.
    class RigidBody {

    public:
        RigidBody(std::shared_ptr<Object3D> object, ::physx::PxRigidActor* actor, Lifetime lifetime)
            : object_(std::move(object)), actor_(actor), lifetime_(std::move(lifetime)) {}

        [[nodiscard]] bool valid() const { return lifetime_.alive(); }

        [[nodiscard]] std::shared_ptr<Object3D> object() const { return object_; }

        [[nodiscard]] bool isStatic() const { return !asDynamic(false); }

        [[nodiscard]] bool isKinematic() const {

            auto* body = asDynamic(false);
            return body && body->getRigidBodyFlags().isSet(::physx::PxRigidBodyFlag::eKINEMATIC);
        }

        [[nodiscard]] Vector3 position() const {

            lifetime_.require("rigid body");
            return toVector3(actor_->getGlobalPose().p);
        }

        [[nodiscard]] Quaternion rotation() const {

            lifetime_.require("rigid body");
            const auto q = actor_->getGlobalPose().q;
            return Quaternion(q.x, q.y, q.z, q.w);
        }

        [[nodiscard]] Vector3 velocity() const { return toVector3(dynamic()->getLinearVelocity()); }

        void setVelocity(const Vector3& v) {

            dynamic()->setLinearVelocity(toPxVec3(v));
        }

        [[nodiscard]] Vector3 angularVelocity() const {

            return toVector3(dynamic()->getAngularVelocity());
        }

        void setAngularVelocity(const Vector3& v) {

            dynamic()->setAngularVelocity(toPxVec3(v));
        }

        [[nodiscard]] float mass() const { return dynamic()->getMass(); }

        void setMass(float mass) {

            ::physx::PxRigidBodyExt::setMassAndUpdateInertia(*dynamic(), std::max(mass, 1e-3f));
        }

        // Continuous force in newtons, applied for the coming step. Call it
        // every update while the thrust should be on — this is not a setting.
        void applyForce(const Vector3& force) {

            dynamic()->addForce(toPxVec3(force), ::physx::PxForceMode::eFORCE);
        }

        // An instantaneous kick, in newton-seconds. One call is one kick.
        void applyImpulse(const Vector3& impulse) {

            dynamic()->addForce(toPxVec3(impulse), ::physx::PxForceMode::eIMPULSE);
        }

        void applyTorque(const Vector3& torque) {

            dynamic()->addTorque(toPxVec3(torque), ::physx::PxForceMode::eFORCE);
        }

        void applyTorqueImpulse(const Vector3& torque) {

            dynamic()->addTorque(toPxVec3(torque), ::physx::PxForceMode::eIMPULSE);
        }

        // Where a kinematic body should be by the end of the next step. PhysX
        // sweeps it there, so it pushes dynamics on the way instead of
        // teleporting through them — which is the entire reason to author a
        // body kinematic rather than moving its transform.
        void setKinematicTarget(const Vector3& position, const Quaternion* rotation) {

            auto* body = dynamic();
            if (!body->getRigidBodyFlags().isSet(::physx::PxRigidBodyFlag::eKINEMATIC)) {
                throw std::runtime_error("set_kinematic_target needs a body authored Kinematic");
            }
            const auto q = rotation ? toPxQuat(*rotation) : body->getGlobalPose().q;
            body->setKinematicTarget(::physx::PxTransform(toPxVec3(position), q));
        }

        [[nodiscard]] bool sleeping() const { return dynamic()->isSleeping(); }

        void wakeUp() { dynamic()->wakeUp(); }

        [[nodiscard]] std::string repr() const {

            if (!valid()) return "<RigidBody (session stopped)>";
            return "<RigidBody '" + object_->name + "' " +
                   (isStatic() ? "static" : (isKinematic() ? "kinematic" : "dynamic")) + ">";
        }

    private:
        std::shared_ptr<Object3D> object_;
        ::physx::PxRigidActor* actor_;
        Lifetime lifetime_;

        // The dynamic face of the actor. `required` false is the query form
        // used by is_static / is_kinematic, which must answer rather than throw.
        [[nodiscard]] ::physx::PxRigidDynamic* asDynamic(bool required = true) const {

            lifetime_.require("rigid body");
            auto* body = actor_->is<::physx::PxRigidDynamic>();
            if (!body && required) {
                throw std::runtime_error("'" + object_->name +
                                         "' is a static body - it has no velocity, mass or forces");
            }
            return body;
        }

        [[nodiscard]] ::physx::PxRigidDynamic* dynamic() const { return asDynamic(true); }
    };


    // A deformable volume, as a script sees it. Deliberately thin: PhysX drives
    // a soft body through per-vertex GPU buffers, so there is no per-actor
    // "push this" to expose. What a script does want is where the thing IS now
    // that it is not a rigid transform any more — the object's own position is
    // zero for the whole of Play, because the mesh carries world-space vertices.
    class SoftBodyHandle {

    public:
        SoftBodyHandle(std::shared_ptr<Object3D> object, SoftBody* body, Lifetime lifetime)
            : object_(std::move(object)), body_(body), lifetime_(std::move(lifetime)) {}

        [[nodiscard]] bool valid() const { return lifetime_.alive(); }

        [[nodiscard]] std::shared_ptr<Object3D> object() const { return object_; }

        [[nodiscard]] Vector3 center() const {

            const auto bounds = worldBounds();
            return toVector3(bounds.getCenter());
        }

        [[nodiscard]] Vector3 boundsMin() const { return toVector3(worldBounds().minimum); }

        [[nodiscard]] Vector3 boundsMax() const { return toVector3(worldBounds().maximum); }

        [[nodiscard]] int vertexCount() const {

            lifetime_.require("soft body");
            const auto geometry = body_->visualGeometry();
            const auto* positions = geometry ? geometry->getAttribute<float>("position") : nullptr;
            return positions ? static_cast<int>(positions->count()) : 0;
        }

        // Off when the script updates normals itself, or does not need them —
        // recomputing them every step is the bulk of a soft body's CPU cost.
        [[nodiscard]] bool recomputeNormals() const {

            lifetime_.require("soft body");
            return body_->recomputeNormals();
        }

        void setRecomputeNormals(bool enabled) {

            lifetime_.require("soft body");
            body_->setRecomputeNormals(enabled);
        }

        [[nodiscard]] std::string repr() const {

            if (!valid()) return "<SoftBody (session stopped)>";
            return "<SoftBody '" + object_->name + "' " + std::to_string(vertexCount()) + " verts>";
        }

    private:
        std::shared_ptr<Object3D> object_;
        SoftBody* body_;
        Lifetime lifetime_;

        [[nodiscard]] ::physx::PxBounds3 worldBounds() const {

            lifetime_.require("soft body");
            return body_->actor()->getWorldBounds(1.f);
        }
    };


    // A robot the play session simulates as a reduced-coordinate articulation,
    // as a script sees it: the JOINT-SPACE face. Positions and velocities read
    // back in the articulation's own DOF order (`joint_names`), radians for a
    // revolute joint and metres for a prismatic one; drive targets go out the
    // same way. Fixed URDF joints are collapsed by the articulation builder, so
    // this order is NOT the visual Robot's joint order — the names are the
    // bridge, which is why they are exposed at all.
    class ArticulationHandle {

    public:
        ArticulationHandle(std::shared_ptr<Object3D> object,
                           const editor::PhysicsPlaySession::PlayedArticulation* played,
                           Lifetime lifetime)
            : object_(std::move(object)), played_(played), lifetime_(std::move(lifetime)) {}

        [[nodiscard]] bool valid() const { return lifetime_.alive(); }

        [[nodiscard]] std::shared_ptr<Object3D> object() const { return object_; }

        [[nodiscard]] std::vector<std::string> jointNames() const {

            require();
            return played_->jointNames;
        }

        [[nodiscard]] std::size_t numDof() const {

            require();
            return played_->jointNames.size();
        }

        [[nodiscard]] std::vector<float> jointPositions() const {

            require();
            return played_->articulation->jointPositions();
        }

        [[nodiscard]] std::vector<float> jointVelocities() const {

            require();
            return played_->articulation->jointVelocities();
        }

        // One target per DOF, in joint_names order. The PD drive authored in the
        // robot's Articulation section pulls each joint toward its target — this
        // is a setpoint, not a teleport, and with zero authored stiffness the
        // drive is off and targets are inert.
        void setDriveTargets(const std::vector<float>& targets) {

            require();
            if (targets.size() != played_->jointNames.size()) {
                throw std::runtime_error(
                        "set_drive_targets: got " + std::to_string(targets.size()) +
                        " values for " + std::to_string(played_->jointNames.size()) +
                        " DOFs - one per entry of joint_names");
            }
            played_->articulation->setDriveTargets(targets.data(), targets.size());
        }

        void setDriveTargetByIndex(std::size_t index, float value) {

            require();
            if (index >= played_->links.size()) {
                throw std::runtime_error("set_drive_target: index " + std::to_string(index) +
                                         " out of range for " +
                                         std::to_string(played_->links.size()) + " DOFs");
            }
            // ArticulationLink is a value handle onto the PhysX joint, so a copy
            // drives the same joint — and it drives the joint's actual motion
            // axis, so this is correct for revolute and prismatic alike.
            ArticulationLink link = played_->links[index];
            link.setDriveTarget(value);
        }

        void setDriveTargetByName(const std::string& joint, float value) {

            require();
            for (std::size_t i = 0; i < played_->jointNames.size(); ++i) {
                if (played_->jointNames[i] == joint) {
                    setDriveTargetByIndex(i, value);
                    return;
                }
            }
            throw std::runtime_error("set_drive_target: \"" + joint +
                                     "\" is not a simulated DOF of this robot (see joint_names)");
        }

        [[nodiscard]] Vector3 rootPosition() const {

            require();
            const auto s = played_->articulation->rootState();
            return {s[0], s[1], s[2]};
        }

        [[nodiscard]] Quaternion rootRotation() const {

            require();
            const auto s = played_->articulation->rootState();
            return Quaternion(s[3], s[4], s[5], s[6]);
        }

        [[nodiscard]] Vector3 rootVelocity() const {

            require();
            const auto v = played_->articulation->rootVelocity();
            return {v[0], v[1], v[2]};
        }

        [[nodiscard]] Vector3 rootAngularVelocity() const {

            require();
            const auto v = played_->articulation->rootVelocity();
            return {v[3], v[4], v[5]};
        }

        [[nodiscard]] std::string repr() const {

            if (!valid()) return "<Articulation (session stopped)>";
            return "<Articulation '" + object_->name + "' " +
                   std::to_string(played_->jointNames.size()) + " dof>";
        }

    private:
        std::shared_ptr<Object3D> object_;
        const editor::PhysicsPlaySession::PlayedArticulation* played_;
        Lifetime lifetime_;

        // played_ points into the session's own list, freed on stop() — the
        // lifetime gate is what makes dereferencing it safe, same contract as
        // RigidBody's raw actor pointer.
        void require() const { lifetime_.require("articulation"); }
    };


    // What a raycast that hit something answers with. A value, entirely: the
    // shape it came off may be gone by the next step, so nothing here points
    // into PhysX.
    struct RaycastHit {

        // The authored object the hit actor belongs to, as its concrete type —
        // or None when the actor answers to nothing the script can name.
        py::object object;
        Vector3 point;
        Vector3 normal;
        float distance = 0.f;
    };


    // Actors a query must not see.
    //
    // A PRE-filter, not a post-filter, because what is being excluded is an
    // IDENTITY: which actor, known before a single triangle is touched. Rejecting
    // the shape ahead of the exact intersection test is both the cheaper answer
    // and the one PhysX documents for this; post-filtering would compute the
    // intersection and then throw it away.
    //
    // A callback rather than query filter DATA, because the shapes' query filter
    // data is not ours to write. PhysxWorld leaves it zero — its one filter bit
    // (kContactReportFilterBit) is SIMULATION data, read by the filter shader —
    // and a query whose own data is zero skips the hardcoded equation entirely,
    // so every shape reaches this callback. Marking an ignore in per-shape data
    // would mean editing the world, and unediting it afterwards, to ask one
    // question.
    class IgnoreActors: public ::physx::PxQueryFilterCallback {

    public:
        explicit IgnoreActors(const std::vector<const ::physx::PxRigidActor*>& actors)
            : actors_(actors) {}

        ::physx::PxQueryHitType::Enum preFilter(const ::physx::PxFilterData&,
                                                const ::physx::PxShape*,
                                                const ::physx::PxRigidActor* actor,
                                                ::physx::PxHitFlags&) override {

            for (const auto* ignored : actors_) {
                if (ignored == actor) return ::physx::PxQueryHitType::eNONE;
            }
            // eBLOCK, not eTOUCH: with a prefilter installed PhysX stops
            // defaulting the hit type, and only a blocking hit reaches
            // PxRaycastBuffer::block — which is the nearest-hit answer this
            // function is.
            return ::physx::PxQueryHitType::eBLOCK;
        }

        ::physx::PxQueryHitType::Enum postFilter(const ::physx::PxFilterData&,
                                                 const ::physx::PxQueryHit&,
                                                 const ::physx::PxShape*,
                                                 const ::physx::PxRigidActor*) override {

            return ::physx::PxQueryHitType::eBLOCK;// never asked for; ePOSTFILTER is off
        }

    private:
        const std::vector<const ::physx::PxRigidActor*>& actors_;
    };


    // The world the editor is playing right now, or nullptr outside Play.
    editor::PhysicsPlaySession* playing() {

        auto* session = editor::PhysicsPlaySession::active();
        return (session && session->world()) ? session : nullptr;
    }

}// namespace

namespace threepp_py {

    void init_editor_physics(py::module_& m) {

        // bind_editor.cpp made the submodule; this adds the physics half to it.
        auto sub = m.attr("editor").cast<py::module_>();

        py::class_<RigidBody, std::shared_ptr<RigidBody>>(sub, "RigidBody")
                .def_property_readonly("object", &RigidBody::object,
                                       "The scene object this body governs.")
                .def_property_readonly("valid", &RigidBody::valid,
                                       "False once the play session that created it has stopped.")
                .def_property_readonly("is_static", &RigidBody::isStatic)
                .def_property_readonly("is_kinematic", &RigidBody::isKinematic)
                .def_property_readonly("position", &RigidBody::position,
                                       "WORLD-SPACE position of the body itself.")
                .def_property_readonly("rotation", &RigidBody::rotation,
                                       "WORLD-SPACE orientation of the body itself.")
                .def_property("velocity", &RigidBody::velocity, &RigidBody::setVelocity,
                              "Linear velocity in m/s. Dynamic bodies only.")
                .def_property("angular_velocity", &RigidBody::angularVelocity,
                              &RigidBody::setAngularVelocity,
                              "Angular velocity in rad/s. Dynamic bodies only.")
                .def_property("mass", &RigidBody::mass, &RigidBody::setMass,
                              "Mass in kg; setting it recomputes the inertia tensor.")
                .def("apply_force", &RigidBody::applyForce, py::arg("force"),
                     "Add a force in newtons for the coming step. Call it every update while "
                     "the force should act - it is not a setting.")
                .def("apply_impulse", &RigidBody::applyImpulse, py::arg("impulse"),
                     "Add an instantaneous impulse in newton-seconds.")
                .def("apply_torque", &RigidBody::applyTorque, py::arg("torque"),
                     "Add a torque in newton-metres for the coming step.")
                .def("apply_torque_impulse", &RigidBody::applyTorqueImpulse, py::arg("torque"),
                     "Add an instantaneous angular impulse.")
                .def("set_kinematic_target", &RigidBody::setKinematicTarget,
                     py::arg("position"), py::arg("rotation") = py::none(),
                     "Where a Kinematic body should be by the end of the next step. PhysX sweeps "
                     "it there, so it pushes dynamics on the way instead of teleporting through "
                     "them. Keeps the current orientation when rotation is None.")
                .def_property_readonly("sleeping", &RigidBody::sleeping,
                                       "True when the solver has parked this body.")
                .def("wake_up", &RigidBody::wakeUp, "Take the body out of sleep.")
                .def("__repr__", &RigidBody::repr);

        py::class_<SoftBodyHandle, std::shared_ptr<SoftBodyHandle>>(sub, "SoftBody")
                .def_property_readonly("object", &SoftBodyHandle::object,
                                       "The scene object whose mesh this body deforms.")
                .def_property_readonly("valid", &SoftBodyHandle::valid,
                                       "False once the play session that created it has stopped.")
                .def_property_readonly("center", &SoftBodyHandle::center,
                                       "WORLD-SPACE centre of the deformed body. The object's own "
                                       "position is zero throughout Play - the mesh carries "
                                       "world-space vertices - so this is how a script follows it.")
                .def_property_readonly("bounds_min", &SoftBodyHandle::boundsMin)
                .def_property_readonly("bounds_max", &SoftBodyHandle::boundsMax)
                .def_property_readonly("vertex_count", &SoftBodyHandle::vertexCount)
                .def_property("recompute_normals", &SoftBodyHandle::recomputeNormals,
                              &SoftBodyHandle::setRecomputeNormals,
                              "Recompute vertex normals every step (on by default). The bulk of a "
                              "soft body's CPU cost, and pointless for a flat-shaded body.")
                .def("__repr__", &SoftBodyHandle::repr);

        py::class_<ArticulationHandle, std::shared_ptr<ArticulationHandle>>(sub, "Articulation")
                .def_property_readonly("object", &ArticulationHandle::object,
                                       "The Robot this articulation simulates — the robot itself, "
                                       "even when the handle was asked for from one of its links.")
                .def_property_readonly("valid", &ArticulationHandle::valid,
                                       "False once the play session that created it has stopped.")
                .def_property_readonly("joint_names", &ArticulationHandle::jointNames,
                                       "The simulated DOFs, in the articulation's own order. Fixed "
                                       "URDF joints are collapsed, so this is NOT the visual Robot's "
                                       "joint order — match by name.")
                .def_property_readonly("num_dof", &ArticulationHandle::numDof)
                .def_property_readonly("joint_positions", &ArticulationHandle::jointPositions,
                                       "Joint positions in joint_names order: radians for a revolute "
                                       "joint, metres for a prismatic one.")
                .def_property_readonly("joint_velocities", &ArticulationHandle::jointVelocities,
                                       "Joint velocities in joint_names order: rad/s or m/s.")
                .def("set_drive_targets", &ArticulationHandle::setDriveTargets, py::arg("targets"),
                     "One PD setpoint per DOF, in joint_names order. The drive authored in the "
                     "Articulation section pulls each joint toward its target over the coming "
                     "steps - a setpoint, not a teleport, and inert with zero authored stiffness.")
                .def("set_drive_target", &ArticulationHandle::setDriveTargetByName,
                     py::arg("joint"), py::arg("value"),
                     "PD setpoint for one DOF, by its URDF joint name.")
                .def("set_drive_target", &ArticulationHandle::setDriveTargetByIndex,
                     py::arg("index"), py::arg("value"),
                     "PD setpoint for one DOF, by its index in joint_names.")
                .def_property_readonly("root_position", &ArticulationHandle::rootPosition,
                                       "WORLD-SPACE position of the root link.")
                .def_property_readonly("root_rotation", &ArticulationHandle::rootRotation,
                                       "WORLD-SPACE orientation of the root link.")
                .def_property_readonly("root_velocity", &ArticulationHandle::rootVelocity,
                                       "Root link linear velocity in m/s, world frame.")
                .def_property_readonly("root_angular_velocity", &ArticulationHandle::rootAngularVelocity,
                                       "Root link angular velocity in rad/s, world frame.")
                .def("__repr__", &ArticulationHandle::repr);

        sub.def(
                "rigid_body_from_object", [](const py::handle& h) -> py::object {
                    auto object = as_object3d(h);
                    auto* session = playing();
                    if (!object || !session) return py::none();
                    // The session's own registry, not PhysxWorld's binding list:
                    // static bodies are never bound (no pose to write back) and
                    // would otherwise be invisible to a script.
                    auto* actor = session->findActor(object.get());
                    if (!actor) return py::none();
                    return py::cast(std::make_shared<RigidBody>(
                            std::move(object), actor, Lifetime{session->lifetime()}));
                },
                py::arg("object"),
                "The RigidBody PhysX is simulating for `object`, or None when Play is not "
                "running or the object has no physics. The lookup walks up the scene graph, so "
                "a script on a child finds the body governing it.");

        sub.def(
                "soft_body_from_object", [](const py::handle& h) -> py::object {
                    auto object = as_object3d(h);
                    auto* session = playing();
                    if (!object || !session) return py::none();
                    auto* body = session->world()->findSoftBody(object.get());
                    if (!body) return py::none();
                    return py::cast(std::make_shared<SoftBodyHandle>(
                            std::move(object), body, Lifetime{session->lifetime()}));
                },
                py::arg("object"),
                "The SoftBody PhysX is simulating for `object`, or None when Play is not running "
                "or the object is not a soft body.");

        sub.def(
                "articulation_from_object", [](const py::handle& h) -> py::object {
                    auto object = as_object3d(h);
                    auto* session = playing();
                    if (!object || !session) return py::none();
                    const auto* played = session->findArticulation(object.get());
                    if (!played || !played->robot) return py::none();
                    // The handle's object is the ROBOT, whatever node asked: the
                    // articulation governs the whole subtree, and a script on a
                    // link wants the robot's joint table, not the link.
                    return py::cast(std::make_shared<ArticulationHandle>(
                            played->robot->shared_from_this(), played,
                            Lifetime{session->lifetime()}));
                },
                py::arg("object"),
                "The Articulation PhysX is simulating for `object`, or None when Play is not "
                "running or no articulated robot governs it. The lookup walks up the scene "
                "graph, so a script on any link of a robot finds the robot's articulation. "
                "Robots simulate only when their Articulation section says Simulate.");

        py::class_<RaycastHit>(
                sub, "RaycastHit",
                "What threepp.editor.raycast answers with when the ray hit something.\n\n"
                "Values, all of it - nothing here points into PhysX, so keeping one is safe.")
                .def_property_readonly(
                        "object", [](const RaycastHit& h) { return h.object; },
                        "The object the physics was authored on, as its concrete type (Mesh, "
                        "Group, Robot, ...) - or None when the actor answers to nothing the "
                        "script can name.")
                .def_readonly("point", &RaycastHit::point, "WORLD-SPACE point of the hit.")
                .def_readonly("normal", &RaycastHit::normal,
                              "Unit surface normal there, pointing OUT of the surface hit.")
                .def_readonly("distance", &RaycastHit::distance,
                              "Metres from `origin` to `point`, along the ray.")
                .def("__repr__", [](const RaycastHit& h) {
                    std::string name = "None";
                    if (!h.object.is_none()) {
                        try {
                            name = "'" + py::cast<std::string>(h.object.attr("name")) + "'";
                        } catch (const py::error_already_set&) {
                            name = "?";
                        }
                    }
                    return "<threepp.editor.RaycastHit " + name + " at " +
                           std::to_string(h.distance) + " m>";
                });

        sub.def(
                "raycast", [](const Vector3& origin, const Vector3& direction, float maxDistance,
                              const py::handle& ignore) -> py::object {
                    using namespace ::physx;

                    auto* session = playing();
                    if (!session) {
                        // A miss is None, so "not playing" must NOT be: the two
                        // would be indistinguishable, and a ground check that
                        // silently answers "nothing there" outside Play is a
                        // script that looks like it works.
                        throw std::runtime_error(
                                "raycast needs a playing physics world - there is none "
                                "(no PhysX build, or Play is not running)");
                    }

                    Vector3 unit(direction);
                    const float length = unit.length();
                    if (!(length > 0.f) || !std::isfinite(length)) {
                        throw std::invalid_argument(
                                "raycast: direction has no length - a ray needs somewhere to go");
                    }
                    unit.divideScalar(length);
                    if (!(maxDistance > 0.f)) {
                        throw std::invalid_argument("raycast: max_distance must be greater than zero");
                    }

                    // ALL the actors governing the ignored object, not just the
                    // one a handle would name: a subtree collider or a compound
                    // is several actors under one authored node, and skipping the
                    // first of them would skip almost nothing.
                    std::vector<const PxRigidActor*> ignored;
                    if (!ignore.is_none()) {
                        if (const auto object = as_object3d(ignore)) {
                            ignored = session->findActors(object.get());
                        }
                    }

                    PxQueryFilterData filter;// eSTATIC | eDYNAMIC
                    IgnoreActors exclude(ignored);
                    if (!ignored.empty()) filter.flags |= PxQueryFlag::ePREFILTER;

                    PxRaycastBuffer buffer;
                    const bool hit = session->world()->scene().raycast(
                            toPxVec3(origin), toPxVec3(unit), maxDistance, buffer,
                            PxHitFlag::eDEFAULT, filter,
                            ignored.empty() ? nullptr : &exclude);
                    if (!hit || !buffer.hasBlock) return py::none();

                    const auto& block = buffer.block;
                    RaycastHit result;
                    result.object = py::none();
                    if (auto* object = session->findObject(block.actor)) {
                        result.object = editor::scripting::handleFor(*object);
                    }
                    result.point = toVector3(block.position);
                    result.normal = toVector3(block.normal);
                    result.distance = block.distance;
                    return py::cast(result);
                },
                py::arg("origin"), py::arg("direction"),
                py::arg("max_distance") = PX_MAX_F32, py::arg("ignore") = py::none(),
                "Cast a ray through the playing physics world and return the NEAREST "
                "RaycastHit, or None when it hits nothing.\n\n"
                "`origin` and `direction` are Vector3, world space; direction is normalised "
                "here, and a zero-length one raises ValueError. `max_distance` is in metres "
                "and defaults to unbounded. `ignore` excludes every actor governing that "
                "object - pass your own object for a ground check, or the ray starts inside "
                "your own collider and hits it.\n\n"
                "Raises RuntimeError when no physics world is playing: a miss is None, so "
                "'not playing' cannot also be None without making the two the same answer.");
    }

}// namespace threepp_py
