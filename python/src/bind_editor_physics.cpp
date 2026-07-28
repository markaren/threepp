// threepp.editor, physics half — the runtime face of the editor's PhysicsConfig.
//
// rigid_body_from_object(obj) / soft_body_from_object(obj) hand a script the
// body PhysX is actually simulating for that object, so a MonoBehaviour-style
// script can push, steer and read it instead of fighting the simulation by
// writing transforms it will overwrite next step.
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
// always failing.

#include "bindings.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

using namespace threepp;

namespace {

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
    }

}// namespace threepp_py
