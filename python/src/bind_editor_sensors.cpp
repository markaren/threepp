// threepp.editor, sensor half — the runtime face of the editor's SensorConfig.
//
// imu_from_object(obj) / encoder_from_object(obj) / force_torque_from_object(obj)
// / contact_from_object(obj) hand a script the sensors the play session is
// actually running for that object. The point is the NOISE: ground truth is
// already one call away (articulation_from_object gives exact joint positions,
// rigid_body_from_object exact velocities), so a controller written against
// those is a controller that has never met a sensor. These handles are the
// seeded, quantized, bias-walked pipeline the Sensors panel plots and the CSV
// recorder capture — the same numbers, read by a control loop instead of a plot.
//
// Four things about the contract, stated once:
//
//   * A sensor exists only DURING Play. An authored SensorConfig is userData
//     until the sensor session builds from it, so these functions return None
//     outside Play (and for an object carrying no sensor of that kind). Same
//     difference from spline_from_object as the physics handles have.
//   * A handle is TIED to the play session that produced it. Stop drops every
//     entry, so a handle kept across a stop/play raises rather than reading
//     freed memory. Ask again after each start.
//   * The lookup walks UP the scene graph, like PhysxWorld::findActor: a script
//     on a child of an instrumented link still finds the sensor measuring it.
//   * A handle reads the session's RETAINED copies, never a live sensor and
//     never PhysX state. That is not an optimisation, it is what makes the
//     handle safe: drain() empties a sensor's ring, so a second drainer would
//     starve the panel's plots and the recording — and physics stops FIRST, so
//     anything touching the sensors during a script's stop() would be reading
//     through a torn-down SDK. See SensorPlaySession::History.
//
// Vision sensors (depth, lidar) are deliberately absent: a scan is a cloud of
// tens of thousands of points, which wants a buffer protocol rather than a
// per-sample handle. Proprioceptive only.
//
// Compiled only where the PhysX SDK was found — the body and joint sensors do
// not run without it — and registers into the same threepp.editor submodule
// bind_editor.cpp creates, so a build without PhysX simply does not have these
// names rather than having ones that always answer None.

#include "bindings.hpp"

// Readings come back as optionals and lists.
#include <pybind11/stl.h>

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // bindings.hpp puts the alias inside threepp_py; the helpers below live out
    // here, where the handle types are.
    namespace py = pybind11;

    using Session = editor::SensorPlaySession;
    using Entry = Session::Entry;

    // The same weak token bind_editor_physics.cpp's handles carry, for the same
    // reason: a handle must not keep a stopped session's entries alive, and must
    // not silently read a dead one. Duplicated rather than shared because both
    // live in anonymous namespaces of their own translation unit.
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


    // What the four handles share: which entry they read, whether the session
    // that made them is still alive, and how far THIS handle has read.
    //
    // The cursor is per-handle by design. Two scripts on the same sensor must
    // each see every measurement — a shared read position would have them steal
    // samples from one another, which is the exact failure the session's single
    // drain() rule exists to prevent one level down.
    class SensorView {

    public:
        SensorView(std::shared_ptr<Object3D> object, const Entry* entry, Lifetime lifetime,
                   const char* what, std::size_t cursor)
            : object_(std::move(object)), entry_(entry), lifetime_(std::move(lifetime)),
              what_(what), label_(entry->label), cursor_(cursor) {}

        [[nodiscard]] bool valid() const { return lifetime_.alive(); }

        [[nodiscard]] std::shared_ptr<Object3D> object() const { return object_; }

    protected:
        // `entry_` points into the session's own list, cleared by stop() — the
        // lifetime gate is what makes dereferencing it safe, the same contract
        // the physics handles' raw actor pointers have.
        template<class T>
        [[nodiscard]] std::optional<T> newest(const Session::History<T> Entry::* field) const {

            lifetime_.require(what_);
            const auto& history = entry_->*field;
            if (history.empty()) return std::nullopt;
            return history.newest();
        }

        template<class T>
        [[nodiscard]] std::vector<T> readNew(const Session::History<T> Entry::* field) {

            lifetime_.require(what_);
            std::vector<T> out;
            cursor_ = (entry_->*field).since(cursor_, out);
            return out;
        }

        // The authored label, cached: repr() has to work after the entry is gone.
        [[nodiscard]] const std::string& label() const { return label_; }

    private:
        std::shared_ptr<Object3D> object_;
        const Entry* entry_;
        Lifetime lifetime_;
        const char* what_;
        std::string label_;
        std::size_t cursor_;
    };


    class ImuHandle: public SensorView {

    public:
        ImuHandle(std::shared_ptr<Object3D> object, const Entry* entry, Lifetime lifetime)
            : SensorView(std::move(object), entry, std::move(lifetime), "imu",
                         entry->imuReadings.total()) {}

        [[nodiscard]] std::optional<Session::ImuReading> latest() const {

            return newest(&Entry::imuReadings);
        }

        [[nodiscard]] std::vector<Session::ImuReading> readNew() {

            return SensorView::readNew(&Entry::imuReadings);
        }

        [[nodiscard]] std::string repr() const {

            if (!valid()) return "<Imu (session stopped)>";
            return "<Imu '" + label() + "'>";
        }
    };


    class EncoderHandle: public SensorView {

    public:
        EncoderHandle(std::shared_ptr<Object3D> object, const Entry* entry, Lifetime lifetime)
            : SensorView(std::move(object), entry, std::move(lifetime), "encoder",
                         entry->encoderReadings.total()),
              joint_(entry->config.joint) {}

        // Which DOF this encoder measures. Not decoration: an all-joints entry
        // becomes one encoder per joint, and this is how a script tells them
        // apart once it holds them.
        [[nodiscard]] const std::string& joint() const { return joint_; }

        [[nodiscard]] std::optional<Session::EncoderReading> latest() const {

            return newest(&Entry::encoderReadings);
        }

        [[nodiscard]] std::vector<Session::EncoderReading> readNew() {

            return SensorView::readNew(&Entry::encoderReadings);
        }

        [[nodiscard]] std::string repr() const {

            if (!valid()) return "<Encoder (session stopped)>";
            return "<Encoder '" + label() + "' joint=\"" + joint_ + "\">";
        }

    private:
        std::string joint_;
    };


    class ForceTorqueHandle: public SensorView {

    public:
        ForceTorqueHandle(std::shared_ptr<Object3D> object, const Entry* entry, Lifetime lifetime)
            : SensorView(std::move(object), entry, std::move(lifetime), "force/torque sensor",
                         entry->wrenchReadings.total()),
              joint_(entry->config.joint) {}

        [[nodiscard]] const std::string& joint() const { return joint_; }

        [[nodiscard]] std::optional<Session::WrenchReading> latest() const {

            return newest(&Entry::wrenchReadings);
        }

        [[nodiscard]] std::vector<Session::WrenchReading> readNew() {

            return SensorView::readNew(&Entry::wrenchReadings);
        }

        [[nodiscard]] std::string repr() const {

            if (!valid()) return "<ForceTorque (session stopped)>";
            return "<ForceTorque '" + label() + "' joint=\"" + joint_ + "\">";
        }

    private:
        std::string joint_;
    };


    class ContactHandle: public SensorView {

    public:
        ContactHandle(std::shared_ptr<Object3D> object, const Entry* entry, Lifetime lifetime)
            : SensorView(std::move(object), entry, std::move(lifetime), "contact sensor",
                         entry->contactReadings.total()) {}

        [[nodiscard]] std::optional<Session::ContactReading> latest() const {

            return newest(&Entry::contactReadings);
        }

        [[nodiscard]] std::vector<Session::ContactReading> readNew() {

            return SensorView::readNew(&Entry::contactReadings);
        }

        [[nodiscard]] std::string repr() const {

            if (!valid()) return "<Contact (session stopped)>";
            return "<Contact '" + label() + "'>";
        }
    };


    // The sensor session currently playing, or nullptr outside Play.
    Session* playing() { return Session::active(); }

    // The one-sensor lookup the IMU, contact and force/torque handles share.
    // Only the encoder can legitimately resolve to several, so anything else
    // taking the first match is not a silent choice — it is the only one.
    const Entry* soleSensor(const Object3D* object, editor::SensorConfig::Type type) {

        auto* session = playing();
        if (!session || !object) return nullptr;
        const auto found = session->findSensors(object, type);
        return found.empty() ? nullptr : found.front();
    }

    // A handle onto `entry`, sited on the object that AUTHORED the sensor rather
    // than on whichever descendant asked. A script that walked up to find its
    // link's IMU wants the link back, not the mesh it happened to sit on.
    template<class Handle>
    py::object handleFor(const Entry* entry) {

        return py::cast(std::make_shared<Handle>(entry->node->shared_from_this(), entry,
                                                 Lifetime{playing()->lifetime()}));
    }

}// namespace

namespace threepp_py {

    void init_editor_sensors(py::module_& m) {

        // bind_editor.cpp made the submodule; this adds the sensor half to it.
        auto sub = m.attr("editor").cast<py::module_>();

        // --- readings --------------------------------------------------------
        //
        // Value types, so a script can stash one and keep it after the session
        // that measured it is gone. Every field is post-noise: this is what the
        // sensor reported, not what the simulation knows.

        py::class_<Session::ImuReading>(
                sub, "ImuSample",
                "One IMU measurement, as the sensor reported it (noise, bias and all).")
                .def_readonly("time", &Session::ImuReading::t, "Sim time of the measurement (s).")
                .def_readonly("angular_velocity", &Session::ImuReading::angularVelocity,
                              "Gyro reading in rad/s, in the sensor's own frame.")
                .def_readonly("acceleration", &Session::ImuReading::acceleration,
                              "Accelerometer reading in m/s^2, in the sensor's own frame. SPECIFIC "
                              "FORCE, so a level sensor at rest reads +9.81 on its up axis and one "
                              "in free fall reads ~0.")
                .def("__repr__", [](const Session::ImuReading& r) {
                    return "<ImuSample t=" + std::to_string(r.t) + ">";
                });

        py::class_<Session::EncoderReading>(
                sub, "EncoderSample",
                "One joint-encoder reading: quantized to whole ticks and noise-corrupted, which is "
                "why a controller tuned on it survives contact with hardware.")
                .def_readonly("time", &Session::EncoderReading::t, "Sim time of the reading (s).")
                .def_readonly("position", &Session::EncoderReading::position,
                              "Joint position: radians for a revolute joint, metres for a prismatic one.")
                .def_readonly("velocity", &Session::EncoderReading::velocity,
                              "Joint velocity, differentiated from the QUANTIZED position - so it "
                              "chatters at standstill, exactly as a real one does.")
                .def("__repr__", [](const Session::EncoderReading& r) {
                    return "<EncoderSample t=" + std::to_string(r.t) +
                           " pos=" + std::to_string(r.position) + ">";
                });

        py::class_<Session::WrenchReading>(
                sub, "WrenchSample",
                "One six-component load-cell reading, in the measured joint's child frame.")
                .def_readonly("time", &Session::WrenchReading::t, "Sim time of the reading (s).")
                .def_readonly("force", &Session::WrenchReading::force, "Force in newtons.")
                .def_readonly("torque", &Session::WrenchReading::torque, "Torque in newton-metres.")
                .def("__repr__", [](const Session::WrenchReading& r) {
                    return "<WrenchSample t=" + std::to_string(r.t) + ">";
                });

        py::class_<Session::ContactReading>(
                sub, "ContactSample", "One contact reading: the touch latch and the force behind it.")
                .def_readonly("time", &Session::ContactReading::t, "Sim time of the reading (s).")
                .def_readonly("touching", &Session::ContactReading::touching,
                              "Latched touch state. Stays True while resting on something, including "
                              "after the contact pair falls asleep and `force` goes quiet - this is "
                              "the channel a foot-down check wants.")
                .def_readonly("force", &Session::ContactReading::force,
                              "Mean contact force over the interval (N). Zero while the pair sleeps, "
                              "even though the touch is real.")
                .def("__repr__", [](const Session::ContactReading& r) {
                    return std::string("<ContactSample t=") + std::to_string(r.t) +
                           " touching=" + (r.touching ? "True" : "False") + ">";
                });

        // --- handles ---------------------------------------------------------

        py::class_<ImuHandle, std::shared_ptr<ImuHandle>>(sub, "Imu")
                .def_property_readonly("object", &ImuHandle::object,
                                       "The scene object this sensor was authored on.")
                .def_property_readonly("valid", &ImuHandle::valid,
                                       "False once the play session that created it has stopped.")
                .def("latest", &ImuHandle::latest,
                     "The newest measurement, or None before the first one. Does not move this "
                     "handle's read cursor.")
                .def("read_new", &ImuHandle::readNew,
                     "Every measurement since this handle last read, oldest first, and advances its "
                     "cursor. A fresh handle starts empty - it reports what arrives from now on. "
                     "Each handle has its own cursor, so two of them never steal each other's "
                     "samples; falling more than 256 behind loses the oldest.")
                .def("__repr__", &ImuHandle::repr);

        py::class_<EncoderHandle, std::shared_ptr<EncoderHandle>>(sub, "Encoder")
                .def_property_readonly("object", &EncoderHandle::object,
                                       "The scene object this sensor was authored on.")
                .def_property_readonly("valid", &EncoderHandle::valid,
                                       "False once the play session that created it has stopped.")
                .def_property_readonly("joint", &EncoderHandle::joint,
                                       "The URDF joint name this encoder measures.")
                .def("latest", &EncoderHandle::latest,
                     "The newest reading, or None before the first one. Does not move this handle's "
                     "read cursor.")
                .def("read_new", &EncoderHandle::readNew,
                     "Every reading since this handle last read, oldest first, and advances its "
                     "cursor. A fresh handle starts empty.")
                .def("__repr__", &EncoderHandle::repr);

        py::class_<ForceTorqueHandle, std::shared_ptr<ForceTorqueHandle>>(sub, "ForceTorque")
                .def_property_readonly("object", &ForceTorqueHandle::object,
                                       "The scene object this sensor was authored on.")
                .def_property_readonly("valid", &ForceTorqueHandle::valid,
                                       "False once the play session that created it has stopped.")
                .def_property_readonly("joint", &ForceTorqueHandle::joint,
                                       "The URDF joint name this load cell sits in.")
                .def("latest", &ForceTorqueHandle::latest,
                     "The newest wrench, or None before the first one. Does not move this handle's "
                     "read cursor.")
                .def("read_new", &ForceTorqueHandle::readNew,
                     "Every wrench since this handle last read, oldest first, and advances its "
                     "cursor. A fresh handle starts empty.")
                .def("__repr__", &ForceTorqueHandle::repr);

        py::class_<ContactHandle, std::shared_ptr<ContactHandle>>(sub, "Contact")
                .def_property_readonly("object", &ContactHandle::object,
                                       "The scene object this sensor was authored on.")
                .def_property_readonly("valid", &ContactHandle::valid,
                                       "False once the play session that created it has stopped.")
                .def("latest", &ContactHandle::latest,
                     "The newest reading, or None before the first one. Does not move this handle's "
                     "read cursor.")
                .def("read_new", &ContactHandle::readNew,
                     "Every reading since this handle last read, oldest first, and advances its "
                     "cursor. A fresh handle starts empty.")
                .def("__repr__", &ContactHandle::repr);

        // --- lookups ---------------------------------------------------------

        sub.def(
                "imu_from_object", [](const py::handle& h) -> py::object {
                    const auto object = as_object3d(h);
                    const auto* entry = soleSensor(object.get(), editor::SensorConfig::Type::Imu);
                    if (!entry) return py::none();
                    return handleFor<ImuHandle>(entry);
                },
                py::arg("object"),
                "The live IMU authored on `object`, or None when Play is not running or no IMU "
                "measures it. The lookup walks up the scene graph, so a script on a child finds "
                "the sensor on its link.");

        sub.def(
                "contact_from_object", [](const py::handle& h) -> py::object {
                    const auto object = as_object3d(h);
                    const auto* entry = soleSensor(object.get(), editor::SensorConfig::Type::Contact);
                    if (!entry) return py::none();
                    return handleFor<ContactHandle>(entry);
                },
                py::arg("object"),
                "The live contact sensor authored on `object`, or None when Play is not running or "
                "none measures it.");

        sub.def(
                "force_torque_from_object", [](const py::handle& h) -> py::object {
                    const auto object = as_object3d(h);
                    const auto* entry =
                            soleSensor(object.get(), editor::SensorConfig::Type::ForceTorque);
                    if (!entry) return py::none();
                    return handleFor<ForceTorqueHandle>(entry);
                },
                py::arg("object"),
                "The live force/torque sensor authored on `object`, or None when Play is not "
                "running or none measures it. A load cell sits in ONE joint, so there is never "
                "more than one to choose between.");

        sub.def(
                "encoder_from_object",
                [](const py::handle& h, const std::optional<std::string>& joint) -> py::object {
                    const auto object = as_object3d(h);
                    auto* session = playing();
                    if (!object || !session) return py::none();
                    const auto found =
                            session->findSensors(object.get(), editor::SensorConfig::Type::Encoder);
                    if (found.empty()) return py::none();

                    if (joint) {
                        for (const auto* entry : found) {
                            if (entry->config.joint == *joint) return handleFor<EncoderHandle>(entry);
                        }
                        // Naming a joint that is not measured is a typo, not an
                        // absence: answering None would look exactly like "not
                        // playing yet" and cost an afternoon.
                        std::string names;
                        for (const auto* entry : found) {
                            names += (names.empty() ? "" : ", ") + entry->config.joint;
                        }
                        throw std::runtime_error("encoder_from_object: no encoder measures joint \"" +
                                                 *joint + "\" here - live joints are: " + names);
                    }

                    if (found.size() > 1) {
                        // The all-joints fan-out: one authored entry, N live
                        // encoders. Picking one silently would hand back an
                        // arbitrary joint that happens to answer.
                        std::string names;
                        for (const auto* entry : found) {
                            names += (names.empty() ? "" : ", ") + entry->config.joint;
                        }
                        throw std::runtime_error(
                                "encoder_from_object: " + std::to_string(found.size()) +
                                " encoders measure this object (" + names +
                                ") - pass joint=\"name\" to pick one, or use encoders_from_object()");
                    }
                    return handleFor<EncoderHandle>(found.front());
                },
                py::arg("object"), py::arg("joint") = py::none(),
                "The live joint encoder authored on `object`, or None when Play is not running or "
                "none measures it. An encoder authored for All joints becomes one live encoder per "
                "DOF, so pass joint=\"name\" to pick one; with no joint this raises rather than "
                "guessing when more than one answers.");

        sub.def(
                "encoders_from_object", [](const py::handle& h) -> std::vector<py::object> {
                    std::vector<py::object> out;
                    const auto object = as_object3d(h);
                    auto* session = playing();
                    if (!object || !session) return out;
                    for (const auto* entry :
                         session->findSensors(object.get(), editor::SensorConfig::Type::Encoder)) {
                        out.push_back(handleFor<EncoderHandle>(entry));
                    }
                    return out;
                },
                py::arg("object"),
                "Every live joint encoder authored on `object`, in the articulation's DOF order - "
                "the whole-robot joint state an All-joints encoder fans out into. Empty outside "
                "Play. Read `joint` on each to know which DOF it is.");
    }

}// namespace threepp_py
