// The sensor suite (extras/sensors), exposed to Python.
//
// Two halves:
//   init_sensor_base — the PhysX-free contract: Sensor, NoiseModel,
//     RangeNoiseModel. Always compiled, and registered EARLY (before
//     init_render) because the vision sensors bound elsewhere — DepthSensor in
//     bind_render.cpp — derive from Sensor, and pybind requires a base class to
//     be registered before any class that names it.
//   init_sensors — the concrete proprioceptive sensors, which are driven by
//     PhysxWorld's step loop and so only exist in a PhysX-enabled build
//     (THREEPP_PY_HAS_PHYSX, defined by python/CMakeLists). A no-op otherwise.
//
// Scope: Sensor, NoiseModel, RangeNoiseModel, Imu + ImuSample, JointEncoder +
// JointSample, ContactSensor + ContactSample, ForceTorqueSensor +
// WrenchSample. The world-side hooks (register_sensor / unregister_sensor /
// sim_time) are added to PhysxWorld in bind_physx.cpp so all PhysxWorld methods
// live together.
#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "threepp/extras/sensors/Sensor.hpp"
#include "threepp/math/Vector3.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace threepp_py {

    void init_sensor_base(pybind11::module_& m) {
        namespace py = pybind11;
        using namespace threepp;

        // --- Sensor (abstract base) ----------------------------------------
        // Bound so PhysxWorld.register_sensor can take any sensor rather than
        // naming each concrete type — the list would otherwise have to be kept in
        // sync by hand every time a sensor is added, which is how JointEncoder
        // and ContactSensor would have silently been unregisterable from Python.
        // Safe to bind as a base here because Sensor is a NON-virtual base (see
        // bind_objects.cpp for why threepp's virtual Object3D bases are not).
        // shared_ptr holder to match Object3D's: the vision sensors are both an
        // Object3D and a Sensor, and pybind requires one holder type across a
        // whole hierarchy.
        py::class_<Sensor, std::shared_ptr<Sensor>>(
                m, "Sensor",
                "Abstract base of the sensor suite. Register a concrete sensor with "
                "PhysxWorld.register_sensor to have it sampled from the step loop.")
                .def_property("rate_hz", &Sensor::rateHz, &Sensor::setRateHz,
                              "Target sample rate (Hz); 0 = every physics substep.")
                .def_property("sim_time", &Sensor::simTime, &Sensor::setSimTime,
                              "The sensor's clock (s) — the time base every measurement is stamped "
                              "with. Latched from the world automatically while registered with a "
                              "PhysxWorld; drive it yourself (advance_clock / sim_time = t) for a "
                              "sensor pulled from a render loop. Always sim time, never wall time, "
                              "so a replayed run reproduces its timestamps.")
                .def("advance_clock", &Sensor::advanceClock, py::arg("dt"),
                     "Advance the sensor clock by dt seconds.");

        // --- NoiseModel ----------------------------------------------------
        py::class_<NoiseModel>(m, "NoiseModel",
                               "Per-axis Gaussian noise config shared by every sensor. Densities are "
                               "continuous-time so the noise is invariant to sample rate: "
                               "white_noise_density [X/sqrt(Hz)] -> per-sample stddev density/sqrt(dt); "
                               "random_walk [X/(s*sqrt(Hz))] -> bias increment stddev random_walk*sqrt(dt); "
                               "constant_bias [X] is a fixed turn-on offset. Same seed + call sequence is "
                               "deterministic. All-zero = a perfect sensor.")
                .def(py::init([](const Vector3& white, const Vector3& walk, const Vector3& bias, std::uint64_t seed) {
                         NoiseModel n;
                         n.whiteNoiseDensity = white;
                         n.randomWalk = walk;
                         n.constantBias = bias;
                         n.seed = seed;
                         return n;
                     }),
                     py::arg("white_noise_density") = Vector3(0.f, 0.f, 0.f),
                     py::arg("random_walk") = Vector3(0.f, 0.f, 0.f),
                     py::arg("constant_bias") = Vector3(0.f, 0.f, 0.f),
                     py::arg("seed") = 0)
                .def_readwrite("white_noise_density", &NoiseModel::whiteNoiseDensity)
                .def_readwrite("random_walk", &NoiseModel::randomWalk)
                .def_readwrite("constant_bias", &NoiseModel::constantBias)
                .def_readwrite("seed", &NoiseModel::seed)
                .def("__repr__", [](const NoiseModel& n) {
                    return "<threepp.NoiseModel seed=" + std::to_string(n.seed) + ">";
                });

        // --- RangeNoiseModel -----------------------------------------------
        py::class_<RangeNoiseModel>(m, "RangeNoiseModel",
                                    "Range noise for a ranging sensor (LIDAR / depth camera). Per RETURN, "
                                    "not per second: sigma = hypot(stddev, r * stddev_per_metre) metres, "
                                    "plus a fixed `bias`. `seed` makes a scan reproducible — same seed and "
                                    "same beam order gives the same cloud on every run and machine. "
                                    "All-zero = a perfect sensor (clean ranges pass through untouched).")
                .def(py::init([](float stddev, float stddev_per_metre, float bias, std::uint64_t seed) {
                         RangeNoiseModel n;
                         n.stddev = stddev;
                         n.stddevPerMetre = stddev_per_metre;
                         n.bias = bias;
                         n.seed = seed;
                         return n;
                     }),
                     py::arg("stddev") = 0.f, py::arg("stddev_per_metre") = 0.f,
                     py::arg("bias") = 0.f, py::arg("seed") = 0)
                .def_readwrite("stddev", &RangeNoiseModel::stddev, "Constant sigma [m].")
                .def_readwrite("stddev_per_metre", &RangeNoiseModel::stddevPerMetre,
                               "Range-proportional sigma [m/m], added in quadrature.")
                .def_readwrite("bias", &RangeNoiseModel::bias, "Fixed offset [m]; positive reads long.")
                .def_readwrite("seed", &RangeNoiseModel::seed)
                .def("__repr__", [](const RangeNoiseModel& n) {
                    return "<threepp.RangeNoiseModel stddev=" + std::to_string(n.stddev) +
                           " seed=" + std::to_string(n.seed) + ">";
                });
    }

}// namespace threepp_py

#ifdef THREEPP_PY_HAS_PHYSX

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/sensors/ContactSensor.hpp"
#include "threepp/extras/sensors/ForceTorqueSensor.hpp"
#include "threepp/extras/sensors/Imu.hpp"
#include "threepp/extras/sensors/JointEncoder.hpp"
#include "threepp/extras/sensors/Sensor.hpp"
#include "threepp/math/Vector3.hpp"

#include <cstring>
#include <optional>
#include <vector>

namespace py = pybind11;
using namespace threepp;

namespace threepp_py {

    void init_sensors(py::module_& m) {

        // --- ImuSample -----------------------------------------------------
        py::class_<ImuSample>(m, "ImuSample",
                              "One IMU measurement. t: sim time (s). angular_velocity: rad/s. "
                              "linear_acceleration: specific force (m/s^2). Both in the sensor frame.")
                .def_readonly("t", &ImuSample::t)
                .def_readonly("angular_velocity", &ImuSample::angularVelocity)
                .def_readonly("linear_acceleration", &ImuSample::linearAcceleration)
                .def("__repr__", [](const ImuSample& s) {
                    return "<threepp.ImuSample t=" + std::to_string(s.t) + ">";
                });

        // --- Imu -----------------------------------------------------------
        py::class_<ImuModel>(m, "ImuModel",
                             "A named IMU noise model from a datasheet: `gyro` and `accel` NoiseModels "
                             "(assign them to Imu.gyro_noise / Imu.accel_noise, then call reset()). Only "
                             "the white-noise densities are datasheet figures; the parts publish no rate "
                             "random walk in NoiseModel's units, so presets leave random_walk at zero. "
                             "Imu's defaults ARE ImuModel.icm42688p().")
                .def(py::init<>())
                .def_readonly("name", &ImuModel::name)
                .def_readwrite("gyro", &ImuModel::gyro, "Gyroscope NoiseModel (rad/s units).")
                .def_readwrite("accel", &ImuModel::accel, "Accelerometer NoiseModel (m/s^2 units).")
                .def_static("icm42688p", &ImuModel::ICM42688P,
                            "TDK InvenSense ICM-42688-P: gyro 2.8 mdps/sqrt(Hz), accel 70 ug/sqrt(Hz) "
                            "(datasheet DS-000347).")
                .def("__repr__", [](const ImuModel& m) { return std::string("<threepp.ImuModel ") + m.name + ">"; });

        py::class_<Imu, Sensor, std::shared_ptr<Imu>>(m, "Imu",
                        "Gyroscope + accelerometer attached to an Object3D. Its measurement frame is "
                        "that node's world frame; register it with a PhysxWorld (world.register_sensor) "
                        "AFTER adding the body it rides. Each substep it measures the body's angular "
                        "velocity (rad/s) and the specific force at the sensor point (m/s^2, gravity "
                        "reaction + lever-arm terms), in the sensor frame. A level body at rest reads "
                        "accel (0, +9.81, 0); free fall reads ~0.")
                .def(py::init([](const py::handle& node, double rate_hz, std::size_t buffer_capacity) {
                         auto obj = as_object3d(node);
                         return new Imu(*obj, rate_hz, buffer_capacity);
                     }),
                     py::arg("node"), py::arg("rate_hz") = 0.0, py::arg("buffer_capacity") = 2048,
                     py::keep_alive<1, 2>(),// the Imu keeps its attachment node alive
                     "Attach an IMU to `node` (its world frame is the sensor frame). rate_hz=0 "
                     "samples every physics substep; buffer_capacity is the ring depth (oldest "
                     "dropped on overflow).")
                .def_readwrite("gyro_noise", &Imu::gyroNoise,
                               "NoiseModel for the gyroscope (rad/s units). Change then call reset().")
                .def_readwrite("accel_noise", &Imu::accelNoise,
                               "NoiseModel for the accelerometer (m/s^2 units). Change then call reset().")
                .def_property("rate_hz", &Imu::rateHz, &Imu::setRateHz)
                .def("reset", &Imu::reset,
                     "Re-arm after an episode reset or a noise change: clears the finite-difference "
                     "history + buffer and re-seeds the noise from the current configs.")
                .def_property_readonly("available", &Imu::available, "Number of buffered samples.")
                .def_property_readonly("attached", &Imu::attached,
                                       "True while bound to a live rigid body. False before registering, "
                                       "after unregistering, and after the body was removed from the world "
                                       "(remove_actor) — sampling is a silent no-op in all three cases.")
                .def("latest", &Imu::latest,
                     "The most recent ImuSample, or None. Survives drain().")
                .def("drain",
                     [](Imu& imu) {
                         std::vector<ImuSample> out;
                         imu.drain(out);
                         return out;
                     },
                     "Move all buffered ImuSamples (oldest-first) out as a list; empties the buffer.")
                .def("drain_array",
                     [](Imu& imu) {
                         std::vector<ImuSample> out;
                         imu.drain(out);
                         const auto n = static_cast<py::ssize_t>(out.size());
                         py::array_t<double> arr({n, static_cast<py::ssize_t>(7)});
                         auto* p = arr.mutable_data();
                         for (std::size_t i = 0; i < out.size(); ++i) {
                             const auto& s = out[i];
                             p[i * 7 + 0] = s.t;
                             p[i * 7 + 1] = s.angularVelocity.x;
                             p[i * 7 + 2] = s.angularVelocity.y;
                             p[i * 7 + 3] = s.angularVelocity.z;
                             p[i * 7 + 4] = s.linearAcceleration.x;
                             p[i * 7 + 5] = s.linearAcceleration.y;
                             p[i * 7 + 6] = s.linearAcceleration.z;
                         }
                         return arr;
                     },
                     "Drain all buffered samples as a (N, 7) float64 numpy array with columns "
                     "[t, gx, gy, gz, ax, ay, az]. Empties the buffer.");

        // --- JointSample / JointEncoder ------------------------------------
        py::class_<JointSample>(m, "JointSample",
                                "One encoder reading. Units follow the joint: rad and rad/s for a "
                                "revolute joint, m and m/s for a prismatic one.")
                .def_readonly("t", &JointSample::t)
                .def_readonly("position", &JointSample::position)
                .def_readonly("velocity", &JointSample::velocity)
                .def("__repr__", [](const JointSample& s) {
                    return "<threepp.JointSample t=" + std::to_string(s.t) +
                           " pos=" + std::to_string(s.position) + ">";
                });

        py::class_<JointEncoder, Sensor, std::shared_ptr<JointEncoder>>(m, "JointEncoder",
                                 "Joint position/velocity encoder on an articulation link's inbound "
                                 "joint. Adds what a real encoder has and Articulation.joint_positions "
                                 "does not: tick quantization, noise, rate gating and buffering.")
                .def(py::init([](const py::handle& node, const ArticulationLink& link,
                                 double rate_hz, std::size_t buffer_capacity) {
                         // as_object3d, not a plain Object3D* cast: the node is normally a
                         // Mesh, and pybind mis-adjusts pointers across threepp's VIRTUAL
                         // Object3D base (see bind_objects.cpp).
                         auto obj = as_object3d(node);
                         return new JointEncoder(*obj, link, rate_hz, buffer_capacity);
                     }),
                     py::arg("node"), py::arg("link"), py::arg("rate_hz") = 0.0,
                     py::arg("buffer_capacity") = 2048,
                     py::keep_alive<1, 2>(),// the encoder keeps its attachment node alive
                     "Attach to `node` (normally the mesh bound to the joint's child link) and "
                     "measure `link`'s inbound joint. Raises if `link` is the root.")
                .def(py::init([](const py::handle& node, const Joint& joint,
                                 double rate_hz, std::size_t buffer_capacity) {
                         auto obj = as_object3d(node);// see the ArticulationLink ctor above
                         return new JointEncoder(*obj, joint, rate_hz, buffer_capacity);
                     }),
                     py::arg("node"), py::arg("joint"), py::arg("rate_hz") = 0.0,
                     py::arg("buffer_capacity") = 2048,
                     py::keep_alive<1, 2>(),// the attachment node
                     py::keep_alive<1, 3>(),// and the Joint it reads every sample
                     "The same encoder on a plain Joint: the coordinate is the joint's scalar "
                     "axis (twist for a hinge, displacement for a slider, anchor distance for "
                     "a tether).")
                .def_readwrite("resolution", &JointEncoder::resolution,
                               "Quantization step: rad (revolute) or m (prismatic) per tick. "
                               "0 = ideal continuous encoder.")
                .def("set_counts_per_rev", &JointEncoder::setCountsPerRev, py::arg("counts"),
                     "Set `resolution` from a rotary encoder's counts per revolution.")
                .def_readwrite("position_noise", &JointEncoder::positionNoise,
                               "Position noise; only the X component of each Vector3 is used.")
                .def_readwrite("velocity_noise", &JointEncoder::velocityNoise,
                               "Velocity noise; applied only when differentiate_velocity is False.")
                .def_readwrite("differentiate_velocity", &JointEncoder::differentiateVelocity,
                               "True (default): differentiate the quantized, noisy position, as a real "
                               "encoder-fed controller does. False: report the simulator's true velocity.")
                .def_property("rate_hz", &JointEncoder::rateHz, &JointEncoder::setRateHz)
                .def("reset", &JointEncoder::reset,
                     "Re-arm after an episode reset: clear the buffer and differentiation history "
                     "and re-seed the noise.")
                .def_property_readonly("available", &JointEncoder::available, "Number of buffered samples.")
                .def("latest", &JointEncoder::latest, "The most recent JointSample, or None. Survives drain().")
                .def("drain",
                     [](JointEncoder& enc) {
                         std::vector<JointSample> out;
                         enc.drain(out);
                         return out;
                     },
                     "Move all buffered JointSamples (oldest-first) out as a list; empties the buffer.")
                .def("drain_array",
                     [](JointEncoder& enc) {
                         std::vector<JointSample> out;
                         enc.drain(out);
                         const auto n = static_cast<py::ssize_t>(out.size());
                         py::array_t<double> arr({n, static_cast<py::ssize_t>(3)});
                         auto* p = arr.mutable_data();
                         for (std::size_t i = 0; i < out.size(); ++i) {
                             p[i * 3 + 0] = out[i].t;
                             p[i * 3 + 1] = out[i].position;
                             p[i * 3 + 2] = out[i].velocity;
                         }
                         return arr;
                     },
                     "Drain all buffered samples as a (N, 3) float64 numpy array with columns "
                     "[t, position, velocity]. Empties the buffer.");

        // --- ContactPoint / ContactSample / ContactSensor -------------------
        py::class_<ContactPoint>(m, "ContactPoint", "One manifold point, world space.")
                .def_readonly("position", &ContactPoint::position)
                .def_readonly("normal", &ContactPoint::normal,
                              "Unit normal pointing INTO the sensor's body.")
                .def_readonly("impulse", &ContactPoint::impulse,
                              "Normal impulse magnitude at this point over the substep (N*s).");

        py::class_<ContactSample>(m, "ContactSample", "One contact reading.")
                .def_readonly("t", &ContactSample::t)
                .def_readonly("in_contact", &ContactSample::inContact,
                              "Latched touch state — stays true while resting, including after the "
                              "contact pair falls asleep and stops producing points.")
                .def_readonly("touch_began", &ContactSample::touchBegan)
                .def_readonly("touch_ended", &ContactSample::touchEnded)
                .def_readonly("force", &ContactSample::force,
                              "Mean contact force over the interval (N). Zero while asleep.")
                .def_readonly("observed_points", &ContactSample::observedPoints,
                              "Manifold points seen this interval, before the report cap.")
                .def_property_readonly("points",
                                       [](const ContactSample& s) {
                                           std::vector<ContactPoint> out(
                                                   s.points.begin(),
                                                   s.points.begin() + static_cast<std::ptrdiff_t>(s.pointCount));
                                           return out;
                                       },
                                       "Reported manifold points (capped; see observed_points).")
                .def("__repr__", [](const ContactSample& s) {
                    return "<threepp.ContactSample t=" + std::to_string(s.t) +
                           " in_contact=" + (s.inContact ? "True" : "False") + ">";
                });

        py::class_<ContactSensor, Sensor, std::shared_ptr<ContactSensor>>(m, "ContactSensor",
                                  "Reports whether the attached body is touching anything, where, and "
                                  "how hard — a bumper, a foot-contact switch, a grasp detector.")
                .def(py::init([](const py::handle& node, double rate_hz, std::size_t buffer_capacity) {
                         auto obj = as_object3d(node);// see the JointEncoder ctor above
                         return new ContactSensor(*obj, rate_hz, buffer_capacity);
                     }),
                     py::arg("node"), py::arg("rate_hz") = 0.0, py::arg("buffer_capacity") = 256,
                     py::keep_alive<1, 2>(),
                     "Attach to `node`; the rigid body in its ancestry is the one whose contacts "
                     "are reported.")
                .def_property("rate_hz", &ContactSensor::rateHz, &ContactSensor::setRateHz)
                .def_property_readonly("attached", &ContactSensor::attached,
                                       "True while bound to a live rigid body.")
                .def_property_readonly("in_contact", &ContactSensor::inContact,
                                       "Current latched touch state — the cheap read for a control loop "
                                       "that only wants a foot-down boolean.")
                .def_property_readonly("touch_count", &ContactSensor::touchCount,
                                       "How many distinct bodies are currently being touched.")
                .def("reset", &ContactSensor::reset, "Re-arm: clear the buffer, latch and pending observations.")
                .def_property_readonly("available", &ContactSensor::available, "Number of buffered samples.")
                .def("latest", &ContactSensor::latest, "The most recent ContactSample, or None. Survives drain().")
                .def("drain", [](ContactSensor& s) {
                    std::vector<ContactSample> out;
                    s.drain(out);
                    return out;
                },
                     "Move all buffered ContactSamples (oldest-first) out as a list; empties the buffer.");

        // --- WrenchSample / ForceTorqueSensor -------------------------------
        py::class_<WrenchSample>(m, "WrenchSample",
                                 "One six-component wrench reading, in the measured joint's child "
                                 "frame. force is N, torque is N*m.")
                .def_readonly("t", &WrenchSample::t)
                .def_readonly("force", &WrenchSample::force)
                .def_readonly("torque", &WrenchSample::torque)
                .def("__repr__", [](const WrenchSample& s) {
                    return "<threepp.WrenchSample t=" + std::to_string(s.t) + ">";
                });

        py::class_<ForceTorqueSensor, Sensor, std::shared_ptr<ForceTorqueSensor>>(
                m, "ForceTorqueSensor",
                "Load cell on an articulation joint: the wrench the parent link transmits to the "
                "child through their joint, as the solver computed it. The input to force control, "
                "admittance control and payload estimation.")
                .def(py::init([](const py::handle& node, Articulation& art,
                                 const ArticulationLink& link, double rate_hz,
                                 std::size_t buffer_capacity) {
                         auto obj = as_object3d(node);// see the JointEncoder ctor above
                         return new ForceTorqueSensor(*obj, art, link, rate_hz, buffer_capacity);
                     }),
                     py::arg("node"), py::arg("articulation"), py::arg("link"),
                     py::arg("rate_hz") = 0.0, py::arg("buffer_capacity") = 2048,
                     py::keep_alive<1, 2>(),// keeps the attachment node alive
                     py::keep_alive<1, 3>(),// and the articulation, whose cache it borrows
                     "Measure `link`'s inbound joint. Raises if `link` is the root, or if the "
                     "articulation has not been finalized.")
                .def(py::init([](const py::handle& node, const Joint& joint,
                                 double rate_hz, std::size_t buffer_capacity) {
                         auto obj = as_object3d(node);
                         return new ForceTorqueSensor(*obj, joint, rate_hz, buffer_capacity);
                     }),
                     py::arg("node"), py::arg("joint"), py::arg("rate_hz") = 0.0,
                     py::arg("buffer_capacity") = 2048,
                     py::keep_alive<1, 2>(),// the attachment node
                     py::keep_alive<1, 3>(),// and the Joint it reads every sample
                     "The same load cell across a plain Joint: the wrench is the solver's "
                     "constraint force on it, in world axes.")
                .def_readwrite("force_noise", &ForceTorqueSensor::forceNoise,
                               "NoiseModel for the force channel (N). Change then call reset().")
                .def_readwrite("torque_noise", &ForceTorqueSensor::torqueNoise,
                               "NoiseModel for the torque channel (N*m). Change then call reset().")
                .def("reset", &ForceTorqueSensor::reset, "Re-arm: clear the buffer and re-seed the noise.")
                .def_property_readonly("available", &ForceTorqueSensor::available,
                                       "Number of buffered samples.")
                .def("latest", &ForceTorqueSensor::latest,
                     "The most recent WrenchSample, or None. Survives drain().")
                .def("drain",
                     [](ForceTorqueSensor& ft) {
                         std::vector<WrenchSample> out;
                         ft.drain(out);
                         return out;
                     },
                     "Move all buffered WrenchSamples (oldest-first) out as a list; empties the buffer.")
                .def("drain_array",
                     [](ForceTorqueSensor& ft) {
                         std::vector<WrenchSample> out;
                         ft.drain(out);
                         const auto n = static_cast<py::ssize_t>(out.size());
                         py::array_t<double> arr({n, static_cast<py::ssize_t>(7)});
                         auto* p = arr.mutable_data();
                         for (std::size_t i = 0; i < out.size(); ++i) {
                             const auto& s = out[i];
                             p[i * 7 + 0] = s.t;
                             p[i * 7 + 1] = s.force.x;
                             p[i * 7 + 2] = s.force.y;
                             p[i * 7 + 3] = s.force.z;
                             p[i * 7 + 4] = s.torque.x;
                             p[i * 7 + 5] = s.torque.y;
                             p[i * 7 + 6] = s.torque.z;
                         }
                         return arr;
                     },
                     "Drain all buffered samples as a (N, 7) float64 numpy array with columns "
                     "[t, fx, fy, fz, tx, ty, tz]. Empties the buffer.");
    }

}// namespace threepp_py

#else// THREEPP_PY_HAS_PHYSX not defined — sensors need the PhysX step loop

namespace threepp_py {
    void init_sensors(py::module_& m) { (void) m; }
}// namespace threepp_py

#endif
