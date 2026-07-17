// Proprioceptive sensors — the scene-graph-attached, physics-driven sensor
// suite (extras/sensors), exposed to Python.
//
// Compiled unconditionally; the body is active only when threepp can see the
// omniverse-physx-sdk (THREEPP_PY_HAS_PHYSX, defined by python/CMakeLists).
// Otherwise init_sensors is a no-op — the sensors are driven by PhysxWorld's
// step loop, so they only exist in a PhysX-enabled build.
//
// Scope (phase 1): NoiseModel, Imu + ImuSample. The world-side hooks
// (register_sensor / unregister_sensor / sim_time) are added to PhysxWorld in
// bind_physx.cpp so all PhysxWorld methods live together.
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_PHYSX

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/Imu.hpp"
#include "threepp/extras/sensors/Sensor.hpp"
#include "threepp/math/Vector3.hpp"

#include <cstring>
#include <optional>
#include <vector>

namespace py = pybind11;
using namespace threepp;

namespace threepp_py {

    void init_sensors(py::module_& m) {

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
        py::class_<Imu>(m, "Imu",
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
    }

}// namespace threepp_py

#else// THREEPP_PY_HAS_PHYSX not defined — sensors need the PhysX step loop

namespace threepp_py {
    void init_sensors(py::module_& m) { (void) m; }
}// namespace threepp_py

#endif
