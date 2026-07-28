// Per-object sensor authoring, stored on the object itself.
//
// The editor writes this into `object.userData["sensor"]`, so a sensor rig
// travels with the scene through ObjectExporter/ObjectLoader for free — a saved
// document carries its instrumentation with no sidecar file and no schema
// extension. Same contract as PhysicsConfig (which is this file's template):
// one flat `key=value;key=value` string, because userData round-trips scalars
// only (see ObjectExporter::writeUserData).
//
// The attachment node's WORLD FRAME IS THE MEASUREMENT FRAME — that is the whole
// convention of threepp/extras/sensors/Sensor.hpp, and it is why a sensor is
// authored ON an object rather than as an object of its own: pick the link you
// want the IMU bolted to, tick a box, and the transform gizmo you already know
// aims it.
//
// EVERY key is emitted on every write, whatever the current type. Flipping the
// type combo from lidar to imu and back must not quietly reset the beam pattern
// — the settings of the type you are not looking at are still your settings.
// (This is the same lesson the soft-body parameters in PhysicsConfig carry.)
//
// Nothing here depends on PhysX, on a renderer, or on the sensor classes
// themselves. The struct is authoring data; turning it into live sensors is
// SensorPlaySession's job, and any other runtime is free to read the same field
// and build something else from it.

#ifndef THREEPP_EDITOR_SENSORCONFIG_HPP
#define THREEPP_EDITOR_SENSORCONFIG_HPP

#include <cstdint>
#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct SensorConfig {

        enum class Type {
            Imu,        // gyro + accelerometer, pushed from the physics substep
            Depth,      // pinhole depth camera, pulled from the frame loop
            Lidar,      // 360-degree ranging, pulled from the frame loop
            Encoder,    // joint position/velocity (needs an articulated robot)
            Contact,    // touch latch + contact force
            ForceTorque // 6-axis load cell (needs an articulated robot)
        };

        // LIDAR beam pattern. `Dense` is every pixel of the six cube faces — a
        // debugging visualisation rather than a real sensor; the rest are the
        // angular patterns of the parts they are named after.
        enum class Beams {
            Dense,
            VLP16,
            HDL32E,
            OS1_64,
            OS0_128
        };

        bool enabled = false;
        Type type = Type::Imu;

        // Sample/scan rate. 0 = every physics substep for a pushed sensor, every
        // frame for a pulled one.
        float rateHz = 50.f;
        // Seeds the sensor's own PRNG. Sensors are rebuilt from this config on
        // every Play, so the same seed replays the same noise — see
        // SensorPlaySession.
        int seed = 1;

        // --- Imu ------------------------------------------------------------
        // One number per channel rather than three: a spec sheet quotes one
        // density for the whole triad, and an anisotropic IMU is a domain-
        // randomization concern, not an authoring one.
        float gyroNoiseDensity = 0.005f;// rad/s/sqrt(Hz)
        float gyroRandomWalk = 4e-5f;   // rad/s^2/sqrt(Hz)
        float accelNoiseDensity = 0.06f;// m/s^2/sqrt(Hz)
        float accelRandomWalk = 4e-3f;  // m/s^3/sqrt(Hz)

        // --- Depth and Lidar (shared) ---------------------------------------
        float nearPlane = 0.1f;
        float farPlane = 30.f;
        // Per-return range noise. NOT a density: a range return's uncertainty
        // comes from one laser pulse, not from an integration interval, so it is
        // independent of the scan rate (see RangeNoiseModel).
        float rangeStddev = 0.02f;      // m
        float rangeStddevPerMetre = 0.f;// m per m of range, added in quadrature
        float rangeBias = 0.f;          // m, positive reads long

        // --- Depth ----------------------------------------------------------
        float fovY = 60.f;// degrees
        int width = 160;
        int height = 120;

        // --- Lidar ----------------------------------------------------------
        Beams beams = Beams::VLP16;
        // Cube-face resolution. Wants to be >= 90/azimuthResolution or the beam
        // table aliases; 128 keeps an editor-rate scan affordable.
        int faceSize = 128;

        // --- Encoder --------------------------------------------------------
        float encoderResolution = 0.f;// rad (or m) per tick; 0 = ideal

        // --- Contact --------------------------------------------------------
        float contactForceThreshold = 0.f;// N, for the readout's "touching" light

        static constexpr const char* userDataKey = "sensor";

        // Guard rails for the fields a runaway value would turn into a
        // multi-second stall or a gigabyte of render target.
        static constexpr int maxFaceSize = 1024;
        static constexpr int maxImageSize = 2048;

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<SensorConfig> decode(const std::string& text);

        // nullopt when the object carries no sensor entry (or an unreadable one).
        [[nodiscard]] static std::optional<SensorConfig> read(const Object3D& object);

        // Writes the entry; `enabled == false` removes it, so an object with no
        // sensor leaves no trace in the saved file.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        // True for the types that ride the physics substep (pushed), false for
        // the ones the frame loop pulls with a renderer in hand.
        [[nodiscard]] static bool isProprioceptive(Type type);
        [[nodiscard]] static bool isVision(Type type);

        // Seed for the numbered sub-stream `index`, so the gyro, the
        // accelerometer and a range channel of ONE authored sensor do not all
        // draw from the same sequence (which would correlate them) while still
        // being reproducible from the single authored seed.
        [[nodiscard]] std::uint64_t streamSeed(int index) const;

        static const char* label(Type type);
        static const char* label(Beams beams);

        bool operator==(const SensorConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SENSORCONFIG_HPP
