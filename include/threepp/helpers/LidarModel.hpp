#ifndef THREEPP_LIDARMODEL_HPP
#define THREEPP_LIDARMODEL_HPP

#include <vector>

namespace threepp {

    /**
     * Describes the beam pattern of a LiDAR sensor.
     *
     * Coordinate convention used by LidarSensor:
     *   azimuth  = 0   → sensor's forward direction (local -Z)
     *   azimuth  increases counter-clockwise when viewed from above (+Y)
     *   elevation = 0  → horizontal
     *   elevation > 0  → upward
     */
    struct LidarModel {

        /// Vertical angle of each laser beam in degrees.
        std::vector<float> elevationAngles;

        /// Angular step between horizontal scan positions, in degrees.
        float azimuthResolution{0.2f};

        /// Horizontal scan range [azimuthMin, azimuthMax), in degrees.
        /// Use -180 / 180 for a full 360-degree sweep.
        float azimuthMin{-180.f};
        float azimuthMax{180.f};

        // ----------------------------------------------------------------
        // Preset sensor models
        // ----------------------------------------------------------------

        /// Velodyne VLP-16: 16 beams, ±15° elevation at 2° steps, 0.2° azimuth.
        static LidarModel VLP16() {
            LidarModel m;
            m.elevationAngles.resize(16);
            for (int i = 0; i < 16; ++i)
                m.elevationAngles[i] = -15.f + i * 2.f;
            m.azimuthResolution = 0.2f;
            return m;
        }

        /// Velodyne HDL-32E: 32 beams from -30.67° to +10.67° at ~1.33° steps, 0.1° azimuth.
        static LidarModel HDL32E() {
            LidarModel m;
            m.elevationAngles.resize(32);
            for (int i = 0; i < 32; ++i)
                m.elevationAngles[i] = -30.67f + i * (10.67f - (-30.67f)) / 31.f;
            m.azimuthResolution = 0.1f;
            return m;
        }

        /// Ouster OS1-64: 64 beams, ±22.5° elevation (uniform), 0.35° azimuth.
        static LidarModel OS1_64() {
            LidarModel m;
            m.elevationAngles.resize(64);
            for (int i = 0; i < 64; ++i)
                m.elevationAngles[i] = -22.5f + i * (45.f / 63.f);
            m.azimuthResolution = 0.35f;
            return m;
        }

        /// Ouster OS0-128: 128 beams, ±45° elevation (uniform), 0.35° azimuth.
        static LidarModel OS0_128() {
            LidarModel m;
            m.elevationAngles.resize(128);
            for (int i = 0; i < 128; ++i)
                m.elevationAngles[i] = -45.f + i * (90.f / 127.f);
            m.azimuthResolution = 0.35f;
            return m;
        }
    };

}// namespace threepp

#endif//THREEPP_LIDARMODEL_HPP
