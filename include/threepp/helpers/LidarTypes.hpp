// Renderer-agnostic types shared by both LIDAR implementations:
//
//   - LidarSensor          (GL backend, cube-face depth raster, positions only)
//   - PathTracedLidarSensor (Vulkan backend, GPU ray tracing, full BRDF intensity)
//
// Both produce LidarReturn so downstream code (point-cloud visualisation,
// perception pipelines, recording) can consume either source uniformly.
// PathTracedLidarSensor populates every field; the raster LidarSensor fills
// only `position` / `distance` and leaves the rest at sentinel defaults
// (intensity = 0, normal = 0, hitInstanceId = -1) because the cube-face
// depth path doesn't have access to the underlying material / geometry
// data that the ray-traced path can read from the TLAS.

#ifndef THREEPP_LIDARTYPES_HPP
#define THREEPP_LIDARTYPES_HPP

#include "threepp/math/Vector3.hpp"

#include <cstdint>

namespace threepp {

    /**
     * One world-space beam. The direction must be unit length; callers in
     * possession of a sensor pose typically derive this from a LidarModel
     * beam pattern transformed through the sensor's world matrix.
     */
    struct LidarBeam {
        Vector3 origin;
        Vector3 direction;
    };

    /**
     * One LIDAR return. Sentinel values in `hitInstanceId`:
     *    >= 0   surface hit; ID indexes the renderer's per-instance
     *           MaterialDesc / GeometryDesc tables
     *    -1     miss / sub-threshold / source can't tell
     *    -2     volume scatter event (fog / haze / participating media);
     *           `position` is the scatter location, `normal` points
     *           back toward the sensor, `hitInstanceId == -2` lets the
     *           consumer distinguish a soft fog return from a real
     *           surface hit
     *
     * `returnNo > 0` is the simplest "is this a real return?" predicate.
     */
    struct LidarReturn {
        Vector3 position;       // world-space hit point / scatter point
        Vector3 normal;         // world surface normal; back-toward-sensor for fog; 0 on miss
        float   distance;       // slant range from sensor (m); 0 on miss
        float   intensity;      // normalised return strength [0, 1]; 0 on miss
        int32_t hitInstanceId;  // see above for sentinel meanings
        int32_t returnNo;       // 1 = first return; 0 = miss
    };

    /**
     * Live-tweakable parameters of the LIDAR equation. Used by the path-
     * traced sensor; the raster sensor only honours `maxRange`.
     */
    struct LidarParams {
        // Maximum slant range. Beams that hit no surface within this
        // distance produce miss results.
        float maxRange = 100.f;

        // Reference transmit power. The product
        //     laserPower · f_back(material) · cos θ · η(r) / r²
        // is multiplied by π·referenceRange² so that, at laserPower = 1,
        // a perpendicular 1.0-albedo Lambertian surface at `referenceRange`
        // reads as 1.0. Raising laserPower scales every return linearly.
        float laserPower = 1.f;
        float referenceRange = 5.f;

        // Beer-Lambert atmospheric extinction (1/m), applied over the
        // round trip. 0 = vacuum; 0.01 ≈ light haze; 0.1 ≈ heavy fog.
        float atmosphericExtinction = 0.f;

        // Returns with normalised intensity below this are dropped
        // (hitInstanceId = -1).
        float detectorThreshold = 0.005f;

        // Maximum returns emitted per beam. When > 1, the path-traced
        // backend continues the beam through transmissive surfaces
        // (water, glass, foliage) and may record subsequent hits —
        // closes the sim-to-real gap on the "second return from the
        // seafloor" / "see-through-glass" signals real LIDAR sensors
        // output. The raster cube-face LidarSensor always reports 1.
        // The result vector can be up to numBeams * maxReturns long;
        // callers filter entries with hitInstanceId < 0.
        uint32_t maxReturns = 1;

        // Stochastic samples fired per logical beam. Each sample jitters
        // the direction within a `beamDivergenceMrad`-wide cone and runs
        // an independent multi-return + delta-tracking chain, with its
        // own RNG state for the fog scatter free-flight sample. Raising
        // `samplesPerBeam` averages out the variance from fog scattering
        // and reveals the beam's true angular footprint on small
        // features. Default 1 reproduces the deterministic legacy
        // behaviour bit-for-bit (sample 0 with zero divergence).
        //
        // Result layout grows to
        //     numBeams × samplesPerBeam × maxReturns
        // indexed as `(beam * samplesPerBeam + sample) * maxReturns + slot`.
        uint32_t samplesPerBeam = 1;

        // Full-cone beam divergence in milliradians (real Velodyne /
        // Ouster sensors run 1.5–3 mrad). Only sampled when
        // `samplesPerBeam > 1`; at 1 sample/beam the direction is the
        // unjittered beam direction regardless of this value.
        float    beamDivergenceMrad = 0.f;
    };

}// namespace threepp

#endif//THREEPP_LIDARTYPES_HPP
