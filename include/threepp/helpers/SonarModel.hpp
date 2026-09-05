// Imaging-sonar model and echogram — the renderer-free half of SonarSensor.
//
// A forward-looking multibeam sonar is, geometrically, a fan of narrow
// azimuth beams that each integrate everything inside a tall vertical
// aperture, reported as echo strength per range bin. Ray-cast, that is:
//
//     for each azimuth beam:   for each sample through the aperture:
//         first hit along the ray -> strength into the bin of its range,
//         the bin keeping the STRONGEST echo it received (max, not sum).
//
// Everything in this header is that reduction and the ray table it needs,
// with no renderer in sight: the Vulkan SonarSensor fires the rays through
// the path tracer's acceleration structure, but the fold from returns to
// echogram is plain arithmetic, so it lives here where CI can pin it without
// a GPU and where any other ray source (a raster depth pass, a recorded
// return set) can reuse it.
//
// What this is NOT: an acoustic model. There is no beam pattern, no
// sidelobes, no multipath, no reverberation, no time-varying gain. It is
// the same class of proxy UUV Simulator, Stonefish and HoloOcean's
// ray-based sonars are — a labelled range/bearing image with the right
// geometry and a plausible intensity, honest about its provenance.
//
// The max-reduction is deliberate twice over. Physically, a bin's echo is
// dominated by its brightest reflector at this level of modelling. And a
// max is order-independent: however the rays are scheduled, the echogram
// is bit-identical, which is the property the sensor-determinism audit
// measures (a sum of floats in scheduling order would not be).
#ifndef THREEPP_SONARMODEL_HPP
#define THREEPP_SONARMODEL_HPP

#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/math/Vector3.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace threepp {

    /**
     * Fan geometry and echo model of an imaging sonar.
     *
     * Angles follow the LIDAR frame: azimuth 0 is the sensor's local -Z
     * (forward), positive azimuth turns toward local +X (the operator's right),
     * and positive elevation is up. Beams span [-horizontalFov/2,
     * +horizontalFov/2] at beam centres, left to right; samples span the
     * vertical aperture the same way, bottom to top.
     */
    struct SonarModel {
        // Full horizontal swath in degrees, and the number of beams across it.
        float horizontalFov = 130.f;
        unsigned int beams = 256;
        // Full vertical aperture in degrees, and the rays fired through it per
        // beam. More samples = a beam that cannot miss a thin target inside
        // its aperture; 8-16 is plenty for a net or a hull.
        float verticalAperture = 20.f;
        unsigned int verticalSamples = 16;
        // Range window in metres. `minRange` is the blind zone: surfaces closer
        // are traced through, not echoed.
        float maxRange = 20.f;
        float minRange = 0.f;
        // Range bins across [0, maxRange].
        unsigned int rangeBins = 512;
        // One-way amplitude attenuation in 1/m, applied over the two-way path:
        //     strength *= exp(-2 * attenuation * range)
        // 0 = lossless. 0.05 reads as a 20 m sonar in clear water; seawater at
        // 700 kHz is ~0.02-0.03 (0.2 dB/m), muddy harbour water more.
        float attenuation = 0.05f;
        // Incidence model: strength *= floor + (1 - floor) * |n . d|. A floor of
        // 0 is Lambertian; real targets scatter back at grazing angles too, so a
        // net seen edge-on still echoes — 0.35 keeps it on the image.
        float incidenceFloor = 0.35f;

        [[nodiscard]] unsigned int rayCount() const { return beams * verticalSamples; }
        [[nodiscard]] float binWidth() const { return rangeBins ? maxRange / static_cast<float>(rangeBins) : 0.f; }

        // ── Presets ──────────────────────────────────────────────────────
        // Each encodes swath, beam count, aperture and range from the
        // manufacturer's datasheet — the geometry. Beam pattern, frequency
        // and gain are not modelled, so two sonars with the same geometry
        // are the same preset here.

        // The generic wide fan: 130 deg x 20 deg, 256 beams, 20 m.
        static SonarModel Wide130() { return SonarModel{}; }

        // Blueprint Oculus M750d, 750 kHz mode: 130 deg x 20 deg, 512 beams, 120 m.
        static SonarModel OculusM750d() {
            SonarModel m;
            m.horizontalFov = 130.f;
            m.beams = 512;
            m.verticalAperture = 20.f;
            m.maxRange = 120.f;
            m.rangeBins = 1024;
            m.attenuation = 0.02f;
            return m;
        }

        // Teledyne BlueView M900-130: 130 deg x 20 deg, 768 beams, 100 m.
        static SonarModel BlueViewM900() {
            SonarModel m;
            m.horizontalFov = 130.f;
            m.beams = 768;
            m.verticalAperture = 20.f;
            m.maxRange = 100.f;
            m.rangeBins = 1024;
            m.attenuation = 0.02f;
            return m;
        }

        // Tritech Gemini 720is: 120 deg x 20 deg, 512 beams, 120 m.
        static SonarModel Gemini720is() {
            SonarModel m;
            m.horizontalFov = 120.f;
            m.beams = 512;
            m.verticalAperture = 20.f;
            m.maxRange = 120.f;
            m.rangeBins = 1024;
            m.attenuation = 0.02f;
            return m;
        }
    };

    /**
     * One sonar frame: echo strength per (beam, range bin), beam-major.
     * `intensity[beam * bins + bin]` is in [0, 1] before gain; 0 = no echo.
     */
    struct SonarImage {
        unsigned int beams = 0;
        unsigned int bins = 0;
        float maxRange = 0.f;
        // Sim time the rays were fired at (VisionSensor::lastScanTime).
        double time = 0.0;
        std::vector<float> intensity;

        // Size for `model` and clear to zero; keeps the allocation across frames.
        void reset(const SonarModel& model);

        [[nodiscard]] float at(unsigned int beam, unsigned int bin) const {
            return intensity[static_cast<std::size_t>(beam) * bins + bin];
        }
        [[nodiscard]] float& at(unsigned int beam, unsigned int bin) {
            return intensity[static_cast<std::size_t>(beam) * bins + bin];
        }
        [[nodiscard]] float rangeOfBin(unsigned int bin) const {
            return bins ? (static_cast<float>(bin) + 0.5f) / static_cast<float>(bins) * maxRange : 0.f;
        }
    };

    /**
     * Acoustic reflectivity per target, keyed on the STABLE instance id the
     * return carries (VulkanRenderer::setObjectInstanceId — the same number the
     * Ids AOV writes). A surface without an entry echoes at `defaultValue`;
     * volume-scatter returns (a water-column medium set through LidarParams)
     * echo at `volume`. Optical material is irrelevant here on purpose: a
     * black-painted steel collar is acoustically bright, and a net that is
     * nearly transparent to a camera is a wall to a sonar.
     */
    struct SonarReflectivity {
        float defaultValue = 1.f;
        float volume = 0.1f;
        std::unordered_map<std::int32_t, float> byInstance;

        void set(std::int32_t instanceId, float reflectivity) { byInstance[instanceId] = reflectivity; }
        [[nodiscard]] float of(const LidarReturn& r) const;
    };

    /**
     * Sensor-local unit ray directions for `model`, beam-major / sample-minor:
     * ray index = beam * verticalSamples + sample. Beam 0 is the sensor's
     * left-most beam (most negative azimuth, toward local -X), so the fan reads
     * left-to-right as the operator sees it; sample 0 is the lowest elevation.
     */
    [[nodiscard]] std::vector<Vector3> sonarRayDirections(const SonarModel& model);

    /**
     * Fold a return set into an echogram. `returns` follows the ray layout
     * above; each ray may carry several returns (LidarParams::maxReturns > 1
     * lets sound pass a transmissive net and echo off the fish behind it), in
     * which case returns.size() is a multiple of model.rayCount() and every
     * return of ray k sits at [k * perRay, (k + 1) * perRay). Misses are
     * skipped. `origin` is the pose the rays were fired from — the incidence
     * term needs the ray direction, which is recovered from it.
     *
     * `out` is reset for the model first; call once per scan.
     */
    void sonarAccumulate(const SonarModel& model, const SonarReflectivity& reflectivity,
                         const std::vector<LidarReturn>& returns, const Vector3& origin,
                         SonarImage& out);

}// namespace threepp

#endif//THREEPP_SONARMODEL_HPP
