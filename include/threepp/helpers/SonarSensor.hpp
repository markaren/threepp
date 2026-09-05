#ifndef THREEPP_SONARSENSOR_HPP
#define THREEPP_SONARSENSOR_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/VisionSensor.hpp"
#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/helpers/SonarModel.hpp"

#include <vector>

namespace threepp {

    class VulkanRenderer;

    /**
     * Forward-looking imaging sonar, ray-cast through the Vulkan renderer's
     * own acceleration structure. Sibling of PathTracedLidarSensor: an
     * Object3D whose world frame is the transducer, fired from the frame loop
     * after `renderer.render()`, producing a SonarImage (echo strength per
     * beam and range bin) instead of a point cloud.
     *
     * The fan is a SonarModel; the rays are the model's beam × sample table
     * transformed by this node's world matrix, traced as one dispatch through
     * `VulkanRenderer::scanLidarBegin`, and folded into the echogram by
     * `sonarAccumulate`. Per-ray hits from the last scan stay readable through
     * `lastReturns()` for a point-cloud view or for joining against an Ids AOV.
     *
     * What the sonar sees is what the tracer's TLAS holds, with two knobs that
     * matter more here than for a lidar:
     *
     *   - Acoustic proxies. A net the camera renders as twine is a wall to a
     *     sonar. Put a solid membrane on VulkanRenderer::kSensorOnlyLayer and
     *     enable setSensorOnlySurfaces(true): the sonar echoes off it and the
     *     camera never draws it. The same layer hides a fish's echo cage, a
     *     hull's simplified shell, anything acoustically present but visually
     *     absent — and its inverse (visible, acoustically transparent) is a
     *     mesh left off the TLAS: Points, lines, a `visible = false` proxy.
     *   - Reflectivity per target through `reflectivity`, keyed on the stable
     *     instance id (setObjectInstanceId). Optical material is ignored.
     *
     * `params` is the tracer's LidarParams. maxRange / minRange mirror the
     * model at construction; `detectorThreshold` is zeroed so an optically
     * dark surface still returns (the sonar has its own intensity model);
     * `maxReturns > 1` lets sound pass a transmissive surface and echo off
     * what lies behind it; the medium fields put back-scatter from a water
     * column into the image as volume returns.
     *
     * Reproducibility follows the VisionSensor contract: the scan is stamped
     * with sim time, `rangeNoise` perturbs each ray's range from a seeded
     * stream in ray order, and `speckle` multiplies each bin by a seeded
     * uniform factor in bin order. Both zero by default — the clean image is
     * bit-exact across processes on the same device, and the fold is a max,
     * so ray scheduling cannot move a bit.
     *
     * Forward is local -Z, so lookAt() aims the fan (camera convention).
     */
    class SonarSensor: public Object3D, public VisionSensor {

    public:
        // Tracer parameters (see LidarParams). Ranges are mirrored from the
        // model at construction; edit model-side ranges through `params`.
        LidarParams params;
        // Echo strength per target; see SonarReflectivity.
        SonarReflectivity reflectivity;
        // Multiplicative speckle: each bin is scaled by a seeded uniform draw in
        // [1 - speckle, 1 + speckle]. 0 = clean. Drawn in bin order from a
        // stream seeded off rangeNoise.seed, so the same seed reproduces it.
        float speckle = 0.f;

        [[nodiscard]] bool usesCameraLookAtConvention() const override { return true; }

        explicit SonarSensor(const SonarModel& model = SonarModel::Wide130());

        [[nodiscard]] const SonarModel& model() const { return model_; }
        [[nodiscard]] unsigned int rayCount() const { return static_cast<unsigned int>(directions_.size()); }
        [[nodiscard]] const std::vector<Vector3>& rayDirections() const { return directions_; }

        // Attenuation / incidence / bins are live-tweakable between scans; the
        // fan geometry is not (it is the ray table), so beams, samples, swath
        // and aperture changes need a new sensor.
        void setAttenuation(float perMetre) { model_.attenuation = perMetre; }
        void setIncidenceFloor(float floor) { model_.incidenceFloor = floor; }
        void setRangeBins(unsigned int bins) { model_.rangeBins = bins; }

        /**
         * One scan from the current pose, blocking on the trace. Prefer the
         * split form from a frame loop.
         */
        void scan(VulkanRenderer& renderer, SonarImage& out);

        /**
         * Fire on one frame, collect on a later one — the same pipelined path
         * PathTracedLidarSensor uses, with the same latency semantics: the
         * image describes the pose the sonar held at scanBegin().
         */
        void scanBegin(VulkanRenderer& renderer);
        [[nodiscard]] bool scanFired() const { return scanHandle_ >= 0; }
        [[nodiscard]] bool scanReady(const VulkanRenderer& renderer) const;
        // False when nothing was fired; `out` is then reset to zeros.
        bool scanCollect(VulkanRenderer& renderer, SonarImage& out);

        // The per-ray hits of the last collected scan, in ray order (with
        // params.maxReturns entries per ray). Positions carry the range noise.
        [[nodiscard]] const std::vector<LidarReturn>& lastReturns() const { return returns_; }
        // The world position the last scan was fired from.
        [[nodiscard]] const Vector3& lastOrigin() const { return scanOrigin_; }

        void resetNoise() override;

    private:
        SonarModel model_;
        std::vector<Vector3> directions_;
        std::vector<LidarBeam> beamScratch_;
        std::vector<LidarReturn> returns_;
        Vector3 scanOrigin_;
        int scanHandle_ = -1;
        // Speckle has its own stream so its draw count never shifts the range
        // noise stream: a scan with speckle on and one with it off perturb the
        // ranges identically.
        SplitMix64 speckleRng_;
        std::uint64_t speckleSeededWith_ = 0;

        void applyRangeNoise(std::vector<LidarReturn>& returns, const Vector3& origin);
        void applySpeckle(SonarImage& image);
        [[nodiscard]] std::uint64_t speckleSeed() const;
    };

}// namespace threepp

#endif//THREEPP_SONARSENSOR_HPP
