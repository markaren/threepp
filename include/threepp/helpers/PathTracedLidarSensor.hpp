#ifndef THREEPP_PATHTRACEDLIDARSENSOR_HPP
#define THREEPP_PATHTRACEDLIDARSENSOR_HPP

#include "LidarModel.hpp"
#include "LidarTypes.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/VisionSensor.hpp"

#include <vector>

namespace threepp {

    class VulkanRenderer;

    /**
     * Path-traced LIDAR scanner. Mirrors `LidarSensor`'s ergonomics
     * (sensor is an `Object3D` added to the scene; `scan()` is called
     * each frame after `renderer.render()`) but routes the ray cast
     * through `VulkanRenderer::scanLidar` — i.e. the renderer's own TLAS,
     * with per-beam intensity computed from the same `MaterialDesc` the
     * path tracer uses. Returns physically-grounded `LidarReturn`
     * tuples including normal + intensity + hit-instance, rather than
     * the position-only output of the raster `LidarSensor`.
     *
     * Two constructors mirror the raster version:
     *   - dense-grid mode: a uniform azimuth × elevation grid (debug /
     *     ground-truth captures);
     *   - model-based mode: a real-sensor beam pattern via `LidarModel`.
     *
     * Coordinate convention (matches `LidarModel`):
     *   azimuth = 0 → sensor-local -Z (forward); increases CCW from +Y.
     *   elevation > 0 → upward.
     *
     * `scan()` requires that `renderer.render(scene, camera)` has been
     * called at least once for the current scene so the TLAS is built.
     * It is safe to call between `render()` invocations; it must NOT be
     * called concurrently with `render()`.
     *
     * Range noise (`rangeNoise`, inherited from VisionSensor) defaults to
     * OFF here — unlike the raster sensors, this back-end reports the ray
     * tracer's own physically-derived range, and the callers that want a
     * noisy sensor (DepthSensor on Vulkan) push their model down into it.
     * Turn it on for a noisy LIDAR: the seeded stream makes the result
     * replayable, and beams are perturbed in beam-table order.
     */
    class PathTracedLidarSensor: public Object3D, public VisionSensor {

    public:
        // Tweakable LIDAR-equation parameters (forwarded to the renderer).
        LidarParams params;

        // Forward is local -Z (azimuth 0), so lookAt() aims it (camera convention).
        [[nodiscard]] bool usesCameraLookAtConvention() const override { return true; }

        /**
         * Dense-grid mode: shoots `hRes × vRes` beams covering the full
         * sphere (azimuth ∈ [-π, π], elevation ∈ [-π/2, π/2]).
         */
        PathTracedLidarSensor(unsigned int hRes, unsigned int vRes, float maxRange = 100.f);

        /**
         * Model-based mode: shoots beams matching a real sensor's pattern.
         */
        explicit PathTracedLidarSensor(const LidarModel& model, float maxRange = 100.f);

        /**
         * Depth-camera mode: a pinhole grid of width × height beams with a
         * vertical FOV of `fovY` degrees (aspect = width/height), looking
         * along sensor-local -Z — the same pattern and mounting convention
         * as the raster `DepthSensor`. Lets perception code swap the raster
         * sensor for a ray-traced one on Vulkan without changing anything
         * downstream.
         */
        PathTracedLidarSensor(float fovY, unsigned int width, unsigned int height, float maxRange = 100.f);

        /**
         * Run one scan. Beams are derived from the current world matrix
         * (composed from `position` / `rotation`) and the cached
         * sensor-local direction table.
         */
        void scan(VulkanRenderer& renderer, std::vector<LidarReturn>& out);

        /**
         * The clean leg of the last collected `params.pairedCleanTrace` scan:
         * the same beams, the same RNG keys, traced with the ParticleField
         * density medium switched off. Empty when the flag is off.
         *
         * DELIBERATELY UN-NOISED. `rangeNoise` models the detector, and the
         * detector is not what the paired trace is measuring — a reference
         * carrying its own independent noise draw would put that noise into
         * every `lidarDegradation::rangeError` twice over. Compare this leg
         * against `out` row for row.
         */
        [[nodiscard]] const std::vector<LidarReturn>& cleanReturns() const { return clean_; }

        /**
         * The same scan, split so a frame loop does not have to block on the
         * readback (which costs every frame already queued on the GPU — see
         * VulkanRenderer::scanLidarBegin). Fire with scanBegin() on one frame,
         * poll scanReady(), take delivery with scanCollect() on a later one.
         *
         * The scan is STAMPED and the beams are AIMED at scanBegin(): the
         * returns describe the pose the sensor held when it fired, and the
         * seeded range noise is drawn at collect in beam-table order, so the
         * stream replays exactly as the synchronous path's does.
         */
        void scanBegin(VulkanRenderer& renderer);
        // Whether scanBegin() actually got a slot (it can be refused when too
        // many scans are already outstanding — the caller retries next frame).
        [[nodiscard]] bool scanFired() const { return scanHandle_ >= 0; }
        [[nodiscard]] bool scanReady(const VulkanRenderer& renderer) const;
        // False when nothing was fired; `out` is then left empty.
        bool scanCollect(VulkanRenderer& renderer, std::vector<LidarReturn>& out);

        [[nodiscard]] unsigned int beamCount() const { return static_cast<unsigned int>(directions_.size()); }
        [[nodiscard]] const std::vector<Vector3>& beamDirections() const { return directions_; }

    private:
        // Sensor-local unit beam directions. Built once at construction.
        std::vector<Vector3> directions_;
        // Scratch buffer reused across scans so we don't reallocate.
        std::vector<LidarBeam> beamScratch_;
        // The paired clean leg, filled by scanCollect when the flag is set.
        std::vector<LidarReturn> clean_;
        // The world position the outstanding scan was fired from. Noise is
        // applied along each beam from ITS origin, which is the pose at
        // scanBegin() and not wherever the sensor has moved to by collect.
        Vector3 scanOrigin_;
        // The renderer's handle for the outstanding dispatch; -1 = none.
        int scanHandle_ = -1;

        void buildDenseBeams(unsigned int hRes, unsigned int vRes);
        void buildModelBeams(const LidarModel& model);
        void buildCameraBeams(float fovY, unsigned int width, unsigned int height);

        // Perturbs each return along its beam per `rangeNoise`; no-op when the
        // model is zero.
        void applyNoise(std::vector<LidarReturn>& out, const Vector3& origin);
    };

}// namespace threepp

#endif//THREEPP_PATHTRACEDLIDARSENSOR_HPP
