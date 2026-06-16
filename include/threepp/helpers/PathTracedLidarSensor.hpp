#ifndef THREEPP_PATHTRACEDLIDARSENSOR_HPP
#define THREEPP_PATHTRACEDLIDARSENSOR_HPP

#include "LidarModel.hpp"
#include "LidarTypes.hpp"
#include "threepp/core/Object3D.hpp"

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
     */
    class PathTracedLidarSensor: public Object3D {

    public:
        // Tweakable LIDAR-equation parameters (forwarded to the renderer).
        LidarParams params;

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

        [[nodiscard]] unsigned int beamCount() const { return static_cast<unsigned int>(directions_.size()); }
        [[nodiscard]] const std::vector<Vector3>& beamDirections() const { return directions_; }

    private:
        // Sensor-local unit beam directions. Built once at construction.
        std::vector<Vector3> directions_;
        // Scratch buffer reused across scans so we don't reallocate.
        std::vector<LidarBeam> beamScratch_;

        void buildDenseBeams(unsigned int hRes, unsigned int vRes);
        void buildModelBeams(const LidarModel& model);
        void buildCameraBeams(float fovY, unsigned int width, unsigned int height);
    };

}// namespace threepp

#endif//THREEPP_PATHTRACEDLIDARSENSOR_HPP
