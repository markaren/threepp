// The layer between VisionSensor and the ranging sensors that scan with a
// renderer (DepthSensor, LidarSensor): one implementation of the two-backend
// scan protocol the two of them used to carry as twins.
//
// VisionSensor stays renderer-free — it is the seeded-noise / sim-clock
// contract the replay tests pin. THIS class is where "a scan needs a Renderer"
// lives, and everything that follows from having two kinds of one:
//
//   * backend dispatch: scan() takes the abstract Renderer&. On a raster
//     backend the sensor renders depth and unprojects it; on a VulkanRenderer
//     there is no raster depth pass, so the same beam pattern is traced
//     through the renderer's path-tracing TLAS via a cached
//     PathTracedLidarSensor back-end. The back-end is built on first use and
//     KEPT — a seeded sensor rebuilt per scan replays its seed each scan,
//     freezing the noise into the geometry.
//   * the fire/collect state machine (scanBegin / scanReady / scanCollect),
//     with the refused-slot and keep-last-cloud semantics both sensors must
//     agree on for one caller (SensorPlaySession) to drive them alike.
//   * the noise + range-shell sync onto the traced back-end, and the
//     resetNoise() cascade — the back-end owns the noise stream on Vulkan,
//     and a reset that stopped at the front door would leave an episode
//     unreplayable on exactly one backend.
//
// What stays in the concrete sensor is its beam geometry: how to build the
// traced back-end, how to fill a cloud from a raster pass, and how to turn
// the tracer's LidarReturns into its own point type — the three hooks below.

#ifndef THREEPP_SENSORS_TRACEDRASTERVISIONSENSOR_HPP
#define THREEPP_SENSORS_TRACEDRASTERVISIONSENSOR_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/VisionSensor.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class PathTracedLidarSensor;
    class Renderer;
    class Scene;
    class VulkanRenderer;
    struct LidarReturn;

    template<typename PointT>
    class TracedRasterVisionSensor: public Object3D, public VisionSensor {

    public:
        using Cloud = std::vector<PointT>;

        // Beams leave along local -Z, so lookAt() aims them (camera convention).
        [[nodiscard]] bool usesCameraLookAtConvention() const override { return true; }

        // Also re-seeds the traced back-end, which owns the stream on Vulkan.
        void resetNoise() override;

        /**
         * Performs a scan and writes the returns into `cloud`.
         *
         * The sensor's world matrix must be current (add it to the scene, or
         * call updateMatrixWorld) before calling this.
         *
         * Raster: renders depth from the sensor's viewpoint, linearizes, reads
         * back and unprojects; the renderer's active render target is restored
         * to nullptr afterwards. Vulkan: the beam pattern is traced through
         * the renderer's path-tracing acceleration structure, so the scene
         * must have been render()-ed at least once beforehand; `scene` is then
         * unused (the TLAS is traced, not the scene graph).
         *
         * Synchronous contract: this call's returns, or an empty cloud.
         */
        void scan(Renderer& renderer, Scene& scene, Cloud& cloud);

        /**
         * The frame-loop form of scan(), split into a fire and a delivery.
         *
         * scan() on Vulkan blocks on a GPU readback, and a blocking readback
         * does not cost the trace — it costs every frame already queued behind
         * the fence. On an RTX 4070 with two frames in flight that turned a
         * 1.2 ms VLP-16 trace into a 28 ms stall, landing on whatever frame
         * the rate gate happened to fire on: a 10 Hz sensor hitching a 60 Hz
         * app ten times a second. Fire on one frame, deliver on the next, and
         * the trace costs what it takes.
         *
         *     if (sensor.scanDue() && sensor.scanBegin(renderer, scene, cloud))
         *         take(cloud);                      // raster: already in hand
         *     if (sensor.scanReady(renderer))       // a poll, never a wait
         *         if (sensor.scanCollect(renderer, cloud)) take(cloud);
         *
         * On a RASTER backend there is nothing to pipeline: scanBegin() does
         * the whole scan (it already blocks on the framebuffer reads), fills
         * `cloud` there and then, and RETURNS TRUE to say so. On Vulkan it
         * returns false and the cloud arrives at a later scanCollect(). A
         * caller written against the pair behaves correctly on both backends;
         * only the frame the cloud lands on differs.
         *
         * The scan is stamped and aimed at scanBegin(); lastScanTime() is the
         * fire time on both backends.
         */
        bool scanBegin(Renderer& renderer, Scene& scene, Cloud& cloud);
        // Whether a fired scan has not been collected yet. Fire again while
        // one is outstanding and the earlier one is thrown away, so a driver
        // on a rate gate should skip a due scan while this is true.
        [[nodiscard]] bool scanPending() const { return scanPending_; }
        // Whether a fired scan can be collected without waiting. Raster: true
        // as soon as scanBegin has run.
        [[nodiscard]] bool scanReady(const Renderer& renderer) const;
        // Take delivery. False when nothing is outstanding (and `cloud` is
        // untouched, so the caller keeps the previous cloud rather than
        // blinking).
        bool scanCollect(Renderer& renderer, Cloud& cloud);

        // The range shell the sensor reports in: a return's distance from the
        // sensor origin always lies in [near(), far()], on every backend.
        // These are RANGES — a blind SPHERE out to a maximum range — not
        // view-space clip planes; see the concrete sensors for the raster
        // frustum note.
        [[nodiscard]] float near() const { return near_; }
        [[nodiscard]] float far() const { return far_; }

    protected:
        TracedRasterVisionSensor(const RangeNoiseModel& noise, float near, float far);

        // Out-of-line: the cached path-traced back-end is only forward-declared.
        ~TracedRasterVisionSensor() override;

        // The reported range shell. Distinct from any raster near plane, which
        // the sensors pull IN so it cannot clip inside the shell.
        float near_;
        float far_;

        // ── the three hooks a concrete sensor implements ─────────────────────

        // Build the traced back-end with this sensor's beam pattern. Called at
        // most once (the back-end is cached), and only on a Vulkan build — the
        // raster-only stub returns nullptr.
        virtual std::unique_ptr<PathTracedLidarSensor> createTracedBackend() = 0;

        // The whole raster scan: render depth, read back, unproject into
        // `cloud`. Runs under a DataPassGuard (linear, un-tonemapped,
        // autoClear on); stamping, gating and the pending flag are already
        // handled by the skeleton.
        virtual void rasterScan(Renderer& renderer, Scene& scene, Cloud& cloud) = 0;

        // Take delivery from the traced back-end and convert/filter into the
        // sensor's own cloud type. Only called with a fired scan outstanding;
        // false when the back-end had nothing (`cloud` is then untouched).
        virtual bool collectTraced(PathTracedLidarSensor& backend, VulkanRenderer& renderer, Cloud& cloud) = 0;

        // ── shared machinery for the sensors ─────────────────────────────────

        // The cached back-end, built on first call, with the noise model and
        // range shell synced onto it. Only meaningful on a Vulkan build — the
        // raster-only createTracedBackend() stub returns nullptr.
        PathTracedLidarSensor& tracedBackend();

        // tracedBackend() with the sensor's world pose decomposed onto it —
        // what a traced fire aims with. Refreshes a parentless sensor's
        // matrix first, so a sensor aimed with position/lookAt but never
        // added to a scene still fires from where it points.
        PathTracedLidarSensor& aimedTracedBackend();

        // The per-sample core of every raster unprojection loop: normalized
        // depth → (view depth, slant range), the range shell applied to the
        // true geometry, then the noise model applied to the RANGE (that is
        // what its sigmas mean, and what the traced back-end perturbs) and the
        // shell re-applied. Both bounds INCLUSIVE, matching traceRayEXT's
        // [tMin, tMax] on the traced path. `k` is the pixel's slant ratio
        // |view ray| / |view z|; false = this sample produces no return.
        bool shellRange(float normalizedDepth, float k, bool addNoise, float& depth, float& slant) {
            depth = normalizedDepth * far_;
            slant = depth * k;
            if (slant < near_ || slant > far_) return false;
            if (addNoise) {
                slant = applyRangeNoise(slant);
                if (slant < near_ || slant > far_) return false;
                depth = slant / k;
            }
            return true;
        }

    private:
        // Path-traced back-end for Vulkan scans, built on first use and kept.
        std::unique_ptr<PathTracedLidarSensor> tracedBackend_;

        // A scanBegin() awaiting its scanCollect(). On a raster backend the
        // cloud is already in the caller's vector and this only says "one
        // delivery is owed"; on Vulkan the back-end holds the outstanding
        // dispatch.
        bool scanPending_ = false;
    };

    // The two instantiations that exist, compiled once in
    // TracedRasterVisionSensor.cpp (which is where VulkanRenderer may be seen).
    extern template class TracedRasterVisionSensor<Vector3>;
    extern template class TracedRasterVisionSensor<LidarReturn>;

}// namespace threepp

#endif// THREEPP_SENSORS_TRACEDRASTERVISIONSENSOR_HPP
