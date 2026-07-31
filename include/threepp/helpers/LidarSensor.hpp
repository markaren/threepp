#ifndef THREEPP_LIDARSENSOR_HPP
#define THREEPP_LIDARSENSOR_HPP

#include "LidarModel.hpp"
#include "LidarTypes.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/VisionSensor.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"

#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace threepp {

    class ShaderMaterial;
    class Renderer;
    class PathTracedLidarSensor;

    /**
     * Simulates a full 360-degree LiDAR sensor using six 90-degree perspective
     * cameras oriented along the ±X, ±Y and ±Z axes.
     *
     * Two modes:
     *   - Dense grid (default constructor): every pixel on every cube face becomes
     *     a point — useful for visualisation and debugging.
     *   - Model-based (LidarModel constructor): only the beams defined by the model
     *     are sampled, matching the angular pattern of a real sensor (VLP-16, etc.).
     *
     * The sensor object must be in the scene (or have its parent chain updated)
     * before calling scan(), so all child camera world matrices are current.
     *
     * Range noise is inherited from VisionSensor: `rangeNoise` is a seeded
     * RangeNoiseModel (default: 0.02 m sigma), so a replayed run reproduces the
     * same cloud. Beams are perturbed in a fixed order (face 0..5, row-major
     * within a face; beam-table order in model mode), which is what makes the
     * seed reproduce a specific cloud rather than merely a specific histogram.
     */
    class LidarSensor: public Object3D, public VisionSensor {

    public:
        /**
         * Dense-grid mode: every pixel on all six cube faces is returned.
         * @param faceSize  Resolution of each cube face in pixels (square).
         * @param near      Near clip plane in metres.
         * @param far       Far clip plane / max range in metres.
         */
        explicit LidarSensor(unsigned int faceSize, float near = 0.1f, float far = 100.f);

        /**
         * Model-based mode: only beams defined by the LidarModel are sampled.
         * @param model     Beam pattern (elevation angles + azimuth resolution).
         * @param faceSize  Cube-face resolution. Should be ≥ 90/azimuthResolution
         *                  to avoid aliasing (e.g. 512 for 0.2° resolution).
         * @param near      Near clip plane in metres.
         * @param far       Far clip plane / max range in metres.
         */
        LidarSensor(const LidarModel& model, unsigned int faceSize, float near = 0.1f, float far = 100.f);

        ~LidarSensor() override;

        void resetNoise() override;

        /**
         * Performs a scan and writes per-beam returns into `cloud`.
         *
         * On a raster backend (GL), position and distance are populated from
         * the cube-face depth read; the remaining `LidarReturn` fields are
         * left at sentinel defaults (intensity = 0, normal = 0,
         * hitInstanceId = -1) because the raster pipeline does not have
         * access to material or geometry data.
         *
         * On a `VulkanRenderer` there is no raster depth cube to read back:
         * the same beam pattern is traced through the renderer's TLAS via a
         * cached `PathTracedLidarSensor` back-end instead (so the scene must
         * have been render()-ed at least once), and the returns carry the
         * tracer's intensity / normal / hitInstanceId. Sentinel returns
         * (misses, fog scatter) and near-clip hits are dropped to match the
         * raster output shape.
         *
         * The renderer's active render target is restored to nullptr after the scan.
         */
        void scan(Renderer& renderer, Scene& scene, std::vector<LidarReturn>& cloud);

        /**
         * The frame-loop form of scan(), split into a fire and a delivery.
         *
         * scan() on Vulkan blocks on a GPU readback, and a blocking readback
         * does not cost the trace — it costs every frame already queued behind
         * the fence. On an RTX 4070 with two frames in flight that turned a
         * 1.2 ms VLP-16 trace into a 28 ms stall, landing on whatever frame the
         * rate gate happened to fire on: a 10 Hz sensor hitching a 60 Hz app ten
         * times a second. Fire on one frame, deliver on the next, and the trace
         * costs what it takes.
         *
         *     if (sensor.scanDue() && sensor.scanBegin(renderer, scene, cloud))
         *         take(cloud);                      // raster: already in hand
         *     if (sensor.scanReady(renderer))       // a poll, never a wait
         *         if (sensor.scanCollect(renderer, cloud)) take(cloud);
         *
         * On a RASTER backend there is nothing to pipeline: scanBegin() does the
         * whole scan (it already blocks on six framebuffer reads), fills `cloud`
         * there and then, and RETURNS TRUE to say so. On Vulkan it returns false
         * and the cloud arrives at a later scanCollect(). A caller written
         * against the pair behaves correctly on both backends; only the frame
         * the cloud lands on differs.
         *
         * The scan is stamped and aimed at scanBegin(); lastScanTime() is the
         * fire time on both backends.
         */
        bool scanBegin(Renderer& renderer, Scene& scene, std::vector<LidarReturn>& cloud);
        // Whether a fired scan has not been collected yet. Fire again while one
        // is outstanding and the earlier one is thrown away, so a driver on a
        // rate gate should skip a due scan while this is true.
        [[nodiscard]] bool scanPending() const { return scanPending_; }
        // Whether a fired scan can be collected without waiting. Raster: true
        // as soon as scanBegin has run.
        [[nodiscard]] bool scanReady(const Renderer& renderer) const;
        // Take delivery. False when nothing is outstanding (and `cloud` is
        // untouched, so the caller keeps the previous cloud rather than blinking).
        bool scanCollect(Renderer& renderer, std::vector<LidarReturn>& cloud);

        [[nodiscard]] unsigned int faceSize() const { return faceSize_; }
        [[nodiscard]] float near() const { return near_; }
        [[nodiscard]] float far() const { return far_; }

    private:
        static constexpr int kNumFaces = 6;

        unsigned int faceSize_;
        float near_;
        float far_;

        Scene postScene_;
        OrthographicCamera postCamera_;
        std::shared_ptr<ShaderMaterial> postMaterial_;

        // Non-owning pointers into the children list for fast per-face access
        std::array<PerspectiveCamera*, kNumFaces> cameras_{};
        std::array<std::unique_ptr<GLRenderTarget>, kNumFaces> sceneTargets_;
        std::array<std::unique_ptr<GLRenderTarget>, kNumFaces> readbackTargets_;

        // Dense-grid mode: shared ray direction factors (tan(90°/2) = 1, so just NDC coords)
        std::vector<float> dir_;

        // Model-based mode: one entry per beam
        struct BeamSample {
            uint8_t face;
            uint16_t pixelX, pixelY;
            float u, v;// exact NDC of this beam's direction in the face camera
        };
        std::vector<BeamSample> beams_;

        // Model-based mode keeps its model so the Vulkan back-end can be built
        // with the same beam pattern; empty in dense-grid mode.
        std::optional<LidarModel> model_;

        // Path-traced back-end for Vulkan scans, built on first use.
        std::unique_ptr<PathTracedLidarSensor> tracedBackend_;

        // A scanBegin() awaiting its scanCollect(). On a raster backend the
        // cloud is already in the caller's vector and this only says "one
        // delivery is owed"; on Vulkan the back-end holds the outstanding
        // dispatch.
        bool scanPending_ = false;

        // Builds tracedBackend_ on first call and syncs the noise model onto it.
        PathTracedLidarSensor& tracedBackend();

        void init(float near, float far);
        void buildBeamTable(const LidarModel& model);

        void renderFaces(Renderer& renderer, Scene& scene);
        void unprojectDense(std::vector<LidarReturn>& points);
        void unprojectBeams(std::vector<LidarReturn>& points);
    };

}// namespace threepp

#endif//THREEPP_LIDARSENSOR_HPP
