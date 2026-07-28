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
