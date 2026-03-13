#ifndef THREEPP_LIDARSENSOR_HPP
#define THREEPP_LIDARSENSOR_HPP

#include "LidarModel.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"

#include <array>
#include <vector>

namespace threepp {

    class ShaderMaterial;
    class GLRenderer;

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
     */
    class LidarSensor: public Object3D {

    public:
        // Gaussian range noise standard deviation in metres (0 = perfect sensor)
        float rangeNoise{0.02f};

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

        /**
         * Performs a scan and returns hit points in world space.
         * The renderer's active render target is restored to nullptr after the scan.
         */
        void scan(GLRenderer& renderer, Scene& scene, std::vector<Vector3>& cloud);

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

        void init(float near, float far);
        void buildBeamTable(const LidarModel& model);

        void renderFaces(GLRenderer& renderer, Scene& scene);
        void unprojectDense(std::vector<Vector3>& points) const;
        void unprojectBeams(std::vector<Vector3>& points) const;
    };

}// namespace threepp

#endif//THREEPP_LIDARSENSOR_HPP
