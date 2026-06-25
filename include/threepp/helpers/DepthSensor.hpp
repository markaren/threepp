#ifndef THREEPP_LIDAR_HPP
#define THREEPP_LIDAR_HPP

#include "CameraHelper.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"


#include <vector>

namespace threepp {

    class ShaderMaterial;
    class Renderer;

    /**
     * Simulates a depth sensor, backend-neutrally.
     *
     * `scan()` takes the abstract `Renderer&` and dispatches on the concrete backend:
     *   - GLRenderer (and other raster backends): renders the scene from the sensor's
     *     viewpoint into a depth texture, linearizes it via a post-process shader, reads
     *     back the pixels, and reprojects them into world-space 3D points.
     *   - VulkanRenderer: there is no raster depth pass, so the same pinhole ray pattern
     *     is traced through the renderer's path-tracing acceleration structure (via an
     *     internal PathTracedLidarSensor in camera mode). The result is the same world-
     *     space point cloud, so perception/deploy code is identical on either backend.
     *
     * Can scan with or without color: the former is slightly more expensive but gives
     * per-point color information, while the latter is faster and uses less GPU memory.
     * (On Vulkan the "color" is the LIDAR intensity as greyscale, not surface albedo.)
     */
    class DepthSensor: public Object3D {

    public:
        // Gaussian range noise standard deviation in metres (0 = perfect sensor)
        float rangeNoise{0.02f};

        DepthSensor(float fovY, unsigned int width, unsigned int height,
                    float near = 0.1f, float far = 100.f);

        /**
         * Performs a scan and returns the hit points in world space.
         *
         * The sensor's world matrix must be current (add it to the scene, or call
         * updateWorldMatrix/updateMatrixWorld) before calling this.
         *
         * GL: the renderer's active render target is restored to nullptr after the scan.
         * Vulkan: the scan traces the renderer's path-tracing acceleration structure, so
         * the scene must have been render()-ed at least once beforehand; `scene` is then
         * unused (the TLAS is traced, not the scene graph).
         */
        void scan(Renderer& renderer, Scene& scene, std::vector<Vector3>& cloud);

        /**
         * RGB-D scan: returns hit points in world space and their corresponding
         * sRGB colors sampled from the scene color buffer.
         *
         * colors[i] matches cloud[i] — both vectors are cleared and filled together.
         */
        void scan(Renderer& renderer, Scene& scene, std::vector<Vector3>& cloud, std::vector<Color>& colors);

        [[nodiscard]] unsigned int width() const { return width_; }
        [[nodiscard]] unsigned int height() const { return height_; }
        [[nodiscard]] float fov() const { return camera_.fov; }
        [[nodiscard]] float near() const { return camera_.nearPlane; }
        [[nodiscard]] float far() const { return camera_.farPlane; }

        Camera& getCamera() { return camera_; }

    private:
        unsigned int width_;
        unsigned int height_;

        Scene postScene_;
        OrthographicCamera postCamera_;
        PerspectiveCamera camera_;
        std::unique_ptr<RenderTarget> sceneTarget_;
        std::unique_ptr<RenderTarget> readbackTarget_;
        std::shared_ptr<ShaderMaterial> postMaterial_;

        // Precomputed per-column and per-row view-space ray direction factors
        std::vector<float> xDir_;
        std::vector<float> yDir_;

        void unprojectPoints(std::vector<Vector3>& points,
                             const unsigned char* colorPixels = nullptr,
                             std::vector<Color>* colors = nullptr) const;
    };

}// namespace threepp

#endif//THREEPP_LIDAR_HPP
