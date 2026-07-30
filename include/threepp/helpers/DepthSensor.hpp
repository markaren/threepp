#ifndef THREEPP_DEPTH_SENSOR_HPP
#define THREEPP_DEPTH_SENSOR_HPP

#include "CameraHelper.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/VisionSensor.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"


#include <memory>
#include <vector>

namespace threepp {

    class ShaderMaterial;
    class Renderer;
    class PathTracedLidarSensor;

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
     *
     * Range noise is inherited from VisionSensor: `rangeNoise` is a seeded
     * RangeNoiseModel (default: 0.02 m sigma), so the same scene scanned from
     * the same pose with the same seed produces the same cloud on every run and
     * every machine. Two default-constructed sensors share the default seed —
     * give them distinct seeds if you want their noise decorrelated.
     */
    class DepthSensor: public Object3D, public VisionSensor {

    public:
        DepthSensor(float fovY, unsigned int width, unsigned int height,
                    float near = 0.1f, float far = 100.f);

        // Out-of-line: the cached path-traced back-end is only forward-declared.
        ~DepthSensor() override;

        // Also re-seeds the Vulkan back-end, which owns the stream on that path.
        void resetNoise() override;

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

        // The range shell the sensor reports in: a return's distance from the
        // sensor origin always lies in [near(), far()], on every backend. These
        // are RANGES, not view-space planes — see the raster camera note below.
        [[nodiscard]] float near() const { return near_; }
        [[nodiscard]] float far() const { return far_; }

        // The camera the GL path rasterizes from. Its near plane is deliberately
        // pulled in from near(): a plane clips on perpendicular view-space Z,
        // while the sensor's near is a blind SPHERE, and an off-axis surface can
        // sit outside the sphere while being inside the plane. Read near()/far()
        // for the sensor's own bounds; this frustum is an implementation detail
        // that is merely guaranteed to contain them.
        Camera& getCamera() { return camera_; }

    private:
        unsigned int width_;
        unsigned int height_;

        // The reported range shell. Distinct from camera_.nearPlane, which is
        // the (wider) raster frustum used to fill the depth buffer.
        float near_;
        float far_;

        Scene postScene_;
        OrthographicCamera postCamera_;
        PerspectiveCamera camera_;
        std::unique_ptr<RenderTarget> sceneTarget_;
        std::unique_ptr<RenderTarget> readbackTarget_;
        std::shared_ptr<ShaderMaterial> postMaterial_;

        // Precomputed per-column and per-row view-space ray direction factors
        std::vector<float> xDir_;
        std::vector<float> yDir_;

        // Vulkan back-end: the pinhole ray pattern traced through the path
        // tracer's TLAS. Built on first use and kept, not rebuilt per scan —
        // rebuilding would also rebuild the beam table every frame.
        std::unique_ptr<PathTracedLidarSensor> tracedBackend_;

        // Builds tracedBackend_ on first call and syncs the noise model onto it.
        // Only defined on a Vulkan build (the back-end's TU is Vulkan-gated).
        PathTracedLidarSensor& tracedBackend();

        void unprojectPoints(std::vector<Vector3>& points,
                             const unsigned char* colorPixels = nullptr,
                             std::vector<Color>* colors = nullptr);
    };

}// namespace threepp

#endif//THREEPP_DEPTH_SENSOR_HPP
