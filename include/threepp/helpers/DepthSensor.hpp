#ifndef THREEPP_DEPTH_SENSOR_HPP
#define THREEPP_DEPTH_SENSOR_HPP

#include "CameraHelper.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/extras/sensors/TracedRasterVisionSensor.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"


#include <memory>
#include <vector>

namespace threepp {

    class ShaderMaterial;

    /**
     * Simulates a depth sensor, backend-neutrally.
     *
     * The scan protocol — raster/traced dispatch, the scanBegin/scanReady/
     * scanCollect fire-and-deliver pair, resetNoise cascading — is inherited
     * from TracedRasterVisionSensor; this class supplies the pinhole beam
     * geometry:
     *   - raster: renders the scene from the sensor's viewpoint into a depth
     *     texture, linearizes it via a post-process shader, reads back the
     *     pixels, and reprojects them into world-space 3D points;
     *   - Vulkan: the same pinhole ray pattern (identical fovY / width /
     *     height and the look-down-local-(-Z) convention) is traced through
     *     the renderer's path-tracing acceleration structure. The result is
     *     the same world-space point cloud, so perception/deploy code is
     *     identical on either backend.
     *
     * Aiming: beams go down the sensor's LOCAL -Z (camera convention), but the
     * sensor is an Object3D, not a Camera, so Object3D::lookAt applies the
     * non-camera convention and turns local +Z toward the target — aiming the
     * sensor exactly backwards, with an empty cloud as the only symptom. Aim
     * via rotation/quaternion, or reflect the target through the sensor
     * position: lookAt(2*position - target). (Deliberately not "fixed" in
     * lookAt itself: existing callers already compensate.)
     *
     * Can scan with or without color: the former is slightly more expensive but
     * gives per-point color information, while the latter is faster and uses
     * less GPU memory. (On Vulkan the "color" is the LIDAR intensity as
     * greyscale, not surface albedo.)
     *
     * Range noise is inherited from VisionSensor: `rangeNoise` is a seeded
     * RangeNoiseModel (default: 0.02 m sigma), so the same scene scanned from
     * the same pose with the same seed produces the same cloud on every run and
     * every machine. Two default-constructed sensors share the default seed —
     * give them distinct seeds if you want their noise decorrelated.
     */
    class DepthSensor: public TracedRasterVisionSensor<Vector3> {

    public:
        DepthSensor(float fovY, unsigned int width, unsigned int height,
                    float near = 0.1f, float far = 100.f);

        ~DepthSensor() override;

        // The point-cloud scan family (scan / scanBegin / scanReady /
        // scanCollect) is inherited; the RGB-D overload below would otherwise
        // hide it.
        using TracedRasterVisionSensor<Vector3>::scan;

        /**
         * RGB-D scan: returns hit points in world space and their corresponding
         * sRGB colors sampled from the scene color buffer.
         *
         * colors[i] matches cloud[i] — both vectors are cleared and filled
         * together.
         *
         * The frame-loop fire/deliver pair is not offered for RGB-D: the colour
         * half is a raster readback of the scene colour buffer with nothing to
         * pipeline.
         */
        void scan(Renderer& renderer, Scene& scene, std::vector<Vector3>& cloud, std::vector<Color>& colors);

        [[nodiscard]] unsigned int width() const { return width_; }
        [[nodiscard]] unsigned int height() const { return height_; }
        [[nodiscard]] float fov() const { return camera_.fov; }

        // The camera the raster path rasterizes from. Its near plane is
        // deliberately pulled in from near(): a plane clips on perpendicular
        // view-space Z, while the sensor's near is a blind SPHERE, and an
        // off-axis surface can sit outside the sphere while being inside the
        // plane. Read near()/far() for the sensor's own bounds; this frustum is
        // an implementation detail that is merely guaranteed to contain them.
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

        // TracedRasterVisionSensor's hooks: the pinhole beam pattern, the
        // depth-render + unproject raster pass, and the tracer-returns → point
        // conversion (drop sentinels, keep positions).
        std::unique_ptr<PathTracedLidarSensor> createTracedBackend() override;
        void rasterScan(Renderer& renderer, Scene& scene, std::vector<Vector3>& cloud) override;
        bool collectTraced(PathTracedLidarSensor& backend, VulkanRenderer& renderer, std::vector<Vector3>& cloud) override;

        void unprojectPoints(std::vector<Vector3>& points,
                             const unsigned char* colorPixels = nullptr,
                             std::vector<Color>* colors = nullptr);
    };

}// namespace threepp

#endif//THREEPP_DEPTH_SENSOR_HPP
