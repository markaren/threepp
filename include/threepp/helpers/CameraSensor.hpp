#ifndef THREEPP_CAMERA_SENSOR_HPP
#define THREEPP_CAMERA_SENSOR_HPP

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/Sensor.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

namespace threepp {

    class RenderTarget;
    class Renderer;
    class Scene;
    class VulkanRenderer;

    /**
     * A colour camera bolted to a link — what the robot SEES.
     *
     * The suite's other two vision sensors both answer "how far away is that":
     * DepthSensor returns a point cloud and LidarSensor a set of ranges. Neither
     * produces a picture, and a picture is what a learned perception policy
     * takes as input, what a teleoperator looks at, and what a wrist camera
     * publishes onto an image topic. That gap is this class.
     *
     * Deliberately NOT a VisionSensor. That base exists to own a
     * RangeNoiseModel, and a range model has nothing to say about a colour
     * pixel — sigma in metres does not apply to a byte of red. What IS shared
     * with the ranging sensors is everything in Sensor: the sim clock, so a
     * frame is stamped on the same time base as the IMU sample beside it, and
     * the rate gate, so a 15 Hz camera in a 60 Hz editor captures on the right
     * frames instead of every one. Image noise, when it comes, belongs here as
     * its own model (shot noise and read noise are per-pixel and per-exposure)
     * rather than borrowed from the ranging one.
     *
     * Backend-portable, two different ways. On a raster backend the capture
     * is a render into this sensor's own RenderTarget plus a synchronous
     * readback: the frame is of THIS instant, of whatever scene the caller
     * passed. VulkanRenderer has no render targets (setRenderTarget is
     * ignored); there the sensor lazily attaches a persistent secondary view
     * (addView) and capture() collects the frame the LAST render() drew — one
     * frame of latency, the same fire/deliver model the ranging sensors have
     * on that backend, and a picture of the scene render() draws rather than
     * of the `scene` argument. Two consequences worth knowing: the first
     * capture or two after the view is created return false while it warms
     * up, and per-view visibility does not exist — hiding an object only
     * around capture() cannot keep it out of pixels that were drawn earlier.
     * In practice the editor's furniture stays out anyway: secondary views
     * skip the overlay pass, which is where the grid, gizmos and cloud
     * overlays live. A MESH visible at render time is in the picture, exactly
     * as it is in the ranging sensors' TLAS.
     *
     * The camera looks down its node's local -Z, the same convention as every
     * threepp camera and as DepthSensor, so a sensor parented to a tool flange
     * aims where the flange's -Z points.
     */
    class CameraSensor: public Object3D, public Sensor {

    public:
        CameraSensor(float fovY, unsigned int width, unsigned int height,
                     float near = 0.1f, float far = 100.f);

        ~CameraSensor() override;

        // Images along local -Z, so lookAt() aims it (camera convention).
        [[nodiscard]] bool usesCameraLookAtConvention() const override { return true; }

        /**
         * Render `scene` from this sensor's pose and read the frame back.
         *
         * True when image() holds a NEW frame. False while the Vulkan view is
         * still warming up, or before the renderer's first frame (see the
         * class note) — in which case image() keeps whatever it had, so a
         * reader sees a stale frame rather than a black one that looks like a
         * measurement.
         *
         * Restores the renderer's previously bound target, so this is safe to
         * call from the middle of a frame loop that is rendering elsewhere.
         */
        bool capture(Renderer& renderer, Scene& scene);

        /**
         * The most recent frame: tightly packed RGB8, row-major, TOP-LEFT
         * origin — row 0 is the top of the image, the convention of every image
         * file and of VulkanRenderer::readViewRGBPixels. GL hands back its
         * readback bottom-up and capture() flips it, so a caller never has to
         * ask which backend produced the bytes.
         *
         * Empty until the first successful capture. Size is width*height*3.
         */
        [[nodiscard]] const std::vector<unsigned char>& image() const { return image_; }

        [[nodiscard]] unsigned int width() const { return width_; }
        [[nodiscard]] unsigned int height() const { return height_; }

        // Frames captured so far, and the sim time of the newest one.
        [[nodiscard]] std::size_t frames() const { return frames_; }
        [[nodiscard]] double lastCaptureTime() const { return lastCaptureTime_; }

        // True when the rate gate says a frame is due. An ungated sensor
        // (rateHz 0) is always due — "capture every frame". Same contract as
        // VisionSensor::scanDue, and consumed by capture().
        [[nodiscard]] bool captureDue() const { return rateHz() <= 0.0 || due_; }

        // The camera the frame is rendered from. Its aspect follows
        // width/height; writing to it is how a caller overrides the projection
        // (an intrinsics-matched frustum, say).
        Camera& getCamera() { return camera_; }

        // Do SplatClouds appear in the picture? ON, because a camera that does
        // not see what the viewport sees is not a camera — and because the GL
        // path has no such switch to begin with, so OFF would be a backend
        // divergence rather than a policy.
        //
        // Vulkan only, where splats are a per-view opt-in
        // (VulkanRenderer::setViewSplats) whose cost is a second radix sort
        // over the whole cloud: ~0.3 ms on a 60k-splat object scan, ~8-13 ms on
        // a 5M-splat town, whatever this sensor's resolution. Clearing it is
        // the opt-out for a sensor that cannot afford that. Read on every
        // capture, so it may be flipped at any time.
        bool renderSplats = true;

        /**
         * Write the newest frame to a .png/.jpg/.bmp, creating parent
         * directories. False when there is no frame yet, on an unsupported
         * extension, or when the encode fails.
         */
        [[nodiscard]] bool writeImage(const std::filesystem::path& file) const;

        // Forget the captured frame and re-arm the rate gate, so the next
        // episode starts from nothing rather than from the last one's picture.
        void reset();

    protected:
        // A pulled sensor has nothing to do on a substep but arm its gate; the
        // capture itself needs a renderer and happens in the frame loop.
        void sample(double /*dt*/, double /*simTime*/) override { due_ = true; }

    private:
        unsigned int width_;
        unsigned int height_;

        PerspectiveCamera camera_;
        std::unique_ptr<RenderTarget> target_;

        // Vulkan only: the persistent secondary view serving capture() there,
        // and the renderer it belongs to. A bare pointer on the dock pane's
        // grounds — whoever owns the renderer outlives the sensors aimed
        // through it. Null / 0 everywhere else.
        VulkanRenderer* viewRenderer_ = nullptr;
        unsigned int viewHandle_ = 0;

        std::vector<unsigned char> image_;
        std::size_t frames_ = 0;
        double lastCaptureTime_ = 0.0;
        bool due_ = false;

        // Hand the view back to its renderer. The view's target still holds
        // the last picture, so this is also what keeps a reset() episode from
        // opening on the previous episode's frame.
        void releaseView();
    };

}// namespace threepp

#endif//THREEPP_CAMERA_SENSOR_HPP
