// https://github.com/mrdoob/three.js/blob/r129/src/cameras/StereoCamera.js

#ifndef THREEPP_STEREOCAMERA_HPP
#define THREEPP_STEREOCAMERA_HPP

#include "threepp/cameras/PerspectiveCamera.hpp"

namespace threepp {

    /*
     * Two perspective cameras, one per eye, derived from a single camera:
     * for side-by-side 3D output, anaglyphs and head-mounted displays.
     *
     * The eyes sit eyeSep apart on the camera's local x axis and look in
     * the camera's direction (parallel axes). Their frusta are off-axis:
     * both cover the same rectangle at the distance camera.focus, so a
     * point at that distance has zero disparity between the two pictures.
     *
     * Each frame: bring the camera's world matrix up to date, call
     * update(camera), then render the scene once with cameraL into the left
     * half of the output and once with cameraR into the right half.
     */
    class StereoCamera {

    public:
        // Multiplies camera.aspect to give one eye's aspect. 0.5 when the
        // camera's aspect is that of a window the two eyes share side by
        // side; 1 when it is already one eye's.
        float aspect = 1;

        // Distance between the eyes, in world units.
        float eyeSep = 0.064f;

        // Layer 1 is enabled on cameraL and layer 2 on cameraR, so an object
        // can be shown to one eye only.
        PerspectiveCamera cameraL;
        PerspectiveCamera cameraR;

        StereoCamera();

        // Reads fov, aspect, near, far, zoom, focus and matrixWorld from
        // `camera`. matrixWorld must be current (updateMatrixWorld), which a
        // camera that is not in the scene graph does not get from a render.
        void update(const PerspectiveCamera& camera);

    private:
        struct Cache {
            float focus, fov, aspect, nearPlane, farPlane, zoom, eyeSep;
            bool operator==(const Cache&) const = default;
        };

        Cache cache_{};
        bool cacheValid_ = false;
    };

}// namespace threepp

#endif//THREEPP_STEREOCAMERA_HPP
