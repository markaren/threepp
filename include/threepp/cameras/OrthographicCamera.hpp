// https://github.com/mrdoob/three.js/blob/r129/src/cameras/OrthographicCamera.js

#ifndef THREEPP_ORTHOGRAPHICCAMERA_HPP
#define THREEPP_ORTHOGRAPHICCAMERA_HPP

#include "threepp/cameras/Camera.hpp"

#include <memory>

namespace threepp {

    /*
     * Camera that uses orthographic projection.
     *
     * In this projection mode, an object's size in the rendered image stays constant regardless of its distance from the camera.
     * This can be useful for rendering 2D scenes and UI elements, amongst other things.
     */
    class OrthographicCamera: public Camera {

    public:
        int left;
        int right;
        int top;
        int bottom;

        explicit OrthographicCamera(int left = -1, int right = 1,
                                    int top = 1, int bottom = -1,
                                    float near = 0.1f, float far = 2000);

        // Sets an offset in a larger viewing frustum.
        // This is useful for multi-window or multi-monitor/multi-machine setups.
        void setViewOffset(int fullWidth, int fullHeight, int x, int y, int width, int height);

        // Removes any offset set by the .setViewOffset method.
        void clearViewOffset();

        // Updates the camera projection matrix. Must be called after any change of parameters.
        void updateProjectionMatrix() override;

        static std::shared_ptr<OrthographicCamera> create(
                int left = -1, int right = 1,
                int top = 1, int bottom = -1,
                float near = 0.1f, float far = 2000);
    };

}// namespace threepp

#endif//THREEPP_ORTHOGRAPHICCAMERA_HPP
