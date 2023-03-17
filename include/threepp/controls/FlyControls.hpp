
#ifndef THREEPP_FLYCONTROLS_HPP
#define THREEPP_FLYCONTROLS_HPP

#include "threepp/math/Vector3.hpp"
#include "threepp/math/infinity.hpp"
#include "threepp/math/MathUtils.hpp"

#include <memory>

namespace threepp {

    class Camera;
    class Canvas;

    class FlyControls {

    public:

        // Set to false to disable this control
        bool enabled = true;

        // "target" sets the location of focus, where the object orbits around
        Vector3 target;

        // How far you can dolly in and out ( PerspectiveCamera only )
        float minDistance = 0;
        float maxDistance = Infinity<float>;

        // How far you can zoom in and out ( OrthographicCamera only )
        float minZoom = 0;
        float maxZoom = Infinity<float>;

        // How far you can orbit vertically, upper and lower limits.
        // Range is 0 to Math.PI radians.
        float minPolarAngle = 0; // radians
        float maxPolarAngle = math::PI; // radians

        // How far you can orbit horizontally, upper and lower limits.
        // If set, the interval [ min, max ] must be a sub-interval of [ - 2 PI, 2 PI ], with ( max - min < 2 PI )
        float minAzimuthAngle = - Infinity<float>; // radians
        float maxAzimuthAngle = Infinity<float>; // radians

        // Set to true to enable damping (inertia)
        // If damping is enabled, you must call controls.update() in your animation loop
        bool enableDamping = false;
        float dampingFactor = 0.05f;

        // This option actually enables dollying in and out; left as "zoom" for backwards compatibility.
        // Set to false to disable zooming
        bool enableZoom = true;
        float zoomSpeed = 1.0f;

        // Set to false to disable rotating
        bool enableRotate = true;
        float rotateSpeed = 1.0f;

        // Set to false to disable panning
        bool enablePan = true;
        float panSpeed = 1.0f;
        bool screenSpacePanning = true; // if false, pan orthogonal to world-space direction camera.up
        float keyPanSpeed = 7.0f;	// pixels moved per arrow key push

        // Set to true to automatically rotate around the target
        // If auto-rotate is enabled, you must call controls.update() in your animation loop
        bool autoRotate = false;
        float autoRotateSpeed = 2.0f; // 30 seconds per orbit when fps is 60

        float movementSpeed = 1.0f;
        float rollSpeed = 0.005f;

        bool dragToLook = false;
        bool autoForward = false;

        FlyControls(Camera* camera, Canvas& canvas);

        void update(float delta);

        float getPolarAngle();

        float getAzimuthalAngle();

        ~FlyControls();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}

#endif//THREEPP_FLYCONTROLS_HPP
