// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/controls/TrackballControls.js

#ifndef THREEPP_TRACKBALLCONTROLS_HPP
#define THREEPP_TRACKBALLCONTROLS_HPP

#include "threepp/math/Vector3.hpp"

#include <limits>
#include <memory>

namespace threepp {

    class Camera;
    class PeripheralsEventSource;

    // Free-rotation camera controls.
    //
    // Where OrbitControls keeps camera.up pinned and parameterizes the orbit as azimuth/polar
    // angles - so the horizon never tilts and the poles have to be clamped - TrackballControls
    // rotates about an arbitrary axis derived from the screen-space drag and carries camera.up
    // along with it. The result is full SO(3): the view can roll, and you can tumble straight
    // through the poles without gimbal lock.
    //
    // That makes it the right tool for inspecting a single object with no meaningful "up"
    // (CAD parts, meshes, scans, point clouds), and the wrong one for scenes with a ground
    // plane, where the free roll just feels seasick. Prefer OrbitControls for those.
    //
    // Unlike OrbitControls, update() must be called every frame: rotate/zoom/pan are eased
    // towards their targets (see staticMoving), and rotation keeps spinning after release.
    class TrackballControls {

    public:
        bool enabled = true;

        // The point the camera orbits, and which pan drags around.
        Vector3 target;

        float minDistance = 0.f;
        float maxDistance = std::numeric_limits<float>::infinity();

        // Orthographic cameras zoom by changing camera.zoom rather than distance.
        float minZoom = 0.f;
        float maxZoom = std::numeric_limits<float>::infinity();

        bool enableRotate = true;
        float rotateSpeed = 1.f;

        bool enableZoom = true;
        float zoomSpeed = 1.2f;

        bool enablePan = true;
        float panSpeed = 0.3f;

        // Set to true to stop dead on mouse-up. Otherwise motion is eased by
        // dynamicDampingFactor and rotation carries on spinning (trackball inertia).
        bool staticMoving = false;
        float dynamicDampingFactor = 0.2f;

        TrackballControls(Camera& camera, PeripheralsEventSource& eventSource);

        // Applies pending input to the camera. Call once per frame.
        // Returns true if the camera moved.
        bool update();

        // Restores target, position, up and zoom as they were at construction.
        void reset();

        [[nodiscard]] float getDistance() const;

        ~TrackballControls();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TRACKBALLCONTROLS_HPP
