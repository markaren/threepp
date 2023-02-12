
#ifndef THREEPP_ORBITCONTROLS_HPP
#define THREEPP_ORBITCONTROLS_HPP

#include "threepp/Canvas.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/EventDispatcher.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include <threepp/math/Spherical.hpp>

#include <memory>
#include <limits>

namespace threepp {

    class OrbitControls {

    public:
        bool enabled = true;

        Vector3 target{};

        float minDistance = 0.f;
        float maxDistance = std::numeric_limits<float>::infinity();

        float minZoom = 0.f;
        float maxZoom = std::numeric_limits<float>::infinity();

        float minPolarAngle = 0.f;
        float maxPolarAngle = math::PI;

        // How far you can orbit horizontally, upper and lower limits.
        // If set, must be a sub-interval of the interval [ - Math.PI, Math.PI ].
        float minAzimuthAngle = -std::numeric_limits<float>::infinity();// radians
        float maxAzimuthAngle = std::numeric_limits<float>::infinity(); // radians

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
        bool screenSpacePanning = false;// if true, pan in screen-space
        float keyPanSpeed = 7.f;        // pixels moved per arrow key push

        // Set to true to automatically rotate around the target
        // If auto-rotate is enabled, you must call controls.update() in your animation loop
        bool autoRotate = false;
        float autoRotateSpeed = 2.0f;// 30 seconds per round when fps is 60

        // Set to false to disable use of the keys
        bool enableKeys = true;

        enum State {

            NONE,
            ROTATE,
            DOLLY,
            PAN,

        };

        State state = State::NONE;

        OrbitControls(std::shared_ptr<Camera> camera, Canvas &canvas);

        bool update();

        [[nodiscard]] float getAutoRotationAngle() const;

        [[nodiscard]] float getZoomScale() const;

        void rotateLeft(float angle);

        void rotateUp(float angle);

        void panLeft(float distance, const Matrix4 &objectMatrix);

        void panUp(float distance, const Matrix4 &objectMatrix);

        // deltaX and deltaY are in pixels; right and down are positive
        void pan(float deltaX, float deltaY);

        void dollyIn(float dollyScale);

        void dollyOut(float dollyScale);

        ~OrbitControls();

    private:
        Canvas &canvas;
        std::shared_ptr<Camera> camera;

        std::unique_ptr<KeyListener> keyListener;
        std::unique_ptr<MouseListener> mouseListener;

        // current position in spherical coordinates
        Spherical spherical{};
        Spherical sphericalDelta{};

        float scale = 1.f;
        Vector3 panOffset{};
        bool zoomChanged = false;

        Vector2 rotateStart{};
        Vector2 rotateEnd{};
        Vector2 rotateDelta{};

        Vector2 panStart{};
        Vector2 panEnd{};
        Vector2 panDelta{};

        Vector2 dollyStart{};
        Vector2 dollyEnd{};
        Vector2 dollyDelta{};

        void handleKeyDown(int key);

        void handleMouseDownRotate(const Vector2 &pos);

        void handleMouseDownDolly(const Vector2 &pos);

        void handleMouseDownPan(const Vector2 &pos);

        void handleMouseMoveRotate(const Vector2 &pos);

        void handleMouseMoveDolly(const Vector2 &pos);

        void handleMouseMovePan(const Vector2 &pos);

        void handleMouseWheel(const Vector2 &delta);

        struct MyKeyListener : KeyListener {

            explicit MyKeyListener(OrbitControls &scope)
                : scope(scope) {}

            void onKeyPressed(KeyEvent evt) override {
                if (scope.enabled && scope.enableKeys && scope.enablePan) {
                    scope.handleKeyDown(evt.key);
                }
            }

            void onKeyRepeat(KeyEvent evt) override {
                if (scope.enabled && scope.enableKeys && scope.enablePan) {
                    scope.handleKeyDown(evt.key);
                }
            }

        private:
            OrbitControls &scope;
        };

        struct MyMouseMoveListener: MouseListener {

            explicit MyMouseMoveListener(OrbitControls &scope) : scope(scope) {}

            void onMouseMove(const Vector2& pos) override {
                if (scope.enabled) {

                    switch (scope.state) {
                        case State::ROTATE:
                            if (scope.enableRotate) {
                                scope.handleMouseMoveRotate(pos);
                            }
                            break;
                        case State::DOLLY:
                            if (scope.enableZoom) {
                                scope.handleMouseMoveDolly(pos);
                            }
                            break;
                        case State::PAN:
                            if (scope.enablePan) {
                                scope.handleMouseMovePan(pos);
                            }
                            break;
                        default:
                            //TODO ?
                            break;
                    }
                }
            }

        private:
            OrbitControls &scope;
        };


        struct MyMouseUpListener: MouseListener {

            explicit MyMouseUpListener(OrbitControls &scope, MyMouseMoveListener* mouseMoveListener)
                : scope(scope), mouseMoveListener(mouseMoveListener) {}

            void onMouseUp(int button, const Vector2& pos) override {
                if (scope.enabled) {

                    scope.canvas.removeMouseListener(mouseMoveListener);
                    scope.canvas.removeMouseListener(this);
                    scope.state = State::NONE;
                }
            }

        private:
            OrbitControls &scope;
            MouseListener* mouseMoveListener;
        };

        struct MyMouseListener : MouseListener {

            explicit MyMouseListener(OrbitControls &scope)
                : scope(scope), mouseMoveListener(scope), mouseUpListener(scope, &mouseMoveListener) {}

            void onMouseDown(int button, const Vector2& pos) override {
                if (scope.enabled) {
                    switch (button) {
                        case 0: // LEFT
                            if (scope.enableRotate) {
                                scope.handleMouseDownRotate(pos);
                                scope.state = State::ROTATE;
                            }
                            break;
                        case 1: // RIGHT
                            if (scope.enablePan) {
                                scope.handleMouseDownRotate(pos);
                                scope.handleMouseDownPan(pos);
                                scope.state = State::PAN;
                            }
                            break;
                        case 2: // MIDDLE
                            if (scope.enableZoom) {
                                scope.handleMouseDownDolly(pos);
                                scope.state = State::DOLLY;
                            }
                            break;
                    }
                }

                if (scope.state != State::NONE) {

                    scope.canvas.addMouseListener(&mouseMoveListener);
                    scope.canvas.addMouseListener(&mouseUpListener);

                }

            }

            void onMouseWheel(const Vector2& delta) override {
                if (scope.enabled && scope.enableZoom && !(scope.state != State::NONE && scope.state != State::ROTATE)) {
                    scope.handleMouseWheel(delta);
                }
            }

        private:
            OrbitControls &scope;
            MyMouseMoveListener mouseMoveListener;
            MyMouseUpListener mouseUpListener;
        };

    };

}// namespace threepp

#endif//THREEPP_ORBITCONTROLS_HPP
