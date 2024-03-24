
#include "threepp/controls/OrbitControls.hpp"

#include "threepp/input/PeripheralsEventSource.hpp"
#include "threepp/math/Spherical.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"

#include <cmath>
#include <iostream>

using namespace threepp;

namespace {

    const float EPS = 0.000001f;

    enum State {

        NONE,
        ROTATE,
        DOLLY,
        PAN,

    };


}// namespace

struct OrbitControls::Impl {

    PeripheralsEventSource& canvas;
    OrbitControls& scope;
    Camera& camera;

    State state = State::NONE;

    // current position in spherical coordinates
    Spherical spherical;
    Spherical sphericalDelta;

    float scale = 1.f;
    Vector3 panOffset;
    bool zoomChanged = false;

    Vector2 rotateStart;
    Vector2 rotateEnd;
    Vector2 rotateDelta;

    Vector2 panStart;
    Vector2 panEnd;
    Vector2 panDelta;

    Vector2 dollyStart;
    Vector2 dollyEnd;
    Vector2 dollyDelta;

    Subscriptions subs_;

    Impl(OrbitControls& scope, PeripheralsEventSource& canvas, Camera& camera)
        : scope(scope), canvas(canvas), camera(camera) {

        subs_ << canvas.keys.Pressed.subscribe([this](auto& e) { onKeyPressed(e); });
        subs_ << canvas.keys.Repeat.subscribe([this](auto& e) { onKeyRepeat(e); });

        subs_ << canvas.mouse.Down.subscribe([this](auto& e) { onMouseDown(e); });
        subs_ << canvas.mouse.Wheel.subscribe([this](auto& e) { onMouseWheel(e); });

        update();
    }

    bool update() {

        Vector3 offset;

        // so camera.up is the orbit axis
        const auto quat = Quaternion().setFromUnitVectors(camera.up, {0, 1, 0});
        auto quatInverse = Quaternion().copy(quat).invert();

        Vector3 lastPosition;
        Quaternion lastQuaternion;

        auto& position = this->camera.position;

        offset.copy(position).sub(scope.target);

        // rotate offset to "y-axis-is-up" space
        offset.applyQuaternion(quat);

        // angle from z-axis around y-axis
        spherical.setFromVector3(offset);

        if (scope.autoRotate && state == State::NONE) {

            rotateLeft(scope.getAutoRotationAngle());
        }

        if (scope.enableDamping) {

            spherical.theta += sphericalDelta.theta * scope.dampingFactor;
            spherical.phi += sphericalDelta.phi * scope.dampingFactor;

        } else {

            spherical.theta += sphericalDelta.theta;
            spherical.phi += sphericalDelta.phi;
        }

        // restrict theta to be between desired limits
        spherical.theta = std::max(scope.minAzimuthAngle, std::min(scope.maxAzimuthAngle, spherical.theta));

        // restrict phi to be between desired limits
        spherical.phi = std::max(scope.minPolarAngle, std::min(scope.maxPolarAngle, spherical.phi));

        spherical.makeSafe();

        spherical.radius *= scale;

        // restrict radius to be between desired limits
        spherical.radius = std::max(scope.minDistance, std::min(scope.maxDistance, spherical.radius));

        // move target to panned location

        if (scope.enableDamping) {

            scope.target.addScaledVector(panOffset, scope.dampingFactor);

        } else {

            scope.target.add(panOffset);
        }

        offset.setFromSpherical(spherical);

        // rotate offset back to "camera-up-vector-is-up" space
        offset.applyQuaternion(quatInverse);

        position.copy(scope.target).add(offset);

        this->camera.lookAt(scope.target);

        if (scope.enableDamping) {

            sphericalDelta.theta *= (1 - scope.dampingFactor);
            sphericalDelta.phi *= (1 - scope.dampingFactor);

            panOffset.multiplyScalar(1 - scope.dampingFactor);

        } else {

            sphericalDelta.set(0, 0, 0);

            panOffset.set(0, 0, 0);
        }

        scale = 1;

        // update condition is:
        // min(camera displacement, camera rotation in radians)^2 > EPS
        // using small-angle approximation cos(x/2) = 1 - x^2 / 8

        if (zoomChanged ||
            lastPosition.distanceToSquared(this->camera.position) > EPS ||
            8 * (1 - lastQuaternion.dot(this->camera.quaternion)) > EPS) {

            lastPosition.copy(this->camera.position);
            lastQuaternion.copy(this->camera.quaternion);
            zoomChanged = false;

            return true;
        }

        return false;
    }

    void rotateLeft(float angle) {

        sphericalDelta.theta -= angle;
    }

    void rotateUp(float angle) {

        sphericalDelta.phi -= angle;
    }

    void panLeft(float distance, const Matrix4& objectMatrix) {

        Vector3 v;

        v.setFromMatrixColumn(objectMatrix, 0);// get X column of objectMatrix
        v.multiplyScalar(-distance);

        panOffset.add(v);
    }

    void panUp(float distance, const Matrix4& objectMatrix) {

        Vector3 v;

        if (scope.screenSpacePanning) {

            v.setFromMatrixColumn(objectMatrix, 1);

        } else {

            v.setFromMatrixColumn(objectMatrix, 0);
            v.crossVectors(this->camera.up, v);
        }

        v.multiplyScalar(distance);

        panOffset.add(v);
    }

    void pan(float deltaX, float deltaY) {

        Vector3 offset;

        if (auto perspective = camera.as<PerspectiveCamera>()) {

            // perspective
            auto& position = this->camera.position;
            offset.copy(position).sub(scope.target);
            auto targetDistance = offset.length();

            // half of the fov is center to top of screen
            targetDistance *= std::tan((perspective->fov / 2) * math::PI / 180.f);

            // we use only clientHeight here so aspect ratio does not distort speed
            const auto size = canvas.size();
            panLeft(2 * deltaX * targetDistance / (float) size.height, *this->camera.matrix);
            panUp(2 * deltaY * targetDistance / (float) size.height, *this->camera.matrix);
        } else if (auto ortho = camera.as<OrthographicCamera>()) {

            const auto size = canvas.size();

            // orthographic
            panLeft(
                    deltaX * (ortho->right - ortho->left) / this->camera.zoom / size.width,
                    *this->camera.matrix);
            panUp(
                    deltaY * (ortho->top - ortho->bottom) / this->camera.zoom / size.height,
                    *this->camera.matrix);

        } else {

            // camera neither orthographic nor perspective
            std::cerr << "[OrbitControls] encountered an unknown camera type - pan disabled." << std::endl;
            scope.enablePan = false;
        }
    }

    void dollyIn(float dollyScale) {

        if (camera.as<PerspectiveCamera>()) {

            scale /= dollyScale;

        } else if (camera.as<OrthographicCamera>()) {

            this->camera.zoom = std::max(scope.minZoom, std::min(scope.maxZoom, this->camera.zoom * dollyScale));
            this->camera.updateProjectionMatrix();
            zoomChanged = true;
        } else {

            std::cerr << "[OrbotControls] encountered an unknown camera type - dolly/zoom disabled." << std::endl;
            scope.enableZoom = false;
        }
    }

    void dollyOut(float dollyScale) {

        if (camera.as<PerspectiveCamera>()) {
            scale *= dollyScale;
        } else if (camera.as<OrthographicCamera>()) {
            this->camera.zoom = std::max(scope.minZoom, std::min(scope.maxZoom, this->camera.zoom / dollyScale));
            this->camera.updateProjectionMatrix();
            zoomChanged = true;
        } else {
            std::cerr << "[OrbotControls] encountered an unknown camera type - dolly/zoom disabled." << std::endl;
            scope.enableZoom = false;
        }
    }


    void handleKeyDown(Key key) {

        bool needsUpdate = true;

        switch (key) {

            case Key::UP:
                pan(0, scope.keyPanSpeed);
                break;
            case Key::DOWN:
                pan(0, -scope.keyPanSpeed);
                break;
            case Key::LEFT:
                pan(scope.keyPanSpeed, 0);
                break;
            case Key::RIGHT:
                pan(-scope.keyPanSpeed, 0);
                break;
            default:
                needsUpdate = false;
                break;
        }

        if (needsUpdate) {
            scope.update();
        }
    }

    void handleMouseDownRotate(const Vector2& pos) {

        rotateStart.copy(pos);
    }

    void handleMouseDownDolly(const Vector2& pos) {

        dollyStart.copy(pos);
    }

    void handleMouseDownPan(const Vector2& pos) {

        panStart.copy(pos);
    }

    void handleMouseMoveRotate(const Vector2& pos) {

        rotateEnd.copy(pos);

        rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed);

        const auto size = canvas.size();
        rotateLeft(2 * math::PI * rotateDelta.x / static_cast<float>(size.height));// yes, height

        rotateUp(2 * math::PI * rotateDelta.y / static_cast<float>(size.height));

        rotateStart.copy(rotateEnd);

        scope.update();
    }

    void handleMouseMoveDolly(const Vector2& pos) {

        dollyEnd.copy(pos);

        dollyDelta.subVectors(dollyEnd, dollyStart);

        if (dollyDelta.y > 0) {

            dollyIn(scope.getZoomScale());

        } else if (dollyDelta.y < 0) {

            dollyOut(scope.getZoomScale());
        }

        dollyStart.copy(dollyEnd);

        scope.update();
    }


    void handleMouseMovePan(const Vector2& pos) {

        panEnd.copy(pos);

        panDelta.subVectors(panEnd, panStart).multiplyScalar(scope.panSpeed);

        pan(panDelta.x, panDelta.y);

        panStart.copy(panEnd);

        scope.update();
    }

    void handleMouseWheel(const Vector2& delta) {
        if (delta.y < 0) {

            dollyIn(scope.getZoomScale());

        } else if (delta.y > 0) {

            dollyOut(scope.getZoomScale());
        }

        scope.update();
    }


    void onKeyPressed(KeyEvent evt) {
        if (scope.enabled && scope.enableKeys && scope.enablePan) {
            handleKeyDown(evt.key);
        }
    }

    void onKeyRepeat(KeyEvent evt) {
        if (scope.enabled && scope.enableKeys && scope.enablePan) {
            handleKeyDown(evt.key);
        }
    }

    void onMouseMove(MouseEvent& e) {
        if (scope.enabled) {
            switch (state) {
                case State::ROTATE:
                    if (scope.enableRotate) {
                        handleMouseMoveRotate(e.pos);
                    }
                    break;
                case State::DOLLY:
                    if (scope.enableZoom) {
                        handleMouseMoveDolly(e.pos);
                    }
                    break;
                case State::PAN:
                    if (scope.enablePan) {
                        handleMouseMovePan(e.pos);
                    }
                    break;
                default:
                    //TODO ?
                    break;
            }
        }
    }


    void onMouseUp() {
        if (scope.enabled) {
            scope.pimpl_->state = State::NONE;
        }
    }

    void onMouseDown(MouseButtonEvent e) {
        if (scope.enabled) {
            switch (e.button) {
                case 0:// LEFT
                    if (scope.enableRotate) {
                        scope.pimpl_->handleMouseDownRotate(e.pos);
                        scope.pimpl_->state = State::ROTATE;
                    }
                    break;
                case 1:// RIGHT
                    if (scope.enablePan) {
                        scope.pimpl_->handleMouseDownRotate(e.pos);
                        scope.pimpl_->handleMouseDownPan(e.pos);
                        scope.pimpl_->state = State::PAN;
                    }
                    break;
                case 2:// MIDDLE
                    if (scope.enableZoom) {
                        scope.pimpl_->handleMouseDownDolly(e.pos);
                        scope.pimpl_->state = State::DOLLY;
                    }
                    break;
            }
        }

        if (scope.pimpl_->state != State::NONE) {
            auto& mouse = scope.pimpl_->canvas.mouse;
            mouse.Up.subscribeOnce([&](MouseButtonEvent&) {
                onMouseUp();
            });
            mouse.Move.subscribeUntil(mouse.Up, [&](MouseEvent& e) {
                onMouseMove(e);
            });
        }
    }

    void onMouseWheel(MouseWheelEvent& ev) {
        if (scope.enabled && scope.enableZoom && !(scope.pimpl_->state != State::NONE && scope.pimpl_->state != State::ROTATE)) {
            scope.pimpl_->handleMouseWheel(ev.offset);
        }
    }
};

OrbitControls::OrbitControls(Camera& camera, PeripheralsEventSource& eventSource)
    : pimpl_(std::make_unique<Impl>(*this, eventSource, camera)) {}


bool OrbitControls::update() {

    return pimpl_->update();
}

float OrbitControls::getAutoRotationAngle() const {

    return 2 * math::PI / 60 / 60 * this->autoRotateSpeed;
}

float OrbitControls::getZoomScale() const {

    return std::pow(0.95f, this->zoomSpeed);
}

OrbitControls::~OrbitControls() = default;
