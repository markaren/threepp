#include "threepp/controls/TrackballControls.hpp"

#include "threepp/input/PeripheralsEventSource.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace threepp;

namespace {

    const float EPS = 0.000001f;

    enum State {

        NONE,
        ROTATE,
        ZOOM,
        PAN,
    };

}// namespace

struct TrackballControls::Impl {

    TrackballControls& scope;
    PeripheralsEventSource& canvas;
    Camera& camera;

    std::unique_ptr<MouseListener> mouseListener;

    State state = NONE;

    // vector from target to camera, the quantity every handler actually manipulates
    Vector3 eye;

    Vector2 movePrev;
    Vector2 moveCurr;

    Vector3 lastAxis;
    float lastAngle = 0;

    Vector2 zoomStart;
    Vector2 zoomEnd;

    Vector2 panStart;
    Vector2 panEnd;

    Vector3 target0;
    Vector3 position0;
    Vector3 up0;
    float zoom0;

    Vector3 lastPosition;
    float lastZoom;

    Impl(TrackballControls& scope, PeripheralsEventSource& canvas, Camera& camera)
        : scope(scope), canvas(canvas), camera(camera),
          mouseListener(std::make_unique<MyMouseListener>(scope)),
          target0(scope.target), position0(camera.position), up0(camera.up),
          zoom0(camera.zoom), lastZoom(camera.zoom) {

        canvas.addMouseListener(*mouseListener);

        update();
    }

    // Drag position projected onto the virtual trackball, in [-1, 1]-ish units with
    // y pointing up. Both axes are divided by the width so a circular mouse gesture
    // stays circular on a non-square viewport.
    [[nodiscard]] Vector2 getMouseOnCircle(const Vector2& pos) const {

        const auto size = canvas.size();
        const auto width = static_cast<float>(size.width());
        const auto height = static_cast<float>(size.height());

        return {(pos.x - width * 0.5f) / (width * 0.5f),
                (height - 2 * pos.y) / width};// width is intentional
    }

    // Drag position as a fraction of the viewport, y pointing down.
    [[nodiscard]] Vector2 getMouseOnScreen(const Vector2& pos) const {

        const auto size = canvas.size();

        return {pos.x / static_cast<float>(size.width()),
                pos.y / static_cast<float>(size.height())};
    }

    void rotateCamera() {

        Vector3 axis;
        Quaternion quaternion;

        Vector3 moveDirection{moveCurr.x - movePrev.x, moveCurr.y - movePrev.y, 0};
        float angle = moveDirection.length();

        if (angle > 0) {

            eye.copy(camera.position).sub(scope.target);

            Vector3 eyeDirection;
            Vector3 objectUpDirection;
            Vector3 objectSidewaysDirection;

            eyeDirection.copy(eye).normalize();
            objectUpDirection.copy(camera.up).normalize();
            objectSidewaysDirection.crossVectors(objectUpDirection, eyeDirection).normalize();

            objectUpDirection.setLength(moveCurr.y - movePrev.y);
            objectSidewaysDirection.setLength(moveCurr.x - movePrev.x);

            moveDirection.copy(objectUpDirection.add(objectSidewaysDirection));

            axis.crossVectors(moveDirection, eye);

            // degenerate when the drag runs parallel to the eye vector, or the camera sits
            // on the target - leave the view alone rather than build a bogus quaternion
            if (axis.lengthSq() > EPS) {

                axis.normalize();
                angle *= scope.rotateSpeed;

                quaternion.setFromAxisAngle(axis, angle);

                eye.applyQuaternion(quaternion);
                camera.up.applyQuaternion(quaternion);

                lastAxis.copy(axis);
                lastAngle = angle;
            }

        } else if (!scope.staticMoving && lastAngle != 0) {

            lastAngle *= std::sqrt(1.f - scope.dynamicDampingFactor);

            eye.copy(camera.position).sub(scope.target);

            quaternion.setFromAxisAngle(lastAxis, lastAngle);

            eye.applyQuaternion(quaternion);
            camera.up.applyQuaternion(quaternion);
        }

        movePrev.copy(moveCurr);
    }

    void zoomCamera() {

        const float factor = 1.f + (zoomEnd.y - zoomStart.y) * scope.zoomSpeed;

        if (factor != 1.f && factor > 0.f) {

            if (camera.as<PerspectiveCamera>()) {

                eye.multiplyScalar(factor);

            } else if (camera.as<OrthographicCamera>()) {

                camera.zoom = std::clamp(camera.zoom / factor, scope.minZoom, scope.maxZoom);
                camera.updateProjectionMatrix();

            } else {

                std::cerr << "[TrackballControls] encountered an unknown camera type - zoom disabled." << std::endl;
                scope.enableZoom = false;
                return;
            }
        }

        if (scope.staticMoving) {

            zoomStart.copy(zoomEnd);

        } else {

            zoomStart.y += (zoomEnd.y - zoomStart.y) * scope.dynamicDampingFactor;
        }
    }

    void panCamera() {

        Vector2 mouseChange;
        mouseChange.copy(panEnd).sub(panStart);

        if (mouseChange.lengthSq() > 0) {

            if (auto ortho = camera.as<OrthographicCamera>()) {

                // the drag is already a fraction of the viewport, so the frustum extent
                // (scaled by zoom) converts it straight into world units
                mouseChange.x *= (ortho->right - ortho->left) / camera.zoom;
                mouseChange.y *= (ortho->top - ortho->bottom) / camera.zoom;
                mouseChange.multiplyScalar(scope.panSpeed);

            } else {

                mouseChange.multiplyScalar(eye.length() * scope.panSpeed);
            }

            Vector3 pan;
            Vector3 objectUp;

            pan.copy(eye).cross(camera.up).setLength(mouseChange.x);
            pan.add(objectUp.copy(camera.up).setLength(mouseChange.y));

            camera.position.add(pan);
            scope.target.add(pan);

            if (scope.staticMoving) {

                panStart.copy(panEnd);

            } else {

                Vector2 delta;
                panStart.add(delta.subVectors(panEnd, panStart).multiplyScalar(scope.dynamicDampingFactor));
            }
        }
    }

    void checkDistances() {

        if (!scope.enableZoom && !scope.enablePan) return;

        if (eye.lengthSq() > scope.maxDistance * scope.maxDistance) {

            camera.position.addVectors(scope.target, eye.setLength(scope.maxDistance));
            zoomStart.copy(zoomEnd);
        }

        if (eye.lengthSq() < scope.minDistance * scope.minDistance) {

            camera.position.addVectors(scope.target, eye.setLength(scope.minDistance));
            zoomStart.copy(zoomEnd);
        }
    }

    bool update() {

        eye.subVectors(camera.position, scope.target);

        if (scope.enableRotate) rotateCamera();

        if (scope.enableZoom) zoomCamera();

        if (scope.enablePan) panCamera();

        camera.position.addVectors(scope.target, eye);

        if (camera.as<PerspectiveCamera>()) {

            checkDistances();
        }

        camera.lookAt(scope.target);

        if (lastPosition.distanceToSquared(camera.position) > EPS || lastZoom != camera.zoom) {

            lastPosition.copy(camera.position);
            lastZoom = camera.zoom;

            return true;
        }

        return false;
    }

    void reset() {

        state = NONE;

        scope.target.copy(target0);
        camera.position.copy(position0);
        camera.up.copy(up0);
        camera.zoom = zoom0;
        camera.updateProjectionMatrix();

        // drop any in-flight easing, or the view would drift straight back out
        moveCurr.set(0, 0);
        movePrev.set(0, 0);
        lastAngle = 0;
        zoomStart.set(0, 0);
        zoomEnd.set(0, 0);
        panStart.set(0, 0);
        panEnd.set(0, 0);

        eye.subVectors(camera.position, scope.target);

        camera.lookAt(scope.target);

        lastPosition.copy(camera.position);
        lastZoom = camera.zoom;
    }

    [[nodiscard]] float getDistance() const {

        return camera.position.distanceTo(scope.target);
    }

    ~Impl() {

        canvas.removeMouseListener(*mouseListener);
    }

    struct MyMouseMoveListener: MouseListener {

        TrackballControls& scope;

        explicit MyMouseMoveListener(TrackballControls& scope): scope(scope) {}

        void onMouseMove(const Vector2& pos) override {

            if (!scope.enabled) return;

            auto& p = *scope.pimpl_;

            switch (p.state) {
                case ROTATE:
                    p.movePrev.copy(p.moveCurr);
                    p.moveCurr.copy(p.getMouseOnCircle(pos));
                    break;
                case ZOOM:
                    p.zoomEnd.copy(p.getMouseOnScreen(pos));
                    break;
                case PAN:
                    p.panEnd.copy(p.getMouseOnScreen(pos));
                    break;
                default:
                    break;
            }
        }
    };

    struct MyMouseUpListener: MouseListener {

        TrackballControls& scope;
        MouseListener* mouseMoveListener;

        MyMouseUpListener(TrackballControls& scope, MouseListener* mouseMoveListener)
            : scope(scope), mouseMoveListener(mouseMoveListener) {}

        void onMouseUp(int, const Vector2&) override {

            scope.pimpl_->canvas.removeMouseListener(*mouseMoveListener);
            scope.pimpl_->canvas.removeMouseListener(*this);
            scope.pimpl_->state = NONE;
        }
    };

    struct MyMouseListener: MouseListener {

        TrackballControls& scope;
        MyMouseMoveListener mouseMoveListener;
        MyMouseUpListener mouseUpListener;

        explicit MyMouseListener(TrackballControls& scope)
            : scope(scope), mouseMoveListener(scope), mouseUpListener(scope, &mouseMoveListener) {}

        void onMouseDown(int button, const Vector2& pos) override {

            if (!scope.enabled || scope.pimpl_->state != NONE) return;

            auto& p = *scope.pimpl_;

            switch (button) {
                case 0:// LEFT
                    if (scope.enableRotate) {
                        p.moveCurr.copy(p.getMouseOnCircle(pos));
                        p.movePrev.copy(p.moveCurr);
                        p.state = ROTATE;
                    }
                    break;
                case 1:// RIGHT
                    if (scope.enablePan) {
                        p.panStart.copy(p.getMouseOnScreen(pos));
                        p.panEnd.copy(p.panStart);
                        p.state = PAN;
                    }
                    break;
                case 2:// MIDDLE
                    if (scope.enableZoom) {
                        p.zoomStart.copy(p.getMouseOnScreen(pos));
                        p.zoomEnd.copy(p.zoomStart);
                        p.state = ZOOM;
                    }
                    break;
                default:
                    break;
            }

            if (p.state != NONE) {

                p.canvas.addMouseListener(mouseMoveListener);
                p.canvas.addMouseListener(mouseUpListener);
            }
        }

        void onMouseWheel(const Vector2& delta) override {

            if (!scope.enabled || !scope.enableZoom) return;

            // scrolling up (positive) pulls zoomEnd below zoomStart, which shrinks eye.
            // The step is eased away over several frames, so one notch ends up around
            // 14% - the same as three.js gets from a 100-unit browser wheel delta.
            scope.pimpl_->zoomStart.y += delta.y * 0.025f;
        }
    };
};

TrackballControls::TrackballControls(Camera& camera, PeripheralsEventSource& eventSource)
    : pimpl_(std::make_unique<Impl>(*this, eventSource, camera)) {}

bool TrackballControls::update() {

    return pimpl_->update();
}

void TrackballControls::reset() {

    pimpl_->reset();
}

float TrackballControls::getDistance() const {

    return pimpl_->getDistance();
}

TrackballControls::~TrackballControls() = default;
