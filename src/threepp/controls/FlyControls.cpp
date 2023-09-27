
#include "threepp/controls/FlyControls.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Spherical.hpp"

using namespace threepp;

namespace {

    const float EPS = 0.000001f;

    struct MoveState {

        int up = 0;
        int down = 0;
        int left = 0;
        int right = 0;
        int forward = 0;
        int back = 0;
        float pitchUp = 0;
        float pitchDown = 0;
        float yawLeft = 0;
        float yawRight = 0;
        float rollLeft = 0;
        float rollRight = 0;
    };

}// namespace

struct FlyControls::Impl {

    Impl(FlyControls& scope, Canvas& canvas, Object3D* object)
        : canvas(canvas), scope(scope), object(object),
          keyUp(scope), keydown(scope),
          mouseDown(scope), mouseMove(scope), mouseUp(scope) {

        canvas.addKeyListener(&keydown);
        canvas.addKeyListener(&keyUp);

        canvas.addMouseListener(&mouseDown);
        canvas.addMouseListener(&mouseMove);
        canvas.addMouseListener(&mouseUp);
    }

    void update(float delta) {

        const auto moveMult = delta * scope.movementSpeed;
        const auto rotMult = delta * scope.rollSpeed;

        object->translateX(moveVector.x * moveMult);
        object->translateY(moveVector.y * moveMult);
        object->translateZ(moveVector.z * moveMult);

        tmpQuaternion.set(rotationVector.x * rotMult, rotationVector.y * rotMult, rotationVector.z * rotMult, 1).normalize();
        object->quaternion.multiply(tmpQuaternion);

        if (
                lastPosition.distanceToSquared(object->position) > EPS ||
                8 * (1 - lastQuaternion.dot(object->quaternion)) > EPS) {

            lastQuaternion.copy(object->quaternion);
            lastPosition.copy(object->position);
        }
    }

    void updateMovementVector() {

        int forward = (moveState.forward || (scope.autoForward && !moveState.back));

        moveVector.x = static_cast<float>(-moveState.left + moveState.right);
        moveVector.y = static_cast<float>(-moveState.down + moveState.up);
        moveVector.z = static_cast<float>(-forward + moveState.back);
    }

    void updateRotationVector() {

        rotationVector.x = (-moveState.pitchDown + moveState.pitchUp);
        rotationVector.y = (-moveState.yawRight + moveState.yawLeft);
        rotationVector.z = (-moveState.rollRight + moveState.rollLeft);
    }

    ~Impl() {

        canvas.removeKeyListener(&keydown);
        canvas.removeKeyListener(&keyUp);

        canvas.removeMouseListener(&mouseDown);
        canvas.removeMouseListener(&mouseMove);
        canvas.removeMouseListener(&mouseUp);
    }

    struct KeyDownListener: KeyListener {

        FlyControls& scope;

        explicit KeyDownListener(FlyControls& scope): scope(scope) {}

        void onKeyPressed(KeyEvent evt) override {

            if (evt.mods == 4) {// altKey

                return;
            }

            switch (evt.key) {
                case Key::W:
                    scope.pimpl_->moveState.forward = 1;
                    break;
                case Key::S:
                    scope.pimpl_->moveState.back = 1;
                    break;
                case Key::A:
                    scope.pimpl_->moveState.left = 1;
                    break;
                case Key::D:
                    scope.pimpl_->moveState.right = 1;
                    break;
                case Key::R:
                    scope.pimpl_->moveState.up = 1;
                    break;
                case Key::F:
                    scope.pimpl_->moveState.down = 1;
                    break;
                case Key::UP:
                    scope.pimpl_->moveState.pitchUp = 1;
                    break;
                case Key::DOWN:
                    scope.pimpl_->moveState.pitchDown = 1;
                    break;
                case Key::LEFT:
                    scope.pimpl_->moveState.yawLeft = 1;
                    break;
                case Key::RIGHT:
                    scope.pimpl_->moveState.yawRight = 1;
                    break;
                case Key::Q:
                    scope.pimpl_->moveState.rollLeft = 1;
                    break;
                case Key::E:
                    scope.pimpl_->moveState.rollRight = 1;
                    break;
                default:
                    break;
            }

            scope.pimpl_->updateMovementVector();
            scope.pimpl_->updateRotationVector();
        }
    };

    struct KeyUpListener: KeyListener {

        FlyControls& scope;

        explicit KeyUpListener(FlyControls& scope): scope(scope) {}

        void onKeyReleased(KeyEvent evt) override {

            switch (evt.key) {
                case Key::W:
                    scope.pimpl_->moveState.forward = 0;
                    break;
                case Key::S:
                    scope.pimpl_->moveState.back = 0;
                    break;
                case Key::A:
                    scope.pimpl_->moveState.left = 0;
                    break;
                case Key::D:
                    scope.pimpl_->moveState.right = 0;
                    break;
                case Key::R:
                    scope.pimpl_->moveState.up = 0;
                    break;
                case Key::F:
                    scope.pimpl_->moveState.down = 0;
                    break;
                case Key::UP:
                    scope.pimpl_->moveState.pitchUp = 0;
                    break;
                case Key::DOWN:
                    scope.pimpl_->moveState.pitchDown = 0;
                    break;
                case Key::LEFT:
                    scope.pimpl_->moveState.yawLeft = 0;
                    break;
                case Key::RIGHT:
                    scope.pimpl_->moveState.yawRight = 0;
                    break;
                case Key::Q:
                    scope.pimpl_->moveState.rollLeft = 0;
                    break;
                case Key::E:
                    scope.pimpl_->moveState.rollRight = 0;
                    break;
                default:
                    break;
            }

            scope.pimpl_->updateMovementVector();
            scope.pimpl_->updateRotationVector();
        }
    };

    struct MouseDownListener: MouseListener {

        FlyControls& scope;

        explicit MouseDownListener(FlyControls& scope): scope(scope) {}

        void onMouseDown(int button, const Vector2& pos) override {

            if (scope.dragToLook) {

                scope.pimpl_->mouseStatus++;

            } else {

                switch (button) {

                    case 0:
                        scope.pimpl_->moveState.forward = 1;
                        break;
                    case 1:
                        scope.pimpl_->moveState.back = 1;
                        break;
                }

                scope.pimpl_->updateMovementVector();
            }
        }
    };

    struct MouseMoveListener: MouseListener {

        FlyControls& scope;

        explicit MouseMoveListener(FlyControls& scope): scope(scope) {}

        void onMouseMove(const Vector2& pos) override {

            if (!scope.dragToLook || scope.pimpl_->mouseStatus > 0) {

                const float halfWidth = static_cast<float>(scope.pimpl_->canvas.size().width) / 2;
                const float halfHeight = static_cast<float>(scope.pimpl_->canvas.size().height) / 2;

                scope.pimpl_->moveState.yawLeft = -((pos.x) - halfWidth) / halfWidth;
                scope.pimpl_->moveState.pitchDown = ((pos.y) - halfHeight) / halfHeight;

                scope.pimpl_->updateRotationVector();
            }
        }
    };

    struct MouseUpListener: MouseListener {

        FlyControls& scope;

        explicit MouseUpListener(FlyControls& scope): scope(scope) {}

        void onMouseUp(int button, const Vector2& pos) override {

            if (scope.dragToLook) {

                scope.pimpl_->mouseStatus--;

                scope.pimpl_->moveState.yawLeft = scope.pimpl_->moveState.pitchDown = 0;

            } else {

                switch (button) {

                    case 0:
                        scope.pimpl_->moveState.forward = 0;
                        break;
                    case 1:
                        scope.pimpl_->moveState.back = 0;
                        break;
                }

                scope.pimpl_->updateMovementVector();
            }

            scope.pimpl_->updateRotationVector();
        }
    };

private:
    Canvas& canvas;
    FlyControls& scope;
    Object3D* object;

    Quaternion lastQuaternion;
    Vector3 lastPosition;

    Quaternion tmpQuaternion;

    int mouseStatus = 0;

    MoveState moveState;
    Vector3 moveVector;
    Vector3 rotationVector;

    KeyUpListener keydown;
    KeyDownListener keyUp;

    MouseDownListener mouseDown;
    MouseMoveListener mouseMove;
    MouseUpListener mouseUp;
};

FlyControls::FlyControls(Object3D& object, Canvas& canvas)
    : pimpl_(std::make_unique<Impl>(*this, canvas, &object)) {}

void threepp::FlyControls::update(float delta) {

    pimpl_->update(delta);
}

threepp::FlyControls::~FlyControls() = default;
