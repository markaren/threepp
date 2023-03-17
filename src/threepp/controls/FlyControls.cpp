
#include "threepp/controls/FlyControls.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/math/Spherical.hpp"
#include "threepp/math/Vector2.hpp"

#include <cmath>

using namespace threepp;

namespace {

    const float EPS = 0.000001f;

    Quaternion tmpQuaternion;

    enum class STATE {
        NONE,
        ROTATE,
        DOLLY,
        PAN,
        TOUCH_ROTATE,
        TOUCH_PAN,
        TOUCH_DOLLY_PAN,
        TOUCH_DOLLY_ROTATE
    };

    struct MoveState {

        bool up = false;
        bool down = false;
        bool left = false;
        bool right = false;
        bool forward = false;
        bool back = false;
        bool pitchUp = false;
        bool pitchDown = false;
        bool yawLeft = false;
        bool yawRight = false;
        bool rollLeft = false;
        bool rollRight = false;

    };

}// namespace

struct FlyControls::Impl {

    Canvas& canvas;
    FlyControls& scope;
    Camera* object;

    STATE state = STATE::NONE;

    // current position in spherical coordinates
    Spherical spherical;
    Spherical sphericalDelta;

    float scale = 1;
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

    bool mouseStatus = false;

    MoveState moveState;

    Vector3 moveVector;
    Vector3 rotationVector;

    Impl(FlyControls& scope, Canvas& canvas): canvas(canvas), scope(scope) {}

    bool update() {

        Vector3 offset;

        // so camera.up is the orbit axis
        Quaternion quat = Quaternion().setFromUnitVectors(object->up, Vector3(0, 1, 0));
        Quaternion quatInverse = quat.clone().invert();

        Vector3 lastPosition;
        Quaternion lastQuaternion;


        auto& position = object->position;

        offset.copy(position).sub(scope.target);

        // rotate offset to "y-axis-is-up" space
        offset.applyQuaternion(quat);

        // angle from z-axis around y-axis
        spherical.setFromVector3(offset);

        if (scope.autoRotate && state == STATE::NONE) {

            rotateLeft(getAutoRotationAngle());
        }

        if (scope.enableDamping) {

            spherical.theta += sphericalDelta.theta * scope.dampingFactor;
            spherical.phi += sphericalDelta.phi * scope.dampingFactor;

        } else {

            spherical.theta += sphericalDelta.theta;
            spherical.phi += sphericalDelta.phi;
        }

        // restrict theta to be between desired limits

        float min = scope.minAzimuthAngle;
        float max = scope.maxAzimuthAngle;

        if (std::isfinite(min) && std::isfinite(max)) {

            if (min < -math::PI) min += math::TWO_PI;
            else if (min > math::PI)
                min -= math::TWO_PI;

            if (max < -math::PI) max += math::TWO_PI;
            else if (max > math::PI)
                max -= math::TWO_PI;

            if (min <= max) {

                spherical.theta = std::max(min, std::min(max, spherical.theta));

            } else {

                spherical.theta = (spherical.theta > (min + max) / 2) ? std::max(min, spherical.theta) : std::min(max, spherical.theta);
            }
        }

        // restrict phi to be between desired limits
        spherical.phi = std::max(scope.minPolarAngle, std::min(scope.maxPolarAngle, spherical.phi));

        spherical.makeSafe();


        spherical.radius *= scale;

        // restrict radius to be between desired limits
        spherical.radius = std::max(scope.minDistance, std::min(scope.maxDistance, spherical.radius));

        // move target to panned location

        if (scope.enableDamping == true) {

            scope.target.addScaledVector(panOffset, scope.dampingFactor);

        } else {

            scope.target.add(panOffset);
        }

        offset.setFromSpherical(spherical);

        // rotate offset back to "camera-up-vector-is-up" space
        offset.applyQuaternion(quatInverse);

        position.copy(scope.target).add(offset);

        object->lookAt(scope.target);

        if (scope.enableDamping == true) {

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
            lastPosition.distanceToSquared(object->position) > EPS ||
            8 * (1 - lastQuaternion.dot(object->quaternion)) > EPS) {

//            scope.dispatchEvent(_changeEvent);

            lastPosition.copy(object->position);
            lastQuaternion.copy(object->quaternion);
            zoomChanged = false;

            return true;
        }

        return false;
    }

    void updateMovementVector() {

        bool forward = (moveState.forward || (scope.autoForward && !moveState.back));

        moveVector.x = static_cast<float>( - moveState.left + moveState.right );
        moveVector.y = static_cast<float>( - moveState.down + moveState.up );
        moveVector.z = static_cast<float>( - forward + moveState.back );

        //console.log( 'move:', [ this.moveVector.x, this.moveVector.y, this.moveVector.z ] );

    }

    [[nodiscard]] float getAutoRotationAngle() const {

        return 2 * math::PI / 60 / 60 * scope.autoRotateSpeed;
    }

    [[nodiscard]] float getZoomScale() const {

        return std::pow(0.95f, scope.zoomSpeed);
    }

    void rotateLeft(float angle) {

        sphericalDelta.theta -= angle;
    }

    void rotateUp(float angle) {

        sphericalDelta.phi -= angle;
    }
};

threepp::FlyControls::FlyControls(Camera* camera, Canvas& canvas)
    : pimpl_(std::make_unique<Impl>(*this, canvas)) {}

void threepp::FlyControls::update(float delta) {
}

float threepp::FlyControls::getPolarAngle() {

    return pimpl_->spherical.phi;
}

float threepp::FlyControls::getAzimuthalAngle() {

    return pimpl_->spherical.theta;
}

threepp::FlyControls::~FlyControls() = default;
