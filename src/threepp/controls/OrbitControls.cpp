
#include "threepp/controls/OrbitControls.hpp"

#include "threepp/utils/InstanceOf.hpp"

#include <cmath>
#include <utility>

using namespace threepp;

namespace {

    const float EPS = 0.000001f;

    const int RIGHT_KEY = 262;
    const int LEFT_KEY = 263;
    const int BOTTOM_KEY = 264;
    const int UP_KEY = 265;

    const int LEFT_BUTTON = 0;
    const int RIGHT_BUTTON = 1;
    const int MIDDLE_BUTTON = 2;

}// namespace

OrbitControls::OrbitControls(std::shared_ptr<Camera> camera, Canvas &canvas)
    : camera(std::move(camera)), canvas(canvas),
      keyListener(new MyKeyListener(*this)),
      mouseListener(new MyMouseListener(*this)) {

    canvas.addMouseListener(mouseListener);
    canvas.addKeyListener(keyListener);

    update();
}

bool OrbitControls::update() {

    Vector3 offset{};

    // so camera.up is the orbit axis
    const auto quat = Quaternion().setFromUnitVectors(camera->up, {0, 1, 0});
    auto quatInverse = Quaternion().copy(quat).invert();

    Vector3 lastPosition{};
    Quaternion lastQuaternion{};

    auto &position = this->camera->position;

    offset.copy(position).sub(this->target);

    // rotate offset to "y-axis-is-up" space
    offset.applyQuaternion(quat);

    // angle from z-axis around y-axis
    spherical.setFromVector3(offset);

    if (this->autoRotate && state == State::NONE) {

        rotateLeft(getAutoRotationAngle());
    }

    if (this->enableDamping) {

        spherical.theta += sphericalDelta.theta * this->dampingFactor;
        spherical.phi += sphericalDelta.phi * this->dampingFactor;

    } else {

        spherical.theta += sphericalDelta.theta;
        spherical.phi += sphericalDelta.phi;
    }

    // restrict theta to be between desired limits
    spherical.theta = std::max(this->minAzimuthAngle, std::min(this->maxAzimuthAngle, spherical.theta));

    // restrict phi to be between desired limits
    spherical.phi = std::max(this->minPolarAngle, std::min(this->maxPolarAngle, spherical.phi));

    spherical.makeSafe();

    spherical.radius *= scale;

    // restrict radius to be between desired limits
    spherical.radius = std::max(this->minDistance, std::min(this->maxDistance, spherical.radius));

    // move target to panned location

    if (this->enableDamping) {

        this->target.addScaledVector(panOffset, this->dampingFactor);

    } else {

        this->target.add(panOffset);
    }

    offset.setFromSpherical(spherical);

    // rotate offset back to "camera-up-vector-is-up" space
    offset.applyQuaternion(quatInverse);

    position.copy(this->target).add(offset);

    this->camera->lookAt(this->target);

    if (this->enableDamping) {

        sphericalDelta.theta *= (1 - this->dampingFactor);
        sphericalDelta.phi *= (1 - this->dampingFactor);

        panOffset.multiplyScalar(1 - this->dampingFactor);

    } else {

        sphericalDelta.set(0, 0, 0);

        panOffset.set(0, 0, 0);
    }

    scale = 1;

    // update condition is:
    // min(camera displacement, camera rotation in radians)^2 > EPS
    // using small-angle approximation cos(x/2) = 1 - x^2 / 8

    if (zoomChanged ||
        lastPosition.distanceToSquared(this->camera->position) > EPS ||
        8 * (1 - lastQuaternion.dot(this->camera->quaternion)) > EPS) {

        lastPosition.copy(this->camera->position);
        lastQuaternion.copy(this->camera->quaternion);
        zoomChanged = false;

        return true;
    }

    return false;
}

float OrbitControls::getAutoRotationAngle() const {

    return 2 * math::PI / 60 / 60 * this->autoRotateSpeed;
}

float OrbitControls::getZoomScale() const {

    return std::pow(0.95f, this->zoomSpeed);
}

void OrbitControls::rotateLeft(float angle) {

    sphericalDelta.theta -= angle;
}

void OrbitControls::rotateUp(float angle) {

    sphericalDelta.phi -= angle;
}

void OrbitControls::panLeft(float distance, const Matrix4 &objectMatrix) {

    Vector3 v{};

    v.setFromMatrixColumn(objectMatrix, 0);// get X column of objectMatrix
    v.multiplyScalar(-distance);

    panOffset.add(v);
}

void OrbitControls::panUp(float distance, const Matrix4 &objectMatrix) {

    Vector3 v{};

    if (this->screenSpacePanning) {

        v.setFromMatrixColumn(objectMatrix, 1);

    } else {

        v.setFromMatrixColumn(objectMatrix, 0);
        v.crossVectors(this->camera->up, v);
    }

    v.multiplyScalar(distance);

    panOffset.add(v);
}

void OrbitControls::pan(float deltaX, float deltaY) {

    Vector3 offset{};

    if (instanceof <PerspectiveCamera>(camera.get())) {

        auto perspective = dynamic_cast<PerspectiveCamera *>(camera.get());

        // perspective
        auto &position = this->camera->position;
        offset.copy(position).sub(this->target);
        auto targetDistance = offset.length();

        // half of the fov is center to top of screen
        targetDistance *= std::tan((perspective->fov / 2) * math::PI / 180.f);

        // we use only clientHeight here so aspect ratio does not distort speed
        const auto size = canvas.getSize();
        panLeft(2 * deltaX * targetDistance / (float) size.height, this->camera->matrix);
        panUp(2 * deltaY * targetDistance / (float) size.height, this->camera->matrix);
    } else if (instanceof <OrthographicCamera>(camera.get())) {

        const auto size = canvas.getSize();
        auto ortho = dynamic_cast<OrthographicCamera *>(camera.get());

        // orthographic
        panLeft(
                deltaX * (ortho->right - ortho->left) / this->camera->zoom / size.width,
                this->camera->matrix);
        panUp(
                deltaY * (ortho->top - ortho->bottom) / this->camera->zoom / size.height,
                this->camera->matrix);

    } else {

        // camera neither orthographic nor perspective
        std::cerr << "encountered an unknown camera type - pan disabled." << std::endl;
        this->enablePan = false;
    }
}

void OrbitControls::dollyIn(float dollyScale) {

    if (instanceof <PerspectiveCamera>(camera.get())) {

        scale /= dollyScale;

    } else if (instanceof <OrthographicCamera>(camera.get())) {

        this->camera->zoom = std::max(this->minZoom, std::min(this->maxZoom, this->camera->zoom * dollyScale));
        this->camera->updateProjectionMatrix();
        zoomChanged = true;
    } else {

        std::cerr << "encountered an unknown camera type - dolly/zoom disabled." << std::endl;
        this->enableZoom = false;
    }
}
void OrbitControls::dollyOut(float dollyScale) {

    if (instanceof <PerspectiveCamera>(camera.get())) {
        scale *= dollyScale;
    } else if (instanceof <OrthographicCamera>(camera.get())) {
        this->camera->zoom = std::max(this->minZoom, std::min(this->maxZoom, this->camera->zoom / dollyScale));
        this->camera->updateProjectionMatrix();
        zoomChanged = true;
    } else {
        std::cerr << "encountered an unknown camera type - dolly/zoom disabled." << std::endl;
        this->enableZoom = false;
    }
}

void OrbitControls::handleKeyDown(int key) {

    bool needsUpdate = true;

    switch (key) {

        case UP_KEY:
            pan(0, keyPanSpeed);
            break;
        case BOTTOM_KEY:
            pan(0, -keyPanSpeed);
            break;
        case LEFT_KEY:
            pan(keyPanSpeed, 0);
            break;
        case RIGHT_KEY:
            pan(-keyPanSpeed, 0);
            break;
        default:
            needsUpdate = false;
            break;
    }

    if (needsUpdate) {
        update();
    }
}

void OrbitControls::handleMouseDownRotate(const Vector2 &pos) {

    rotateStart.copy(pos);
}

void OrbitControls::handleMouseDownDolly(const Vector2 &pos) {

    dollyStart.copy(pos);
}

void OrbitControls::handleMouseDownPan(const Vector2 &pos) {

    panStart.copy(pos);
}

void OrbitControls::handleMouseMoveRotate(const Vector2 &pos) {

    rotateEnd.copy(pos);

    rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(rotateSpeed);

    const auto size = canvas.getSize();
    rotateLeft(2 * math::PI * rotateDelta.x / static_cast<float>(size.height));// yes, height

    rotateUp(2 * math::PI * rotateDelta.y / static_cast<float>(size.height));

    rotateStart.copy(rotateEnd);

    update();
}

void OrbitControls::handleMouseMoveDolly(const Vector2 &pos) {

    dollyEnd.copy(pos);

    dollyDelta.subVectors(dollyEnd, dollyStart);

    if (dollyDelta.y > 0) {

        dollyIn(getZoomScale());

    } else if (dollyDelta.y < 0) {

        dollyOut(getZoomScale());
    }

    dollyStart.copy(dollyEnd);

    update();
}


void OrbitControls::handleMouseMovePan(const Vector2 &pos) {

    panEnd.copy(pos);

    panDelta.subVectors(panEnd, panStart).multiplyScalar(panSpeed);

    pan(panDelta.x, panDelta.y);

    panStart.copy(panEnd);

    update();
}

void OrbitControls::handleMouseWheel(const Vector2 &delta) {
    if (delta.y < 0) {

        dollyOut(getZoomScale());

    } else if (delta.y > 0) {

        dollyIn(getZoomScale());
    }

    update();
}

OrbitControls::~OrbitControls() {

    canvas.removeMouseListener(mouseListener->uuid);
    canvas.removeKeyListener(keyListener->uuid);
}
