
#include "threepp/cameras/Camera.hpp"

using namespace threepp;

Camera::Camera(float _near, float _far)
    : nearPlane(_near), farPlane(_far) {}

void Camera::getWorldDirection(Vector3& target) {

    Object3D::getWorldDirection(target);
    target.negate();
}

void Camera::updateMatrixWorld(bool force) {

    Object3D::updateMatrixWorld(force);

    this->matrixWorldInverse.copy(*this->matrixWorld).invert();
}

void Camera::updateWorldMatrix(bool updateParents, bool updateChildren) {

    Object3D::updateWorldMatrix(updateParents, updateChildren);

    this->matrixWorldInverse.copy(*this->matrixWorld).invert();
}

void Camera::copy(const Object3D& source, bool recursive) {
    Object3D::copy(source, recursive);

    if (const auto c = source.as<Camera>()) {

        zoom = c->zoom;
        nearPlane = c->nearPlane;
        farPlane = c->farPlane;
        view = c->view;

        matrixWorldInverse.copy(c->matrixWorldInverse);
        projectionMatrix.copy(c->projectionMatrix);
        projectionMatrixInverse.copy(c->projectionMatrixInverse);
    }
}

std::shared_ptr<Object3D> Camera::createDefault() {

    return std::make_shared<Camera>();
}
