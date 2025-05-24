
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

void Camera::updateWorldMatrix(std::optional<bool> updateParents, std::optional<bool> updateChildren) {

    Object3D::updateWorldMatrix(updateParents, updateChildren);

    this->matrixWorldInverse.copy(*this->matrixWorld).invert();
}
