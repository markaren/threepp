
#include "threepp/math/Vector3.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Spherical.hpp"

#include "threepp/cameras/Camera.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace threepp;

namespace {
    thread_local Quaternion _quaternion;
}

Vector3& Vector3::applyAxisAngle(const Vector3& axis, float angle) {

    return this->applyQuaternion(_quaternion.setFromAxisAngle(axis, angle));
}

Vector3& Vector3::applyEuler(const Euler& euler) {

    return this->applyQuaternion(_quaternion.setFromEuler(euler));
}

Vector3& Vector3::applyQuaternion(const Quaternion& q) {

    const auto x = this->x, y = this->y, z = this->z;
    const auto qx = q.x, qy = q.y, qz = q.z, qw = q.w;

    // calculate quat * vector

    const auto ix = qw * x + qy * z - qz * y;
    const auto iy = qw * y + qz * x - qx * z;
    const auto iz = qw * z + qx * y - qy * x;
    const auto iw = -qx * x - qy * y - qz * z;

    // calculate result * inverse quat

    this->x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
    this->y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
    this->z = iz * qw + iw * -qz + ix * -qy - iy * -qx;

    return *this;
}

Vector3& Vector3::project(const Camera& camera) {

    return this->applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
}

Vector3& Vector3::unproject(const Camera& camera) {

    return this->applyMatrix4(camera.projectionMatrixInverse).applyMatrix4(*camera.matrixWorld);
}

float Vector3::angleTo(const Vector3& v) const {

    const auto denominator = std::sqrt(lengthSq() * v.lengthSq());

    if (denominator == 0) return math::PI / 2;

    const auto theta = dot(v) / denominator;

    // clamp, to handle numerical problems

    return std::acos(std::clamp(theta, -1.0f, 1.0f));
}

Vector3& Vector3::setFromSpherical(const Spherical& s) {

    return this->setFromSphericalCoords(s.radius, s.phi, s.theta);
}
