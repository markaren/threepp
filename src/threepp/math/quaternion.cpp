
#include "threepp/math/quaternion.hpp"

#include "threepp/math/matrix4.hpp"
#include "threepp/math/vector3.hpp"

#include <cmath>
#include <algorithm>

using namespace threepp;

quaternion::quaternion(float x, float y, float z, float w) : x_(x), y_(y), z_(z), w_(w) {}

quaternion &quaternion::set(float x, float y, float z, float w) {

    this->x_ = x;
    this->y_ = y;
    this->z_ = z;
    this->w_ = w;

    this->onChangeCallback_();

    return *this;
}

quaternion &quaternion::setFromAxisAngle(const vector3 &axis, float angle) {

    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm

    // assumes axis is normalized

    const auto halfAngle = angle / 2, s = std::sin(halfAngle);

    this->x_ = axis.x * s;
    this->y_ = axis.y * s;
    this->z_ = axis.z * s;
    this->w_ = std::cos(halfAngle);

    this->onChangeCallback_();

    return *this;
}

quaternion &quaternion::setFromRotationMatrix(const matrix4 &m) {

    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm

    // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)

    const auto te = m.elements_;

    const auto m11 = te[0], m12 = te[4], m13 = te[8],
               m21 = te[1], m22 = te[5], m23 = te[9],
               m31 = te[2], m32 = te[6], m33 = te[10],

               trace = m11 + m22 + m33;

    if (trace > 0) {

        const auto s = 0.5f / std::sqrt(trace + 1.0f);

        this->w_ = 0.25f / s;
        this->x_ = (m32 - m23) * s;
        this->y_ = (m13 - m31) * s;
        this->z_ = (m21 - m12) * s;

    } else if (m11 > m22 && m11 > m33) {

        const auto s = 2.0f * std::sqrt(1.0f + m11 - m22 - m33);

        this->w_ = (m32 - m23) / s;
        this->x_ = 0.25f * s;
        this->y_ = (m12 + m21) / s;
        this->z_ = (m13 + m31) / s;

    } else if (m22 > m33) {

        const auto s = 2.0f * std::sqrt(1.0f + m22 - m11 - m33);

        this->w_ = (m13 - m31) / s;
        this->x_ = (m12 + m21) / s;
        this->y_ = 0.25f * s;
        this->z_ = (m23 + m32) / s;

    } else {

        const auto s = 2.f * std::sqrt(1.0f + m33 - m11 - m22);

        this->w_ = (m21 - m12) / s;
        this->x_ = (m13 + m31) / s;
        this->y_ = (m23 + m32) / s;
        this->z_ = 0.25f * s;
    }

    this->onChangeCallback_();

    return *this;
}

quaternion &quaternion::identity() {

    return this->set(0, 0, 0, 1);
}

quaternion &quaternion::invert() {

    // quaternion is assumed to have unit length

    return this->conjugate();
}

quaternion &quaternion::conjugate() {

    this->x_ *= -1;
    this->y_ *= -1;
    this->z_ *= -1;

    this->onChangeCallback_();

    return *this;
}

float quaternion::dot(const quaternion &v) const {

    return this->x_ * v.x_ + this->y_ * v.y_ + this->z_ * v.z_ + this->w_ * v.w_;
}

float quaternion::lengthSq() const {

    return this->x_ * this->x_ + this->y_ * this->y_ + this->z_ * this->z_ + this->w_ * this->w_;
}

float quaternion::length() const {

    return std::sqrt(this->x_ * this->x_ + this->y_ * this->y_ + this->z_ * this->z_ + this->w_ * this->w_);
}

quaternion &quaternion::normalize() {

    auto l = length();

    if (l == 0) {

        this->x_ = 0;
        this->y_ = 0;
        this->z_ = 0;
        this->w_ = 1;

    } else {

        l = 1.0f / l;

        this->x_ = this->x_ * l;
        this->y_ = this->y_ * l;
        this->z_ = this->z_ * l;
        this->w_ = this->w_ * l;
    }

    this->onChangeCallback_();

    return *this;
}

float quaternion::angleTo(const quaternion &q) const{

    return 2 * std::acos( std::abs( std::clamp( this->dot( q ), - 1.0f, 1.0f ) ) );

}

quaternion &quaternion::_onChange(std::function<void()> callback) {

    this->onChangeCallback_ = std::move(callback);

    return *this;

}
