
#include "threepp/math/euler.hpp"

#include "threepp/math/vector3.hpp"
#include "threepp/math/matrix3.hpp"
#include "threepp/math/matrix4.hpp"
#include "threepp/math/quaternion.hpp"

#include <algorithm>

using namespace threepp;

matrix4 euler::_matrix = matrix4();
quaternion euler::_quaternion = quaternion();

euler &euler::set(double x, double y, double z, const std::optional<RotationOrders> &order) {

    this->x_ = x;
    this->y_ = y;
    this->z_ = z;
    this->order_ = order.value_or(this->order_);

    this->onChangeCallback_();

    return *this;
}

euler &euler::setFromRotationMatrix(const matrix4 &m, std::optional<RotationOrders> order, bool update) {

    // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)

    const auto te = m.elements_;
    const auto m11 = te[0], m12 = te[4], m13 = te[8];
    const auto m21 = te[1], m22 = te[5], m23 = te[9];
    const auto m31 = te[2], m32 = te[6], m33 = te[10];

    this->order_ = order.value_or(this->order_);

    switch (this->order_) {

        case XYZ:

            this->y_ = std::asin(std::clamp(m13, -1.0, 1.0));

            if (std::abs(m13) < 0.9999999) {

                this->x_ = std::atan2(-m23, m33);
                this->z_ = std::atan2(-m12, m11);

            } else {

                this->x_ = std::atan2(m32, m22);
                this->z_ = 0;
            }

            break;

        case YXZ:

            this->x_ = std::asin(-std::clamp(m23, -1.0, 1.0));

            if (std::abs(m23) < 0.9999999) {

                this->y_ = std::atan2(m13, m33);
                this->z_ = std::atan2(m21, m22);

            } else {

                this->y_ = std::atan2(-m31, m11);
                this->z_ = 0;
            }

            break;

        case ZXY:

            this->x_ = std::asin(std::clamp(m32, -1.0, 1.0));

            if (std::abs(m32) < 0.9999999) {

                this->y_ = std::atan2(-m31, m33);
                this->z_ = std::atan2(-m12, m22);

            } else {

                this->y_ = 0;
                this->z_ = std::atan2(m21, m11);
            }

            break;

        case ZYX:

            this->y_ = std::asin(-std::clamp(m31, -1.0, 1.0));

            if (std::abs(m31) < 0.9999999) {

                this->x_ = std::atan2(m32, m33);
                this->z_ = std::atan2(m21, m11);

            } else {

                this->x_ = 0;
                this->z_ = std::atan2(-m12, m22);
            }

            break;

        case YZX:

            this->z_ = std::asin(std::clamp(m21, -1.0, 1.0));

            if (std::abs(m21) < 0.9999999) {

                this->x_ = std::atan2(-m23, m22);
                this->y_ = std::atan2(-m31, m11);

            } else {

                this->x_ = 0;
                this->y_ = std::atan2(m13, m33);
            }

            break;

        case XZY:

            this->z_ = std::asin(-std::clamp(m12, -1.0, 1.0));

            if (std::abs(m12) < 0.9999999) {

                this->x_ = std::atan2(m32, m22);
                this->y_ = std::atan2(m13, m11);

            } else {

                this->x_ = std::atan2(-m23, m33);
                this->y_ = 0;
            }

            break;
    }

    if (update) this->onChangeCallback_();

    return *this;
}

euler &euler::setFromQuaternion(const quaternion &q, std::optional<RotationOrders> order, bool update) {

    _matrix.makeRotationFromQuaternion( q );

    return this->setFromRotationMatrix( _matrix, order, update );

}

euler &euler::setFromVector3(const vector3 &v, std::optional<RotationOrders> order) {

    return this->set(v.x, v.y, v.z, order);
}
