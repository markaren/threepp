
#include "threepp/math/Quaternion.hpp"

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <string>

using namespace threepp;


Quaternion::Quaternion(float x, float y, float z, float w)
    : x(x), y(y), z(z), w(w) {}

float Quaternion::operator[](unsigned int index) const {
    switch (index) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
        default:
            throw std::runtime_error("index out of bound: " + std::to_string(index));
    }
}

Quaternion& Quaternion::set(float x, float y, float z, float w) {

    this->x.value_ = x;
    this->y.value_ = y;
    this->z.value_ = z;
    this->w.value_ = w;

    this->onChangeCallback_();

    return *this;
}

Quaternion& Quaternion::copy(const Quaternion& quaternion) {

    this->x.value_ = quaternion.x;
    this->y.value_ = quaternion.y;
    this->z.value_ = quaternion.z;
    this->w.value_ = quaternion.w;

    this->onChangeCallback_();

    return *this;
}

Quaternion& Quaternion::setFromEuler(const Euler& euler, bool update) {

    const auto x = euler.x, y = euler.y, z = euler.z;
    const auto order = euler.order_;

    // http://www.mathworks.com/matlabcentral/fileexchange/
    // 	20696-function-to-convert-between-dcm-euler-angles-quaternions-and-euler-vectors/
    //	content/SpinCalc.m

    const float c1 = std::cos(x / 2.f);
    const float c2 = std::cos(y / 2.f);
    const float c3 = std::cos(z / 2.f);

    const float s1 = std::sin(x / 2.f);
    const float s2 = std::sin(y / 2.f);
    const float s3 = std::sin(z / 2.f);

    switch (order) {

        case Euler::RotationOrders::XYZ:
            this->x.value_ = s1 * c2 * c3 + c1 * s2 * s3;
            this->y.value_ = c1 * s2 * c3 - s1 * c2 * s3;
            this->z.value_ = c1 * c2 * s3 + s1 * s2 * c3;
            this->w.value_ = c1 * c2 * c3 - s1 * s2 * s3;
            break;

        case Euler::RotationOrders::YXZ:
            this->x.value_ = s1 * c2 * c3 + c1 * s2 * s3;
            this->y.value_ = c1 * s2 * c3 - s1 * c2 * s3;
            this->z.value_ = c1 * c2 * s3 - s1 * s2 * c3;
            this->w.value_ = c1 * c2 * c3 + s1 * s2 * s3;
            break;

        case Euler::RotationOrders::ZXY:
            this->x.value_ = s1 * c2 * c3 - c1 * s2 * s3;
            this->y.value_ = c1 * s2 * c3 + s1 * c2 * s3;
            this->z.value_ = c1 * c2 * s3 + s1 * s2 * c3;
            this->w.value_ = c1 * c2 * c3 - s1 * s2 * s3;
            break;

        case Euler::RotationOrders::ZYX:
            this->x.value_ = s1 * c2 * c3 - c1 * s2 * s3;
            this->y.value_ = c1 * s2 * c3 + s1 * c2 * s3;
            this->z.value_ = c1 * c2 * s3 - s1 * s2 * c3;
            this->w.value_ = c1 * c2 * c3 + s1 * s2 * s3;
            break;

        case Euler::RotationOrders::YZX:
            this->x.value_ = s1 * c2 * c3 + c1 * s2 * s3;
            this->y.value_ = c1 * s2 * c3 + s1 * c2 * s3;
            this->z.value_ = c1 * c2 * s3 - s1 * s2 * c3;
            this->w.value_ = c1 * c2 * c3 - s1 * s2 * s3;
            break;

        case Euler::RotationOrders::XZY:
            this->x.value_ = s1 * c2 * c3 - c1 * s2 * s3;
            this->y.value_ = c1 * s2 * c3 - s1 * c2 * s3;
            this->z.value_ = c1 * c2 * s3 + s1 * s2 * c3;
            this->w.value_ = c1 * c2 * c3 + s1 * s2 * s3;
            break;
    }

    if (update) {
        this->onChangeCallback_();
    }

    return *this;
}

Quaternion& Quaternion::setFromAxisAngle(const Vector3& axis, float angle) {

    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm

    // assumes axis is normalized

    const float halfAngle = angle / 2.f, s = std::sin(halfAngle);

    this->x.value_ = axis.x * s;
    this->y.value_ = axis.y * s;
    this->z.value_ = axis.z * s;
    this->w.value_ = std::cos(halfAngle);

    this->onChangeCallback_();

    return *this;
}

Quaternion& Quaternion::setFromRotationMatrix(const Matrix4& m) {

    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm

    // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)

    const auto& te = m.elements;

    const auto m11 = te[0], m12 = te[4], m13 = te[8],
               m21 = te[1], m22 = te[5], m23 = te[9],
               m31 = te[2], m32 = te[6], m33 = te[10],

               trace = m11 + m22 + m33;

    if (trace > 0) {

        const auto s = 0.5f / std::sqrt(trace + 1.0f);

        this->w.value_ = 0.25f / s;
        this->x.value_ = (m32 - m23) * s;
        this->y.value_ = (m13 - m31) * s;
        this->z.value_ = (m21 - m12) * s;

    } else if (m11 > m22 && m11 > m33) {

        const auto s = 2.0f * std::sqrt(1.0f + m11 - m22 - m33);

        this->w.value_ = (m32 - m23) / s;
        this->x.value_ = 0.25f * s;
        this->y.value_ = (m12 + m21) / s;
        this->z.value_ = (m13 + m31) / s;

    } else if (m22 > m33) {

        const auto s = 2.0f * std::sqrt(1.0f + m22 - m11 - m33);

        this->w.value_ = (m13 - m31) / s;
        this->x.value_ = (m12 + m21) / s;
        this->y.value_ = 0.25f * s;
        this->z.value_ = (m23 + m32) / s;

    } else {

        const auto s = 2.f * std::sqrt(1.0f + m33 - m11 - m22);

        this->w.value_ = (m21 - m12) / s;
        this->x.value_ = (m13 + m31) / s;
        this->y.value_ = (m23 + m32) / s;
        this->z.value_ = 0.25f * s;
    }

    this->onChangeCallback_();

    return *this;
}

Quaternion& Quaternion::setFromUnitVectors(const Vector3& vFrom, const Vector3& vTo) {
    // assumes direction vectors vFrom and vTo are normalized

    const auto EPS = 0.000001f;

    auto r = vFrom.dot(vTo) + 1;

    if (r < EPS) {

        // vFrom and vTo point in opposite directions

        r = 0;

        if (std::abs(vFrom.x) > std::abs(vFrom.z)) {

            this->x.value_ = -vFrom.y;
            this->y.value_ = vFrom.x;
            this->z.value_ = 0;
            this->w.value_ = r;

        } else {

            this->x.value_ = 0;
            this->y.value_ = -vFrom.z;
            this->z.value_ = vFrom.y;
            this->w.value_ = r;
        }

    } else {

        // crossVectors( vFrom, vTo ); // inlined to avoid cyclic dependency on Vector3

        this->x.value_ = vFrom.y * vTo.z - vFrom.z * vTo.y;
        this->y.value_ = vFrom.z * vTo.x - vFrom.x * vTo.z;
        this->z.value_ = vFrom.x * vTo.y - vFrom.y * vTo.x;
        this->w.value_ = r;
    }

    return this->normalize();
}


float Quaternion::angleTo(const Quaternion& q) const {

    return 2 * std::acos(std::abs(std::clamp(this->dot(q), -1.0f, 1.0f)));
}

Quaternion& Quaternion::rotateTowards(const Quaternion& q, float step) {

    const float angle = this->angleTo(q);

    if (angle == 0) return *this;

    auto t = std::min(1.f, step / angle);

    this->slerp(q, t);

    return *this;
}

Quaternion& Quaternion::slerp(const Quaternion& qb, float t) {

    if (t == 0) return *this;
    if (t == 1) return this->copy(qb);

    const float x = this->x.value_, y = this->y.value_, z = this->z.value_, w = this->w.value_;

    // http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/

    float cosHalfTheta = w * qb.w.value_ + x * qb.x.value_ + y * qb.y.value_ + z * qb.z.value_;

    if (cosHalfTheta < 0) {

        this->w.value_ = -qb.w.value_;
        this->x.value_ = -qb.x.value_;
        this->y.value_ = -qb.y.value_;
        this->z.value_ = -qb.z.value_;

        cosHalfTheta = -cosHalfTheta;

    } else {

        this->copy(qb);
    }

    if (cosHalfTheta >= 1.0) {

        this->w.value_ = w;
        this->x.value_ = x;
        this->y.value_ = y;
        this->z.value_ = z;

        return *this;
    }

    const float sqrSinHalfTheta = 1.f - cosHalfTheta * cosHalfTheta;

    if (sqrSinHalfTheta <= std::numeric_limits<float>::epsilon()) {

        const float s = 1 - t;
        this->w.value_ = s * w + t * this->w.value_;
        this->x.value_ = s * x + t * this->x.value_;
        this->y.value_ = s * y + t * this->y.value_;
        this->z.value_ = s * z + t * this->z.value_;

        this->normalize();
        this->onChangeCallback_();

        return *this;
    }

    const float sinHalfTheta = std::sqrt(sqrSinHalfTheta);
    const float halfTheta = std::atan2(sinHalfTheta, cosHalfTheta);
    const float ratioA = std::sin((1 - t) * halfTheta) / sinHalfTheta,
                ratioB = std::sin(t * halfTheta) / sinHalfTheta;

    this->w = (w * ratioA + this->w.value_ * ratioB);
    this->x = (x * ratioA + this->x.value_ * ratioB);
    this->y = (y * ratioA + this->y.value_ * ratioB);
    this->z = (z * ratioA + this->z.value_ * ratioB);

    this->onChangeCallback_();

    return *this;
}

void Quaternion::slerpQuaternions(const Quaternion& qa, const Quaternion& qb, float t) {

    copy(qa).slerp(qb, t);
}

Quaternion& Quaternion::identity() {

    return this->set(0, 0, 0, 1);
}

Quaternion& Quaternion::invert() {

    // Quaternion is assumed to have unit length

    return this->conjugate();
}

Quaternion& Quaternion::conjugate() {

    this->x.value_ *= -1;
    this->y.value_ *= -1;
    this->z.value_ *= -1;

    this->onChangeCallback_();

    return *this;
}

float Quaternion::dot(const Quaternion& v) const {

    return this->x * v.x + this->y * v.y + this->z * v.z + this->w * v.w;
}

float Quaternion::lengthSq() const {

    return this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w;
}

float Quaternion::length() const {

    return std::sqrt(this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w);
}

Quaternion& Quaternion::normalize() {

    auto l = length();

    if (l == 0) {

        this->x.value_ = 0;
        this->y.value_ = 0;
        this->z.value_ = 0;
        this->w.value_ = 1;

    } else {

        l = 1.0f / l;

        this->x.value_ = this->x * l;
        this->y.value_ = this->y * l;
        this->z.value_ = this->z * l;
        this->w.value_ = this->w * l;
    }

    this->onChangeCallback_();

    return *this;
}

Quaternion& Quaternion::multiply(const Quaternion& q) {

    return this->multiplyQuaternions(*this, q);
}

Quaternion& Quaternion::premultiply(const Quaternion& q) {

    return this->multiplyQuaternions(q, *this);
}

Quaternion& Quaternion::multiplyQuaternions(const Quaternion& a, const Quaternion& b) {

    // from http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/code/index.htm

    const auto qax = a.x, qay = a.y, qaz = a.z, qaw = a.w;
    const auto qbx = b.x, qby = b.y, qbz = b.z, qbw = b.w;

    this->x.value_ = qax * qbw + qaw * qbx + qay * qbz - qaz * qby;
    this->y.value_ = qay * qbw + qaw * qby + qaz * qbx - qax * qbz;
    this->z.value_ = qaz * qbw + qaw * qbz + qax * qby - qay * qbx;
    this->w.value_ = qaw * qbw - qax * qbx - qay * qby - qaz * qbz;

    this->onChangeCallback_();

    return *this;
}

Quaternion Quaternion::clone() const {

    return Quaternion(x.value_, y.value_, z.value_, w.value_);
}

bool Quaternion::equals(const Quaternion& v) const {

    return ((v.x == this->x) && (v.y == this->y) && (v.z == this->z) && (v.w == this->w));
}

Quaternion& Quaternion::_onChange(std::function<void()> callback) {

    this->onChangeCallback_ = std::move(callback);
    this->x.setCallback(this->onChangeCallback_);
    this->y.setCallback(this->onChangeCallback_);
    this->z.setCallback(this->onChangeCallback_);
    this->w.setCallback(this->onChangeCallback_);

    return *this;
}

bool Quaternion::operator==(const Quaternion& other) const {

    return equals(other);
}

bool Quaternion::operator!=(const Quaternion& other) const {

    return !equals(other);
}

void Quaternion::slerpFlat(std::vector<float>& dst, size_t dstOffset, const std::vector<float>& src0, size_t srcOffset0, const std::vector<float>& src1, size_t srcOffset1, float t) {

    // fuzz-free, array-based Quaternion SLERP operation

    auto x0 = src0[srcOffset0 + 0],
         y0 = src0[srcOffset0 + 1],
         z0 = src0[srcOffset0 + 2],
         w0 = src0[srcOffset0 + 3];

    const auto x1 = src1[srcOffset1 + 0],
               y1 = src1[srcOffset1 + 1],
               z1 = src1[srcOffset1 + 2],
               w1 = src1[srcOffset1 + 3];

    if (t == 0) {

        dst[dstOffset + 0] = x0;
        dst[dstOffset + 1] = y0;
        dst[dstOffset + 2] = z0;
        dst[dstOffset + 3] = w0;
        return;
    }

    if (t == 1) {

        dst[dstOffset + 0] = x1;
        dst[dstOffset + 1] = y1;
        dst[dstOffset + 2] = z1;
        dst[dstOffset + 3] = w1;
        return;
    }

    if (w0 != w1 || x0 != x1 || y0 != y1 || z0 != z1) {

        auto s = 1 - t;
        const auto cos = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1;
        const auto dir = (cos >= 0 ? 1.f : -1.f);
        const auto sqrSin = 1 - cos * cos;

        // Skip the Slerp for tiny steps to avoid numeric problems:
        if (sqrSin > std::numeric_limits<float>::epsilon()) {

            const auto sin = std::sqrt(sqrSin),
                       len = std::atan2(sin, cos * dir);

            s = std::sin(s * len) / sin;
            t = std::sin(t * len) / sin;
        }

        const auto tDir = t * dir;

        x0 = x0 * s + x1 * tDir;
        y0 = y0 * s + y1 * tDir;
        z0 = z0 * s + z1 * tDir;
        w0 = w0 * s + w1 * tDir;

        // Normalize in case we just did a lerp:
        if (s == 1 - t) {

            const auto f = 1.f / std::sqrt(x0 * x0 + y0 * y0 + z0 * z0 + w0 * w0);

            x0 *= f;
            y0 *= f;
            z0 *= f;
            w0 *= f;
        }
    }

    dst[dstOffset] = x0;
    dst[dstOffset + 1] = y0;
    dst[dstOffset + 2] = z0;
    dst[dstOffset + 3] = w0;
}

void Quaternion::multiplyQuaternionsFlat(std::vector<float>& dst, size_t dstOffset, const std::vector<float>& src0, size_t srcOffset0, const std::vector<float>& src1, size_t srcOffset1) {
    const auto x0 = src0[srcOffset0];
    const auto y0 = src0[srcOffset0 + 1];
    const auto z0 = src0[srcOffset0 + 2];
    const auto w0 = src0[srcOffset0 + 3];

    const auto x1 = src1[srcOffset1];
    const auto y1 = src1[srcOffset1 + 1];
    const auto z1 = src1[srcOffset1 + 2];
    const auto w1 = src1[srcOffset1 + 3];

    dst[dstOffset] = x0 * w1 + w0 * x1 + y0 * z1 - z0 * y1;
    dst[dstOffset + 1] = y0 * w1 + w0 * y1 + z0 * x1 - x0 * z1;
    dst[dstOffset + 2] = z0 * w1 + w0 * z1 + x0 * y1 - y0 * x1;
    dst[dstOffset + 3] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;
}
