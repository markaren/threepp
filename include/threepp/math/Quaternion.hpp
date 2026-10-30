// https://github.com/mrdoob/three.js/blob/r129/src/math/Quaternion.js

#ifndef THREEPP_QUATERNION_HPP
#define THREEPP_QUATERNION_HPP

#include "threepp/math/float_view.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace threepp {

    class Vector3;
    class Matrix4;
    class Euler;

    class Quaternion {

    public:
        float_view x;
        float_view y;
        float_view z;
        float_view w;

        explicit Quaternion(float x = 0, float y = 0, float z = 0, float w = 1)
            : x(x), y(y), z(z), w(w) {}

        float operator[](unsigned int index) const {
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

        Quaternion& set(float x, float y, float z, float w) {
            this->x.value_ = x;
            this->y.value_ = y;
            this->z.value_ = z;
            this->w.value_ = w;

            this->onChangeCallback_();

            return *this;
        }

        Quaternion& copy(const Quaternion& quaternion) {
            this->x.value_ = quaternion.x;
            this->y.value_ = quaternion.y;
            this->z.value_ = quaternion.z;
            this->w.value_ = quaternion.w;

            this->onChangeCallback_();

            return *this;
        }

        Quaternion& setFromEuler(const Euler& euler, bool update = true);

        Quaternion& setFromAxisAngle(const Vector3& axis, float angle);

        Quaternion& setFromRotationMatrix(const Matrix4& m);

        Quaternion& setFromUnitVectors(const Vector3& vFrom, const Vector3& vTo);

        [[nodiscard]] float angleTo(const Quaternion& q) const {
            return 2 * std::acos(std::abs(std::clamp(this->dot(q), -1.0f, 1.0f)));
        }

        Quaternion& rotateTowards(const Quaternion& q, float step);

        Quaternion& identity() {
            return this->set(0, 0, 0, 1);
        }

        Quaternion& invert() {
            // Quaternion is assumed to have unit length

            return this->conjugate();
        }

        Quaternion& conjugate() {
            this->x.value_ *= -1;
            this->y.value_ *= -1;
            this->z.value_ *= -1;

            this->onChangeCallback_();

            return *this;
        }

        [[nodiscard]] float dot(const Quaternion& v) const {
            return this->x * v.x + this->y * v.y + this->z * v.z + this->w * v.w;
        }

        [[nodiscard]] float lengthSq() const {
            return this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w;
        }

        [[nodiscard]] float length() const {
            return std::sqrt(this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w);
        }

        Quaternion& normalize() {
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

        Quaternion& multiply(const Quaternion& q) {
            return this->multiplyQuaternions(*this, q);
        }

        Quaternion& premultiply(const Quaternion& q) {
            return this->multiplyQuaternions(q, *this);
        }

        Quaternion& multiplyQuaternions(const Quaternion& a, const Quaternion& b) {
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

        Quaternion& slerp(const Quaternion& qb, float t);

        void slerpQuaternions(const Quaternion& qa, const Quaternion& qb, float t);

        [[nodiscard]] Quaternion clone() const {
            return Quaternion(x.value_, y.value_, z.value_, w.value_);
        }

        [[nodiscard]] bool equals(const Quaternion& v) const {
            return ((v.x == this->x) && (v.y == this->y) && (v.z == this->z) && (v.w == this->w));
        }

        bool operator==(const Quaternion& other) const {
            return equals(other);
        }

        bool operator!=(const Quaternion& other) const {
            return !equals(other);
        }

        Quaternion& _onChange(std::function<void()> callback) {
            this->onChangeCallback_ = std::move(callback);
            this->x.setCallback(this->onChangeCallback_);
            this->y.setCallback(this->onChangeCallback_);
            this->z.setCallback(this->onChangeCallback_);
            this->w.setCallback(this->onChangeCallback_);

            return *this;
        }

        template<class ArrayLike>
        Quaternion& fromArray(const ArrayLike& array, unsigned int offset = 0) {

            this->x.value_ = array[offset];
            this->y.value_ = array[offset + 1];
            this->z.value_ = array[offset + 2];
            this->w.value_ = array[offset + 3];

            this->onChangeCallback_();

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike& array, unsigned int offset = 0) const {

            array[offset] = this->x;
            array[offset + 1] = this->y;
            array[offset + 2] = this->z;
            array[offset + 3] = this->w;
        }

        friend std::ostream& operator<<(std::ostream& os, const Quaternion& v) {
            os << "Quaternion(x=" << v.x << ", y=" << v.y << ", z=" << v.z << ", w=" << v.w << ")";
            return os;
        }

        static void slerpFlat(std::vector<float>& dst, size_t dstOffset, const std::vector<float>& src0, size_t srcOffset0, const std::vector<float>& src1, size_t srcOffset1, float t);
        static void multiplyQuaternionsFlat(std::vector<float>& dst, size_t dstOffset, const std::vector<float>& src0, size_t srcOffset0, const std::vector<float>& src1, size_t srcOffset1) {
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

    private:
        std::function<void()> onChangeCallback_ = [] {};
    };

}// namespace threepp

#endif//THREEPP_QUATERNION_HPP
