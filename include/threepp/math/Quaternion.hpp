// https://github.com/mrdoob/three.js/blob/r129/src/math/Quaternion.js

#ifndef THREEPP_QUATERNION_HPP
#define THREEPP_QUATERNION_HPP

#include <functional>
#include <iostream>
#include <string>

namespace threepp {

    class Vector3;
    class Matrix4;

    class Euler;

    class Quaternion {

    public:
        Quaternion() = default;

        Quaternion(float x, float y, float z, float w);

        float &operator[](unsigned int index);

        [[nodiscard]] float x() const {
            return x_;
        }

        Quaternion &x(float value) {

            this->x_ = value;
            this->onChangeCallback_();

            return *this;
        }

        [[nodiscard]] float y() const {
            return y_;
        }

        Quaternion &y(float value) {

            this->y_ = value;
            this->onChangeCallback_();

            return *this;
        }

        [[nodiscard]] float z() const {
            return z_;
        }

        Quaternion &z(float value) {

            this->z_ = value;
            this->onChangeCallback_();

            return *this;
        }

        [[nodiscard]] float w() const {
            return w_;
        }

        Quaternion &w(float value) {

            this->w_ = value;
            this->onChangeCallback_();

            return *this;
        }

        Quaternion &set(float x, float y, float z, float w);

        Quaternion &copy(const Quaternion &quaternion);

        Quaternion &setFromEuler(const Euler &euler, bool update = true);

        Quaternion &setFromAxisAngle(const Vector3 &axis, float angle);

        Quaternion &setFromRotationMatrix(const Matrix4 &m);

        Quaternion &setFromUnitVectors(const Vector3 &vFrom, const Vector3 &vTo);

        [[nodiscard]] float angleTo(const Quaternion &q) const;

        Quaternion &identity();

        Quaternion &invert();

        Quaternion &conjugate();

        [[nodiscard]] float dot(const Quaternion &v) const;

        [[nodiscard]] float lengthSq() const;

        [[nodiscard]] float length() const;

        Quaternion &normalize();

        Quaternion &multiply(const Quaternion &q);

        Quaternion &premultiply(const Quaternion &q);

        Quaternion &multiplyQuaternions(const Quaternion &a, const Quaternion &b);

        [[nodiscard]] bool equals(const Quaternion &v) const;

        bool operator==(const Quaternion &other) const {
            return equals(other);
        }

        Quaternion &_onChange(std::function<void()> callback);

        template<class ArrayLike>
        Quaternion &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x_ = array[offset];
            this->y_ = array[offset + 1];
            this->z_ = array[offset + 2];
            this->w_ = array[offset + 3];

            this->onChangeCallback_();

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            array[offset] = this->x_;
            array[offset + 1] = this->y_;
            array[offset + 2] = this->z_;
            array[offset + 3] = this->w_;
        }

        friend std::ostream &operator<<(std::ostream &os, const Quaternion &v) {
            os << "Quaternion(x=" << v.x_ << ", y=" << v.y_ << ", z=" << v.z_ << ", w=" << v.w_ << ")";
            return os;
        }

    private:
        float x_{0.f};
        float y_{0.f};
        float z_{0.f};
        float w_{1.f};

        std::function<void()> onChangeCallback_ = [] {};
    };

}// namespace threepp

#endif//THREEPP_QUATERNION_HPP
