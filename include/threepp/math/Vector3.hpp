/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector3.js
 */

#ifndef THREEPP_VECTOR3_HPP
#define THREEPP_VECTOR3_HPP

#include <iostream>
#include <string>

namespace threepp {

    class Matrix3;
    class Matrix4;
    class Spherical;
    class Quaternion;
    class Camera;

    class Vector3 {

    public:
        float x{0.f};
        float y{0.f};
        float z{0.f};

        Vector3() = default;

        Vector3(float x, float y, float z);

        Vector3 &set(float x, float y, float z);

        Vector3 &setScalar(float value);

        Vector3 &setX(float value);

        Vector3 &setY(float value);

        Vector3 &setZ(float value);

        float &operator[](unsigned int index);

        Vector3 &copy(const Vector3 &v);

        Vector3 &add(const Vector3 &v);

        Vector3 &addScalar(float s);

        Vector3 &addVectors(const Vector3 &a, const Vector3 &b);

        Vector3 &addScaledVector(const Vector3 &v, float s);

        Vector3 &sub(const Vector3 &v);

        Vector3 &subScalar(float s);

        Vector3 &subVectors(const Vector3 &a, const Vector3 &b);

        Vector3 &multiply(const Vector3 &v);

        Vector3 &multiplyScalar(float scalar);

        Vector3 &multiplyVectors(const Vector3 &a, const Vector3 &b);

        Vector3 &applyMatrix3(const Matrix3 &m);

        Vector3 &applyNormalMatrix(const Matrix3 &m);

        Vector3 &applyMatrix4(const Matrix4 &m);

        Vector3 &applyQuaternion(const Quaternion &q);

        Vector3 &project(const Camera &camera);

        Vector3 &unproject(const Camera &camera);

        Vector3 &transformDirection(const Matrix4 &m);

        Vector3 &divide(const Vector3 &v);

        Vector3 &divideScalar(float v);

        Vector3 &min(const Vector3 &v);

        Vector3 &max(const Vector3 &v);

        Vector3 &clamp(const Vector3 &min, const Vector3 &max);

        Vector3 &floor();

        Vector3 &ceil();

        Vector3 &round();

        Vector3 &roundToZero();

        Vector3 &negate();

        [[nodiscard]] float dot(const Vector3 &v) const;

        [[nodiscard]] float lengthSq() const;

        [[nodiscard]] float length() const;

        [[nodiscard]] float manhattanLength() const;

        Vector3 &normalize();

        Vector3 &setLength(float length);

        Vector3 &lerp(const Vector3 &v, float alpha);

        Vector3 &lerpVectors(const Vector3 &v1, const Vector3 &v2, float alpha);

        Vector3 &cross(const Vector3 &v);

        Vector3 &crossVectors(const Vector3 &a, const Vector3 &b);

        Vector3 &projectOnVector(const Vector3 &v);

        Vector3 &projectOnPlane(const Vector3 &planeNormal);

        Vector3 &reflect(const Vector3 &normal);

        [[nodiscard]] float angleTo(const Vector3 &v) const;

        [[nodiscard]] float distanceTo(const Vector3 &v) const;

        [[nodiscard]] float distanceToSquared(const Vector3 &v) const;

        [[nodiscard]] float manhattanDistanceTo(const Vector3 &v) const;

        Vector3 &setFromSpherical(const Spherical &s);

        Vector3 &setFromSphericalCoords(float radius, float phi, float theta);

        Vector3 &setFromMatrixPosition(const Matrix4 &m);

        Vector3 &setFromMatrixScale(const Matrix4 &m);

        Vector3 &setFromMatrixColumn(const Matrix4 &m, unsigned int index);

        Vector3 &setFromMatrix3Column(const Matrix3 &m, unsigned int index);

        [[nodiscard]] Vector3 clone() const;

        [[nodiscard]] bool equals(const Vector3 &v) const;

        bool operator==(const Vector3 &other) const {
            return equals(other);
        }

        bool operator!=(const Vector3 &other) const {
            return !equals(other);
        }

        Vector3 operator+(const Vector3 &other) const {

            return clone().add(other);
        }

        Vector3 &operator+=(const Vector3 &other) {

            return add(other);
        }

        Vector3 operator+(float s) const {

            return clone().addScalar(s);
        }

        Vector3 &operator+=(float s) {

            return addScalar(s);
        }

        Vector3 operator-(const Vector3 &other) const {

            return clone().sub(other);
        }

        Vector3 &operator-=(const Vector3 &other) {

            return sub(other);
        }

        Vector3 operator-(float s) const {

            return clone().subScalar(s);
        }

        Vector3 &operator-=(float s) {

            return subScalar(s);
        }

        Vector3 operator*(const Vector3 &other) const {

            return clone().multiply(other);
        }

        Vector3 &operator*=(const Vector3 &other) {

            return multiply(other);
        }

        Vector3 operator*(float s) const {

            return clone().multiplyScalar(s);
        }

        Vector3 &operator*=(float s) {

            return multiplyScalar(s);
        }

        Vector3 operator/(const Vector3 &other) const {

            return clone().divide(other);
        }

        Vector3 &operator/=(const Vector3 &other) {

            return divide(other);
        }

        Vector3 operator/(float s) const {

            return clone().divideScalar(s);
        }

        Vector3 &operator/=(float s) {

            return divideScalar(s);
        }

        template<class ArrayLike>
        Vector3 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x = array[offset];
            this->y = array[offset + 1];
            this->z = array[offset + 2];

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            array[offset] = this->x;
            array[offset + 1] = this->y;
            array[offset + 2] = this->z;
        }

        static const Vector3 X;
        static const Vector3 Y;
        static const Vector3 Z;

        static const Vector3 ONES;
        static const Vector3 ZEROS;

        friend std::ostream &operator<<(std::ostream &os, const Vector3 &v) {
            os << "Vector3(x=" << v.x << ", y=" << v.y << ", z=" << v.z << ")";
            return os;
        }
    };

}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
