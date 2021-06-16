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

    class Vector3 {

    public:
        float x = 0.0;
        float y = 0.0;
        float z = 0.0;

        Vector3() = default;

        Vector3(float x, float y, float z);

        Vector3 &set(float x, float y, float z);

        Vector3 &setScalar(float value);

        Vector3 &setX(float value);

        Vector3 &setY(float value);

        Vector3 &setZ(float value);

        float &operator[](unsigned int index);

        Vector3 &add(const Vector3 &v);

        Vector3 &add(float s);

        Vector3 &addVectors(const Vector3 &a, const Vector3 &b);

        Vector3 &addScaledVector(const Vector3 &v, float s);

        Vector3 &sub(const Vector3 &v);

        Vector3 &sub(float s);

        Vector3 &subVectors(const Vector3 &a, const Vector3 &b);

        Vector3 &multiply(const Vector3 &v);

        Vector3 &multiply(float scalar);

        Vector3 &multiplyVectors(const Vector3 &a, const Vector3 &b);

        Vector3 &applyMatrix3(const Matrix3 &m);

        Vector3 &applyNormalMatrix(const Matrix3 &m);

        Vector3 &applyMatrix4(const Matrix4 &m);

        Vector3 &divide(const Vector3 &v);

        Vector3 &divide(const float &v);

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

        [[nodiscard]] float angleTo(const Vector3 &v) const;

        [[nodiscard]] float distanceTo(const Vector3 &v) const;

        [[nodiscard]] float distanceToSquared(const Vector3 &v) const;

        [[nodiscard]] float manhattanDistanceTo(const Vector3 &v) const;

        Vector3 &setFromMatrixPosition(const Matrix4 &m);

        Vector3 &setFromMatrixScale(const Matrix4 &m);

        Vector3 &setFromMatrixColumn(const Matrix4 &m, unsigned int index);

        Vector3 &setFromMatrix3Column(const Matrix3 &m, unsigned int index);


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

    };


//    std::ostream &operator<<(std::ostream &os, const threepp::Vector3 &v);

}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
