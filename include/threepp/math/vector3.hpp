/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector3.js
 */

#ifndef THREEPP_VECTOR3_HPP
#define THREEPP_VECTOR3_HPP

#include <iostream>
#include <string>

namespace threepp {

    class matrix3;
    class matrix4;

    class vector3 {

    public:
        float x = 0.0;
        float y = 0.0;
        float z = 0.0;

        vector3() = default;

        vector3(float x, float y, float z);

        vector3 &set(float x, float y, float z);

        vector3 &setScalar(float value);

        vector3 &setX(float value);

        vector3 &setY(float value);

        vector3 &setZ(float value);

        float &operator[](unsigned int index);

        vector3 &add(const vector3 &v);

        vector3 &add(float s);

        vector3 &addVectors(const vector3 &a, const vector3 &b);

        vector3 &addScaledVector(const vector3 &v, float s);

        vector3 &sub(const vector3 &v);

        vector3 &sub(float s);

        vector3 &subVectors(const vector3 &a, const vector3 &b);

        vector3 &multiply(const vector3 &v);

        vector3 &multiply(float scalar);

        vector3 &multiplyVectors(const vector3 &a, const vector3 &b);

        vector3 &applyMatrix3(const matrix3 &m);

        vector3 &applyNormalMatrix(const matrix3 &m);

        vector3 &applyMatrix4(const matrix4 &m);

        vector3 &divide(const vector3 &v);

        vector3 &divide(const float &v);

        vector3 &min(const vector3 &v);

        vector3 &max(const vector3 &v);

        vector3 &clamp(const vector3 &min, const vector3 &max);

        vector3 &floor();

        vector3 &ceil();

        vector3 &round();

        vector3 &roundToZero();

        vector3 &negate();

        [[nodiscard]] float dot(const vector3 &v) const;

        [[nodiscard]] float lengthSq() const;

        [[nodiscard]] float length() const;

        [[nodiscard]] float manhattanLength() const;

        vector3 &normalize();

        vector3 &setLength(float length);

        vector3 &lerp(const vector3 &v, float alpha);

        vector3 &lerpVectors(const vector3 &v1, const vector3 &v2, float alpha);

        vector3 &cross(const vector3 &v);

        vector3 &crossVectors(const vector3 &a, const vector3 &b);

        [[nodiscard]] float angleTo(const vector3 &v) const;

        [[nodiscard]] float distanceTo(const vector3 &v) const;

        [[nodiscard]] float distanceToSquared(const vector3 &v) const;

        [[nodiscard]] float manhattanDistanceTo(const vector3 &v) const;

        vector3 &setFromMatrixPosition(const matrix4 &m);

        vector3 &setFromMatrixScale(const matrix4 &m);

        vector3 &setFromMatrixColumn(const matrix4 &m, unsigned int index);

        vector3 &setFromMatrix3Column(const matrix3 &m, unsigned int index);


        template<class ArrayLike>
        vector3 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

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


//    std::ostream &operator<<(std::ostream &os, const threepp::vector3 &v);

}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
