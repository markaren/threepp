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
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        vector3() = default;

        vector3(double x, double y, double z);;

        vector3 &set(double x, double y, double z);

        vector3 &setScalar(double value);

        vector3 &setX(double value);

        vector3 &setY(double value);

        vector3 &setZ(double value);

        double &operator[](unsigned int index);

        vector3 &add(const vector3 &v);

        vector3 &add(double s);

        vector3 &addVectors(const vector3 &a, const vector3 &b);

        vector3 &addScaledVector(const vector3 &v, double s);

        vector3 &sub(const vector3 &v);

        vector3 &sub(double s);

        vector3 &subVectors(const vector3 &a, const vector3 &b);

        vector3 &multiply(const vector3 &v);

        vector3 &multiply(double scalar);

        vector3 &multiplyVectors(const vector3 &a, const vector3 &b);

        vector3 &applyMatrix3(const matrix3 &m);

        vector3 &applyNormalMatrix(const matrix3 &m);

        vector3 &applyMatrix4(const matrix4 &m);

        vector3 &divide(const vector3 &v);

        vector3 &divide(const double &v);

        vector3 &min(const vector3 &v);

        vector3 &max(const vector3 &v);

        vector3 &clamp(const vector3 &min, const vector3 &max);

        vector3 &floor();

        vector3 &ceil();

        vector3 &round();

        vector3 &roundToZero();

        vector3 &negate();

        [[nodiscard]] double dot(const vector3 &v) const;

        [[nodiscard]] double lengthSq() const;

        [[nodiscard]] double length() const;

        [[nodiscard]] double manhattanLength() const;

        vector3 &normalize();

        vector3 &setLength(double length);

        vector3 &lerp(const vector3 &v, double alpha);

        vector3 &lerpVectors(const vector3 &v1, const vector3 &v2, double alpha);

        vector3 &cross(const vector3 &v);

        vector3 &crossVectors(const vector3 &a, const vector3 &b);

        [[nodiscard]] double angleTo(const vector3 &v) const;

        [[nodiscard]] double distanceTo(const vector3 &v) const;

        [[nodiscard]] double distanceToSquared(const vector3 &v) const;

        [[nodiscard]] double manhattanDistanceTo(const vector3 &v) const;

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
