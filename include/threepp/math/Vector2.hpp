/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector2.js
 */

#ifndef THREEPP_VECTOR2_HPP
#define THREEPP_VECTOR2_HPP

#include <string>
#include <iostream>

namespace threepp {

    class Matrix3;

    class Vector2 {

    public:
        float x = 0.0;
        float y = 0.0;

        Vector2() = default;

        Vector2(float x, float y);

        Vector2 &set(float x, float y);

        Vector2 &setScalar(float value);

        Vector2 &setX(float value);

        Vector2 &setY(float value);

        float &operator[](unsigned int index);

        Vector2 &copy( const Vector2 &v );

        Vector2 &add(const Vector2 &v);

        Vector2 &addScalar(float s);

        Vector2 &addVectors(const Vector2 &a, const Vector2 &b);

        Vector2 &addScaledVector(const Vector2 &v, float s);

        Vector2 &sub( const Vector2 &v );

        Vector2 &subScalar( float s );

        Vector2 &subVectors( const Vector2 &a, const Vector2 &b );

        Vector2 &multiply( const Vector2 &v );

        Vector2 &multiplyScalar( float scalar );

        Vector2 &divide( const Vector2 &v );

        Vector2 &divideScalar( float scalar );

        Vector2 &applyMatrix3( const Matrix3 &m );

        Vector2 &min( const Vector2 &v );

        Vector2 &max( const Vector2 &v );

        Vector2 &clamp( const Vector2 &min, const Vector2 &max );

        Vector2 &clampScalar( float minVal, float maxVal );

        Vector2 &clampLength( float min, float max );

        Vector2 &floor();

        Vector2 &ceil();

        Vector2 &round();

        Vector2 &roundToZero();

        Vector2 &negate();

        [[nodiscard]] float dot( const Vector2 &v ) const;

        [[nodiscard]] float cross( const Vector2 &v ) const;

        [[nodiscard]] float lengthSq() const;

        [[nodiscard]] float length() const;

        [[nodiscard]] float manhattanLength() const;

        Vector2 &normalize();

        [[nodiscard]] float angle() const;

        float distanceTo( const Vector2 &v );

        [[nodiscard]] float distanceToSquared( const Vector2 &v ) const;

        [[nodiscard]] float manhattanDistanceTo( const Vector2 &v ) const;

        Vector2 &setLength( float length );

        Vector2 &lerp( const Vector2 &v, float alpha );

        Vector2 &lerpVectors( const Vector2 &v1, const Vector2 &v2, float alpha );

        [[nodiscard]] bool equals( const Vector2 &v ) const;

        bool operator==(const Vector2 &other) const {
            return equals(other);
        }

        template<class ArrayLike>
        Vector2 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x = array[offset];
            this->y = array[offset + 1];

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            array[offset] = this->x;
            array[offset + 1] = this->y;
        }

        friend std::ostream &operator<<(std::ostream &os, const Vector2 &v) {
            os << "Vector2(x=" << v.x << ", y=" << v.y << ")";
            return os;
        }

    };

}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
