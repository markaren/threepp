/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector2.js
 */

#ifndef THREEPP_VECTOR2_HPP
#define THREEPP_VECTOR2_HPP

#include <string>

namespace threepp {

    class Matrix3;

    template <typename T>
    class BufferAttribute;

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

        Vector2 &add(const Vector2 &v);

        Vector2 &add(float s);

        Vector2 &addVectors(const Vector2 &a, const Vector2 &b);

        Vector2 &addScaledVector(const Vector2 &v, float s);

        Vector2 &sub( const Vector2 &v );

        Vector2 &sub( float s );

        Vector2 &subVectors( const Vector2 &a, const Vector2 &b );

        Vector2 &multiply( const Vector2 &v );

        Vector2 &multiply( float scalar );

        Vector2 &divide( const Vector2 &v );

        Vector2 &divide( float scalar );

        Vector2 &applyMatrix3( const Matrix3 &m );

        Vector2 &min( const Vector2 &v );

        Vector2 &max( const Vector2 &v );

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

        Vector2 &fromBufferAttribute( const BufferAttribute<float> &attribute, int index );

    };

    //    std::ostream &operator<<(std::ostream &os, const Vector2 &v) {
    //        return os << "Vector2(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ")";
    //    }

}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
