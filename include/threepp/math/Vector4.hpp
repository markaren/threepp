// https://github.com/mrdoob/three.js/blob/r129/src/math/Vector4.js

#ifndef THREEPP_VECTOR4_HPP
#define THREEPP_VECTOR4_HPP

#include <ostream>

namespace threepp {

    class Matrix3;
    class Matrix4;

    class Vector4 {

    public:
        float x{0.f};
        float y{0.f};
        float z{0.f};
        float w{1.f};

        Vector4() = default;

        Vector4(int x, int y, int z, int w);

        Vector4(float x, float y, float z, float w);

        float& operator[](unsigned int index);

        Vector4& set(float x, float y, float z, float w);

        Vector4& setScalar(float value);

        Vector4& copy(const Vector4& v);

        Vector4& add(const Vector4& v);

        Vector4& addScalar(float s);

        Vector4& addVectors(const Vector4& a, const Vector4& b);

        Vector4& addScaledVector(const Vector4& v, float s);

        Vector4& multiply(const Vector4& v);

        Vector4& multiplyScalar(float scalar);

        Vector4& applyMatrix4(const Matrix4& m);

        Vector4& divideScalar(float scalar);

        Vector4& floor();

        Vector4& ceil();

        Vector4& round();

        Vector4& roundToZero();

        Vector4& negate();

        [[nodiscard]] float dot(const Vector4& v) const;

        [[nodiscard]] float lengthSq() const;

        [[nodiscard]] float length() const;

        [[nodiscard]] float manhattanLength() const;

        Vector4& normalize();

        Vector4& setLength(float length);

        [[nodiscard]] Vector4 clone() const;

        [[nodiscard]] bool equals(const Vector4& v) const;

        bool operator==(const Vector4& other) const;

        bool operator!=(const Vector4& other) const;

        template<class ArrayLike>
        Vector4& fromArray(const ArrayLike& array, unsigned int offset = 0) {

            this->x = array[offset + 0];
            this->y = array[offset + 1];
            this->z = array[offset + 2];
            this->w = array[offset + 2];

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike& array, unsigned int offset = 0) const {

            array[offset + 0] = this->x;
            array[offset + 1] = this->y;
            array[offset + 2] = this->z;
            array[offset + 3] = this->w;
        }

        friend std::ostream& operator<<(std::ostream& os, const Vector4& v) {
            os << "Vector4(x=" << v.x << ", y=" << v.y << ", z=" << v.z << ", w=" << v.w << ")";
            return os;
        }
    };

}// namespace threepp

#endif//THREEPP_VECTOR4_HPP
