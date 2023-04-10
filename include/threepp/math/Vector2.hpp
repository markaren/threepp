// https://github.com/mrdoob/three.js/blob/r129/src/math/Vector2.js

#ifndef THREEPP_VECTOR2_HPP
#define THREEPP_VECTOR2_HPP

#include <ostream>
#include <string>

namespace threepp {

    class Matrix3;

    class Vector2 {

    public:
        float x{0.f};
        float y{0.f};

        Vector2() = default;

        Vector2(int x, int y);

        Vector2(float x, float y);

        Vector2& set(float x, float y);

        Vector2& setScalar(float value);

        Vector2& setX(float value);

        Vector2& setY(float value);

        float& operator[](unsigned int index);

        Vector2& copy(const Vector2& v);

        Vector2& add(const Vector2& v);

        Vector2& addScalar(float s);

        Vector2& addVectors(const Vector2& a, const Vector2& b);

        Vector2& addScaledVector(const Vector2& v, float s);

        Vector2& sub(const Vector2& v);

        Vector2& subScalar(float s);

        template<class VectorA, class VectorB>
        Vector2& subVectors(const VectorA& a, const VectorB& b) {
            this->x = a.x - b.x;
            this->y = a.y - b.y;

            return *this;
        }

        template<class Vector>
        Vector2& multiply(const Vector& v) {
            this->x *= v.x;
            this->y *= v.y;

            return *this;
        }

        Vector2& multiplyScalar(float scalar);

        Vector2& divide(const Vector2& v);

        Vector2& divideScalar(float scalar);

        Vector2& applyMatrix3(const Matrix3& m);

        Vector2& min(const Vector2& v);

        Vector2& max(const Vector2& v);

        Vector2& clamp(const Vector2& min, const Vector2& max);

        Vector2& clampScalar(float minVal, float maxVal);

        Vector2& clampLength(float min, float max);

        Vector2& floor();

        Vector2& ceil();

        Vector2& round();

        Vector2& roundToZero();

        Vector2& negate();

        [[nodiscard]] float dot(const Vector2& v) const;

        [[nodiscard]] float cross(const Vector2& v) const;

        [[nodiscard]] float lengthSq() const;

        [[nodiscard]] float length() const;

        [[nodiscard]] float manhattanLength() const;

        Vector2& normalize();

        [[nodiscard]] float angle() const;

        [[nodiscard]] float angleTo(const Vector2& v) const;

        [[nodiscard]] float distanceTo(const Vector2& v) const;

        [[nodiscard]] float distanceToSquared(const Vector2& v) const;

        [[nodiscard]] float manhattanDistanceTo(const Vector2& v) const;

        Vector2& setLength(float length);

        Vector2& lerp(const Vector2& v, float alpha);

        Vector2& lerpVectors(const Vector2& v1, const Vector2& v2, float alpha);

        [[nodiscard]] bool isNan() const;

        Vector2& makeNan();

        [[nodiscard]] Vector2 clone() const;

        [[nodiscard]] bool equals(const Vector2& v) const;

        bool operator==(const Vector2& other) const;

        bool operator!=(const Vector2& other) const;

        Vector2 operator+(const Vector2& other) const;

        Vector2& operator+=(const Vector2& other);

        Vector2 operator-(const Vector2& other) const;

        Vector2& operator-=(const Vector2& other);

        template<class ArrayLike>
        Vector2& fromArray(const ArrayLike& array, unsigned int offset = 0) {

            this->x = array[offset];
            this->y = array[offset + 1];

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike& array, unsigned int offset = 0) const {

            array[offset] = this->x;
            array[offset + 1] = this->y;
        }

        friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
            os << "Vector2(x=" << v.x << ", y=" << v.y << ")";
            return os;
        }
    };

}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
