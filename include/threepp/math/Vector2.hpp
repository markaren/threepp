// https://github.com/mrdoob/three.js/blob/r129/src/math/Vector2.js

#ifndef THREEPP_VECTOR2_HPP
#define THREEPP_VECTOR2_HPP

#include "threepp/math/Matrix3.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>

namespace threepp {

    class Matrix3;

    /**
     * Class representing a 2D vector. A 2D vector is an ordered pair of numbers (labeled x and y),
     * which can be used to represent a number of things, such as:
     *
     * A point in 2D space (i.e. a position on a plane).
     *   - A direction and length across a plane. In threepp the length will always be the Euclidean distance (straight-line distance) from (0, 0) to (x, y) and the direction is also measured from (0, 0) towards (x, y).
     *   - Any arbitrary ordered pair of numbers.
     *   - There are other things a 2D vector can be used to represent, such as momentum vectors, complex numbers and so on, however these are the most common uses in three.js.
     *
     * Iterating through a Vector2 instance will yield its components (x, y) in the corresponding order.
     */
    class Vector2 {

    public:
        float x{0.f};
        float y{0.f};

        Vector2() = default;

        Vector2(int x, int y)
            : Vector2(static_cast<float>(x), static_cast<float>(y)) {}

        Vector2(float x, float y)
            : x(x), y(y) {}

        Vector2(double x, double y)
            : x(x), y(y) {}

        Vector2& set(float x, float y) {
            this->x = x;
            this->y = y;

            return *this;
        }

        Vector2& setScalar(float value) {
            this->x = value;
            this->y = value;

            return *this;
        }

        Vector2& setX(float value) {
            this->x = value;

            return *this;
        }

        Vector2& setY(float value) {
            this->y = value;

            return *this;
        }

        float& operator[](unsigned int index) {
            if (index >= 2) throw std::runtime_error("index out of bounds: " + std::to_string(index));
            switch (index) {
                case 0:
                    return x;
                case 1:
                    return y;
                default:
                    throw std::runtime_error("index out of bound: " + std::to_string(index));
            }
        }

        float operator[](unsigned int index) const {
            if (index >= 2) throw std::runtime_error("index out of bounds: " + std::to_string(index));
            switch (index) {
                case 0:
                    return x;
                case 1:
                    return y;
                default:
                    throw std::runtime_error("index out of bound: " + std::to_string(index));
            }
        }

        Vector2& copy(const Vector2& v) {
            this->x = v.x;
            this->y = v.y;

            return *this;
        }

        Vector2& add(const Vector2& v) {
            this->x += v.x;
            this->y += v.y;

            return *this;
        }

        Vector2& addScalar(float s) {
            this->x += s;
            this->y += s;

            return *this;
        }

        Vector2& addVectors(const Vector2& a, const Vector2& b) {
            this->x = a.x + b.x;
            this->y = a.y + b.y;

            return *this;
        }

        Vector2& addScaledVector(const Vector2& v, float s) {
            this->x += v.x * s;
            this->y += v.y * s;

            return *this;
        }

        Vector2& sub(const Vector2& v) {
            this->x -= v.x;
            this->y -= v.y;

            return *this;
        }

        Vector2& subScalar(float s) {
            this->x -= s;
            this->y -= s;

            return *this;
        }

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

        Vector2& multiplyScalar(float scalar) {
            this->x *= scalar;
            this->y *= scalar;

            return *this;
        }

        Vector2& divide(const Vector2& v) {
            this->x /= v.x;
            this->y /= v.y;

            return *this;
        }

        Vector2& divideScalar(float scalar) {
            return this->multiplyScalar(1.0f / scalar);
        }

        Vector2& applyMatrix3(const Matrix3& m) {
            const auto x_ = this->x, y_ = this->y;
            const auto& e = m.elements;

            this->x = e[0] * x_ + e[3] * y_ + e[6];
            this->y = e[1] * x_ + e[4] * y_ + e[7];

            return *this;
        }

        Vector2& min(const Vector2& v) {
            this->x = std::min(this->x, v.x);
            this->y = std::min(this->y, v.y);

            return *this;
        }

        Vector2& max(const Vector2& v) {
            this->x = std::max(this->x, v.x);
            this->y = std::max(this->y, v.y);

            return *this;
        }

        Vector2& clamp(const Vector2& min, const Vector2& max) {
            // assumes min < max, componentwise

            this->x = std::max(min.x, std::min(max.x, this->x));
            this->y = std::max(min.y, std::min(max.y, this->y));

            return *this;
        }

        Vector2& clampScalar(float minVal, float maxVal) {
            this->x = std::max(minVal, std::min(maxVal, this->x));
            this->y = std::max(minVal, std::min(maxVal, this->y));

            return *this;
        }

        Vector2& clampLength(float min, float max) {
            const auto length = this->length();

            return this->divideScalar(std::isnan(length) ? 1 : length).multiplyScalar(std::max(min, std::min(max, length)));
        }

        Vector2& floor() {
            this->x = std::floor(this->x);
            this->y = std::floor(this->y);

            return *this;
        }

        Vector2& ceil() {
            this->x = std::ceil(this->x);
            this->y = std::ceil(this->y);

            return *this;
        }

        Vector2& round() {
            this->x = std::round(this->x);
            this->y = std::round(this->y);

            return *this;
        }

        Vector2& roundToZero() {
            this->x = (this->x < 0) ? std::ceil(this->x) : std::floor(this->x);
            this->y = (this->y < 0) ? std::ceil(this->y) : std::floor(this->y);

            return *this;
        }

        Vector2& negate() {
            this->x = -this->x;
            this->y = -this->y;

            return *this;
        }

        [[nodiscard]] float dot(const Vector2& v) const {
            return this->x * v.x + this->y * v.y;
        }

        [[nodiscard]] float cross(const Vector2& v) const {
            return this->x * v.y - this->y * v.x;
        }

        [[nodiscard]] float lengthSq() const {
            return this->x * this->x + this->y * this->y;
        }

        [[nodiscard]] float length() const {
            return std::sqrt(this->x * this->x + this->y * this->y);
        }

        [[nodiscard]] float manhattanLength() const {
            return std::abs(this->x) + std::abs(this->y);
        }

        Vector2& normalize() {
            const auto len = this->length();
            return this->divideScalar(std::isnan(len) ? 1 : len);
        }

        [[nodiscard]] float angle() const;

        [[nodiscard]] float angleTo(const Vector2& v) const;

        [[nodiscard]] float distanceTo(const Vector2& v) const {
            return std::sqrt(this->distanceToSquared(v));
        }

        [[nodiscard]] float distanceToSquared(const Vector2& v) const {
            const auto dx = this->x - v.x, dy = this->y - v.y;
            return dx * dx + dy * dy;
        }

        [[nodiscard]] float manhattanDistanceTo(const Vector2& v) const {
            return std::abs(this->x - v.x) + std::abs(this->y - v.y);
        }

        Vector2& setLength(float length) {
            return this->normalize().multiplyScalar(length);
        }

        Vector2& lerp(const Vector2& v, float alpha) {
            this->x += (v.x - this->x) * alpha;
            this->y += (v.y - this->y) * alpha;

            return *this;
        }

        Vector2& lerpVectors(const Vector2& v1, const Vector2& v2, float alpha) {
            this->x = v1.x + (v2.x - v1.x) * alpha;
            this->y = v1.y + (v2.y - v1.y) * alpha;

            return *this;
        }

        Vector2& rotateAround(const Vector2& center, float angle) {
            float c = std::cos(angle), s = std::sin(angle);

            float x = this->x - center.x;
            float y = this->y - center.y;

            this->x = x * c - y * s + center.x;
            this->y = x * s + y * c + center.y;

            return *this;
        }

        [[nodiscard]] bool isNan() const {
            return std::isnan(x) || std::isnan(y);
        }

        Vector2& makeNan() {
            return set(NAN, NAN);
        }

        [[nodiscard]] Vector2 clone() const {
            return {x, y};
        }

        [[nodiscard]] bool equals(const Vector2& v) const {
            return ((v.x == this->x) && (v.y == this->y));
        }

        bool operator==(const Vector2& other) const {
            return equals(other);
        }
        bool operator!=(const Vector2& other) const {
            return !equals(other);
        }

        Vector2 operator+(const Vector2& other) const {
            return clone().add(other);
        }
        Vector2& operator+=(const Vector2& other) {
            return add(other);
        }

        Vector2 operator-(const Vector2& other) const {
            return clone().sub(other);
        }
        Vector2& operator-=(const Vector2& other) {
            return sub(other);
        }

        Vector2 operator+(float value) const {
            return Vector2(x, y).addScalar(value);
        }
        Vector2 operator-(float value) const {
            return Vector2(x, y).subScalar(value);
        }
        Vector2 operator*(float value) const {
            return Vector2(x, y).multiplyScalar(value);
        }
        Vector2 operator/(float value) const {
            return Vector2(x, y).divideScalar(value);
        }

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

        operator std::pair<int, int>() const;

        operator std::pair<float, float>() const;

        friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
            os << "Vector2(x=" << v.x << ", y=" << v.y << ")";
            return os;
        }
    };

    // Implementing get function template
    template<std::size_t N>
    auto get(const Vector2& p) {
        if constexpr (N == 0) return p.x;
        else if constexpr (N == 1)
            return p.y;
    }

}// namespace threepp

#endif//THREEPP_VECTOR2_HPP
