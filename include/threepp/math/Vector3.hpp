// https://github.com/mrdoob/three.js/blob/r129/src/math/Vector3.js

#ifndef THREEPP_VECTOR3_HPP
#define THREEPP_VECTOR3_HPP

#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>

namespace threepp {

    class Euler;
    class Matrix3;
    class Matrix4;
    class Spherical;
    class Quaternion;
    class Camera;

    /*
     * Class representing a 3D vector.
     * A 3D vector is an ordered triplet of numbers (labeled x, y, and z),
     * which can be used to represent a number of things, such as:
     *
     * A point in 3D space.
     *  - A direction and length in 3D space. In three.js the length will always be the Euclidean distance (straight-line distance) from (0, 0, 0) to (x, y, z) and the direction is also measured from (0, 0, 0) towards (x, y, z).
     *  - Any arbitrary ordered triplet of numbers.
     *  - There are other things a 3D vector can be used to represent, such as momentum vectors and so on, however these are the most common uses in three.js.
     *
     * Iterating through a Vector3 instance will yield its components (x, y, z) in the corresponding order.
     */
    class Vector3 {

    public:
        float x;
        float y;
        float z;

        Vector3()
            : Vector3(0, 0, 0) {}
        Vector3(float x, float y, float z)
            : x(x), y(y), z(z) {}

        Vector3(const Vector3& other) = default;

        // enable implicit conversions from Vector3 like objects
        template<typename T>
            requires requires(T t) {
                { t.x } -> std::convertible_to<float>;
                { t.y } -> std::convertible_to<float>;
                { t.z } -> std::convertible_to<float>;
            }
        Vector3(const T& other)
            : x(static_cast<float>(other.x)),
              y(static_cast<float>(other.y)),
              z(static_cast<float>(other.z)) {}

        // Sets the x, y and z components of this vector.
        Vector3& set(float x, float y, float z) {
            this->x = x;
            this->y = y;
            this->z = z;

            return *this;
        }

        // Set the x, y and z values of this vector both equal to scalar.
        Vector3& setScalar(float value) {
            this->x = value;
            this->y = value;
            this->z = value;

            return *this;
        }

        Vector3& setX(float value) {
            this->x = value;

            return *this;
        }

        Vector3& setY(float value) {
            this->y = value;

            return *this;
        }

        Vector3& setZ(float value) {
            this->z = value;

            return *this;
        }

        float& operator[](size_t index) {
            switch (index) {
                case 0:
                    return x;
                case 1:
                    return y;
                case 2:
                    return z;
                default:
                    throw std::runtime_error("index out of bound: " + std::to_string(index));
            }
        }

        float operator[](size_t index) const {
            switch (index) {
                case 0:
                    return x;
                case 1:
                    return y;
                case 2:
                    return z;
                default:
                    throw std::runtime_error("index out of bound: " + std::to_string(index));
            }
        }

        Vector3& copy(const Vector3& v) {
            this->x = v.x;
            this->y = v.y;
            this->z = v.z;

            return *this;
        }

        // Adds v to this vector.
        Vector3& add(const Vector3& v) {
            this->x += v.x;
            this->y += v.y;
            this->z += v.z;

            return *this;
        }

        // Adds the scalar value s to this vector's x, y and z values.
        Vector3& addScalar(float s) {
            this->x += s;
            this->y += s;
            this->z += s;

            return *this;
        }

        // Sets this vector to a + b.
        Vector3& addVectors(const Vector3& a, const Vector3& b) {
            this->x = a.x + b.x;
            this->y = a.y + b.y;
            this->z = a.z + b.z;

            return *this;
        }

        // Adds the multiple of v and s to this vector.
        Vector3& addScaledVector(const Vector3& v, float s) {
            this->x += v.x * s;
            this->y += v.y * s;
            this->z += v.z * s;

            return *this;
        }

        // Subtracts v from this vector.
        Vector3& sub(const Vector3& v) {
            this->x -= v.x;
            this->y -= v.y;
            this->z -= v.z;

            return *this;
        }

        // Subtracts s from this vector's x, y and z components.
        Vector3& subScalar(float s) {
            this->x -= s;
            this->y -= s;
            this->z -= s;

            return *this;
        }

        // Sets this vector to a - b.
        Vector3& subVectors(const Vector3& a, const Vector3& b) {
            this->x = a.x - b.x;
            this->y = a.y - b.y;
            this->z = a.z - b.z;

            return *this;
        }

        // Multiplies this vector by v.
        Vector3& multiply(const Vector3& v) {
            this->x *= v.x;
            this->y *= v.y;
            this->z *= v.z;

            return *this;
        }

        // Multiplies this vector by scalar s.
        Vector3& multiplyScalar(float scalar) {
            this->x *= scalar;
            this->y *= scalar;
            this->z *= scalar;

            return *this;
        }

        // Sets this vector equal to a * b, component-wise.
        Vector3& multiplyVectors(const Vector3& a, const Vector3& b) {
            this->x = a.x * b.x;
            this->y = a.y * b.y;
            this->z = a.z * b.z;

            return *this;
        }

        // Applies a rotation specified by an axis and an angle to this vector.
        Vector3& applyAxisAngle(const Vector3& axis, float angle);

        // Applies euler transform to this vector by converting the Euler object to a Quaternion and applying.
        Vector3& applyEuler(const Euler& euler);

        // Multiplies this vector by m
        Vector3& applyMatrix3(const Matrix3& m) {
            const auto x_ = this->x, y_ = this->y, z_ = this->z;
            const auto& e = m.elements;

            this->x = e[0] * x_ + e[3] * y_ + e[6] * z_;
            this->y = e[1] * x_ + e[4] * y_ + e[7] * z_;
            this->z = e[2] * x_ + e[5] * y_ + e[8] * z_;

            return *this;
        }

        // Multiplies this vector by normal matrix m and normalizes the result.
        Vector3& applyNormalMatrix(const Matrix3& m) {
            return applyMatrix3(m).normalize();
        }

        // Multiplies this vector (with an implicit 1 in the 4th dimension) by m, and divides by perspective.
        Vector3& applyMatrix4(const Matrix4& m) {
            const auto x_ = this->x, y_ = this->y, z_ = this->z;
            const auto& e = m.elements;

            const auto w = 1.0f / (e[3] * x + e[7] * y + e[11] * z + e[15]);

            this->x = (e[0] * x_ + e[4] * y_ + e[8] * z_ + e[12]) * w;
            this->y = (e[1] * x_ + e[5] * y_ + e[9] * z_ + e[13]) * w;
            this->z = (e[2] * x_ + e[6] * y_ + e[10] * z_ + e[14]) * w;

            return *this;
        }

        // Applies a Quaternion transform to this vector.
        Vector3& applyQuaternion(const Quaternion& q);

        // Projects this vector from world space into the camera's normalized device coordinate (NDC) space.
        Vector3& project(const Camera& camera);

        // Projects this vector from the camera's normalized device coordinate (NDC) space into world space.
        Vector3& unproject(const Camera& camera);

        // Transforms the direction of this vector by a matrix (the upper left 3 x 3 subset of a m) and then normalizes the result.
        Vector3& transformDirection(const Matrix4& m) {
            // input: THREE.Matrix4 affine matrix
            // vector interpreted as a direction

            const auto x = this->x, y = this->y, z = this->z;
            const auto& e = m.elements;

            this->x = e[0] * x + e[4] * y + e[8] * z;
            this->y = e[1] * x + e[5] * y + e[9] * z;
            this->z = e[2] * x + e[6] * y + e[10] * z;

            return this->normalize();
        }

        // Divides this vector by v.
        Vector3& divide(const Vector3& v) {
            this->x /= v.x;
            this->y /= v.y;
            this->z /= v.z;

            return *this;
        }

        // Divides this vector by scalar s.
        Vector3& divideScalar(float v) {
            this->x /= v;
            this->y /= v;
            this->z /= v;

            return *this;
        }

        Vector3& min(const Vector3& v) {
            this->x = std::min(this->x, v.x);
            this->y = std::min(this->y, v.y);
            this->z = std::min(this->z, v.z);

            return *this;
        }

        Vector3& max(const Vector3& v) {
            this->x = std::max(this->x, v.x);
            this->y = std::max(this->y, v.y);
            this->z = std::max(this->z, v.z);

            return *this;
        }

        // If this vector's x, y or z value is greater than the max vector's x, y or z value, it is replaced by the corresponding value.
        // If this vector's x, y or z value is less than the min vector's x, y or z value, it is replaced by the corresponding value.
        Vector3& clamp(const Vector3& min, const Vector3& max) {
            // assumes min < max, componentwise

            this->x = std::max(min.x, std::min(max.x, this->x));
            this->y = std::max(min.y, std::min(max.y, this->y));
            this->z = std::max(min.z, std::min(max.z, this->z));

            return *this;
        }

        // If this vector's length is greater than the max value, the vector will be scaled down so its length is the max value.
        // If this vector's length is less than the min value, the vector will be scaled up so its length is the min value.
        Vector3& clampLength(float min, float max) {
            const auto length = this->length();

            return this->divideScalar(length != 0 ? length : 1).multiplyScalar(std::max(min, std::min(max, length)));
        }

        //If this vector's x, y or z values are greater than the max value, they are replaced by the max value.
        // If this vector's x, y or z values are less than the min value, they are replaced by the min value.
        Vector3& clampScalar(float minVal, float maxVal) {
            this->x = std::max(minVal, std::min(maxVal, this->x));
            this->y = std::max(minVal, std::min(maxVal, this->y));
            this->z = std::max(minVal, std::min(maxVal, this->z));

            return *this;
        }

        // The components of this vector are rounded down to the nearest integer value.
        Vector3& floor() {
            this->x = std::floor(this->x);
            this->y = std::floor(this->y);
            this->z = std::floor(this->z);

            return *this;
        }

        // The x, y and z components of this vector are rounded up to the nearest integer value.
        Vector3& ceil() {
            this->x = std::ceil(this->x);
            this->y = std::ceil(this->y);
            this->z = std::ceil(this->z);

            return *this;
        }

        Vector3& round() {
            this->x = std::round(this->x);
            this->y = std::round(this->y);
            this->z = std::round(this->z);

            return *this;
        }

        Vector3& roundToZero() {
            this->x = (x < 0) ? std::ceil(this->x) : std::floor(this->x);
            this->y = (y < 0) ? std::ceil(this->y) : std::floor(this->y);
            this->z = (z < 0) ? std::ceil(this->z) : std::floor(this->z);

            return *this;
        }

        // Inverts this vector - i.e. sets x = -x, y = -y and z = -z.
        Vector3& negate() {
            x = -x;
            y = -y;
            z = -z;

            return *this;
        }

        // Calculate the dot product of this vector and v.
        [[nodiscard]] float dot(const Vector3& v) const {
            return x * v.x + y * v.y + z * v.z;
        }

        // Computes the square of the Euclidean length (straight-line length) from (0, 0, 0) to (x, y, z). If you are comparing the lengths of vectors,
        // you should compare the length squared instead as it is slightly more efficient to calculate.
        [[nodiscard]] float lengthSq() const {
            return x * x + y * y + z * z;
        }

        // Computes the Euclidean length (straight-line length) from (0, 0, 0) to (x, y, z).
        [[nodiscard]] float length() const {
            return std::sqrt(x * x + y * y + z * z);
        }

        // Computes the Manhattan length of this vector.
        [[nodiscard]] float manhattanLength() const {
            return std::abs(x) + std::abs(y) + std::abs(z);
        }

        // Convert this vector to a unit vector - that is, sets it equal to a vector with the same direction as this one, but length 1.
        Vector3& normalize() {
            auto l = length();
            this->divideScalar(std::isnan(l) ? 1 : l);

            return *this;
        }

        // Set this vector to a vector with the same direction as this one, but length l.
        Vector3& setLength(float length) {
            return normalize().multiplyScalar(length);
        }

        // Linearly interpolate between this vector and v, where alpha is the percent distance along the line - alpha = 0 will be this vector, and alpha = 1 will be v.
        Vector3& lerp(const Vector3& v, float alpha) {
            this->x += (v.x - x) * alpha;
            this->y += (v.y - y) * alpha;
            this->z += (v.z - z) * alpha;

            return *this;
        }

        // Sets this vector to be the vector linearly interpolated between v1 and v2 where alpha is the percent
        // distance along the line connecting the two vectors - alpha = 0 will be v1, and alpha = 1 will be v2.
        Vector3& lerpVectors(const Vector3& v1, const Vector3& v2, float alpha) {
            this->x = v1.x + (v2.x - v1.x) * alpha;
            this->y = v1.y + (v2.y - v1.y) * alpha;
            this->z = v1.z + (v2.z - v1.z) * alpha;

            return *this;
        }

        // Sets this vector to cross product of itself and v.
        Vector3& cross(const Vector3& v) {
            return crossVectors(*this, v);
        }

        // Sets this vector to cross product of a and b.
        Vector3& crossVectors(const Vector3& a, const Vector3& b) {
            const auto ax = a.x, ay = a.y, az = a.z;
            const auto bx = b.x, by = b.y, bz = b.z;

            this->x = ay * bz - az * by;
            this->y = az * bx - ax * bz;
            this->z = ax * by - ay * bx;

            return *this;
        }

        Vector3& projectOnVector(const Vector3& v) {
            const auto denominator = v.lengthSq();

            if (denominator == 0) return this->set(0, 0, 0);

            const auto scalar = v.dot(*this) / denominator;

            return this->copy(v).multiplyScalar(scalar);
        }

        Vector3& projectOnPlane(const Vector3& planeNormal) {
            Vector3 _vector;
            _vector.copy(*this).projectOnVector(planeNormal);

            return this->sub(_vector);
        }

        Vector3& reflect(const Vector3& normal) {
            // reflect incident vector off plane orthogonal to normal
            // normal is assumed to have unit length

            Vector3 _vector;
            return this->sub(_vector.copy(normal).multiplyScalar(2 * this->dot(normal)));
        }

        // Returns the angle between this vector and vector v in radians.
        [[nodiscard]] float angleTo(const Vector3& v) const;

        // Computes the distance from this vector to v.
        [[nodiscard]] float distanceTo(const Vector3& v) const {
            return std::sqrt(distanceToSquared(v));
        }

        // Computes the squared distance from this vector to v. If you are just comparing the distance with another distance,
        // you should compare the distance squared instead as it is slightly more efficient to calculate.
        [[nodiscard]] float distanceToSquared(const Vector3& v) const {
            const auto dx = this->x - v.x, dy = this->y - v.y, dz = this->z - v.z;

            return dx * dx + dy * dy + dz * dz;
        }

        // Computes the Manhattan distance from this vector to v.
        [[nodiscard]] float manhattanDistanceTo(const Vector3& v) const {
            return std::abs(this->x - v.x) + std::abs(this->y - v.y) + std::abs(this->z - v.z);
        }

        // Sets this vector from the spherical coordinates s.
        Vector3& setFromSpherical(const Spherical& s);

        // Sets this vector from the spherical coordinates radius, phi and theta.
        Vector3& setFromSphericalCoords(float radius, float phi, float theta) {
            const auto sinPhiRadius = std::sin(phi) * radius;

            this->x = sinPhiRadius * std::sin(theta);
            this->y = std::cos(phi) * radius;
            this->z = sinPhiRadius * std::cos(theta);

            return *this;
        }

        // Sets this vector to the position elements of the transformation matrix m.
        Vector3& setFromMatrixPosition(const Matrix4& m) {
            const auto& e = m.elements;

            this->x = e[12];
            this->y = e[13];
            this->z = e[14];

            return *this;
        }

        // Sets this vector to the scale elements of the transformation matrix m.
        Vector3& setFromMatrixScale(const Matrix4& m) {
            const auto sx = this->setFromMatrixColumn(m, 0).length();
            const auto sy = this->setFromMatrixColumn(m, 1).length();
            const auto sz = this->setFromMatrixColumn(m, 2).length();

            this->x = sx;
            this->y = sy;
            this->z = sz;

            return *this;
        }

        // Sets this vector's x, y and z components from index column of matrix.
        Vector3& setFromMatrixColumn(const Matrix4& m, unsigned int index) {
            return this->fromArray(m.elements, index * 4);
        }

        // Sets this vector's x, y and z components from index column of matrix.
        Vector3& setFromMatrix3Column(const Matrix3& m, unsigned int index) {
            return this->fromArray(m.elements, index * 3);
        }

        [[nodiscard]] Vector3 clone() const {
            return Vector3{x, y, z};
        }

        [[nodiscard]] bool equals(const Vector3& v) const {
            return ((v.x == this->x) && (v.y == this->y) && (v.z == this->z));
        }

        bool operator==(const Vector3& other) const {
            return equals(other);
        }

        bool operator!=(const Vector3& other) const {
            return !equals(other);
        }

        Vector3 operator+(const Vector3& other) const {
            return clone().add(other);
        }

        Vector3& operator+=(const Vector3& other) {
            return add(other);
        }

        Vector3 operator+(float s) const {
            return clone().addScalar(s);
        }

        Vector3& operator+=(float s) {
            return addScalar(s);
        }

        Vector3 operator-(const Vector3& other) const {
            return clone().sub(other);
        }

        Vector3& operator-=(const Vector3& other) {
            return sub(other);
        }

        Vector3 operator-(float s) const {
            return clone().subScalar(s);
        }

        Vector3& operator-=(float s) {
            return subScalar(s);
        }

        Vector3 operator*(const Vector3& other) const {
            return clone().multiply(other);
        }

        Vector3& operator*=(const Vector3& other) {
            return multiply(other);
        }

        Vector3 operator*(float s) const {
            return clone().multiplyScalar(s);
        }

        Vector3& operator*=(float s) {
            return multiplyScalar(s);
        }

        Vector3 operator/(const Vector3& other) const {
            return clone().divide(other);
        }

        Vector3& operator/=(const Vector3& other) {
            return divide(other);
        }

        Vector3 operator/(float s) const {
            return clone().divideScalar(s);
        }

        Vector3& operator/=(float s) {
            return divideScalar(s);
        }

        Vector3& makeNan() {
            return set(NAN, NAN, NAN);
        }

        [[nodiscard]] bool isNan() const {
            return std::isnan(x) || std::isnan(y) || std::isnan(z);
        }

        template<class ArrayLike>
        Vector3& fromArray(const ArrayLike& array, size_t offset = 0) {

            this->x = array[offset];
            this->y = array[offset + 1];
            this->z = array[offset + 2];

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike& array, size_t offset = 0) const {

            array[offset] = this->x;
            array[offset + 1] = this->y;
            array[offset + 2] = this->z;
        }

        friend std::ostream& operator<<(std::ostream& os, const Vector3& v) {
            os << "Vector3(x=" << v.x << ", y=" << v.y << ", z=" << v.z << ")";
            return os;
        }

        static Vector3 X() {
            return {1, 0, 0};
        }

        static Vector3 Y() {
            return {0, 1, 0};
        }

        static Vector3 Z() {
            return {0, 0, 1};
        }

        static Vector3 ZEROS() {
            return {0, 0, 0};
        }

        static Vector3 ONES() {
            return {1, 1, 1};
        }
    };

    // Implementing get function template
    template<std::size_t N>
    auto get(const Vector3& p) {
        if constexpr (N == 0) return p.x;
        else if constexpr (N == 1)
            return p.y;
        else if constexpr (N == 2)
            return p.z;
    }

}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
