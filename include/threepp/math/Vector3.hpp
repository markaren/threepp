// https://github.com/mrdoob/three.js/blob/r129/src/math/Vector3.js

#ifndef THREEPP_VECTOR3_HPP
#define THREEPP_VECTOR3_HPP

#include <ostream>

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

        Vector3();
        Vector3(float x, float y, float z);

        // Sets the x, y and z components of this vector.
        Vector3& set(float x, float y, float z);

        // Set the x, y and z values of this vector both equal to scalar.
        Vector3& setScalar(float value);

        Vector3& setX(float value);

        Vector3& setY(float value);

        Vector3& setZ(float value);

        float& operator[](size_t index);

        Vector3& copy(const Vector3& v);

        // Adds v to this vector.
        Vector3& add(const Vector3& v);

        // Adds the scalar value s to this vector's x, y and z values.
        Vector3& addScalar(float s);

        // Sets this vector to a + b.
        Vector3& addVectors(const Vector3& a, const Vector3& b);

        // Adds the multiple of v and s to this vector.
        Vector3& addScaledVector(const Vector3& v, float s);

        // Subtracts v from this vector.
        Vector3& sub(const Vector3& v);

        // Subtracts s from this vector's x, y and z components.
        Vector3& subScalar(float s);

        // Sets this vector to a - b.
        Vector3& subVectors(const Vector3& a, const Vector3& b);

        // Multiplies this vector by v.
        Vector3& multiply(const Vector3& v);

        // Multiplies this vector by scalar s.
        Vector3& multiplyScalar(float scalar);

        // Sets this vector equal to a * b, component-wise.
        Vector3& multiplyVectors(const Vector3& a, const Vector3& b);

        // Applies a rotation specified by an axis and an angle to this vector.
        Vector3& applyAxisAngle(const Vector3& axis, float angle);

        // Applies euler transform to this vector by converting the Euler object to a Quaternion and applying.
        Vector3& applyEuler(const Euler& euler);

        // Multiplies this vector by m
        Vector3& applyMatrix3(const Matrix3& m);

        // Multiplies this vector by normal matrix m and normalizes the result.
        Vector3& applyNormalMatrix(const Matrix3& m);

        // Multiplies this vector (with an implicit 1 in the 4th dimension) by m, and divides by perspective.
        Vector3& applyMatrix4(const Matrix4& m);

        // Applies a Quaternion transform to this vector.
        Vector3& applyQuaternion(const Quaternion& q);

        // Projects this vector from world space into the camera's normalized device coordinate (NDC) space.
        Vector3& project(const Camera& camera);

        // Projects this vector from the camera's normalized device coordinate (NDC) space into world space.
        Vector3& unproject(const Camera& camera);

        // Transforms the direction of this vector by a matrix (the upper left 3 x 3 subset of a m) and then normalizes the result.
        Vector3& transformDirection(const Matrix4& m);

        // Divides this vector by v.
        Vector3& divide(const Vector3& v);

        // Divides this vector by scalar s.
        Vector3& divideScalar(float v);

        Vector3& min(const Vector3& v);

        Vector3& max(const Vector3& v);

        // If this vector's x, y or z value is greater than the max vector's x, y or z value, it is replaced by the corresponding value.
        //If this vector's x, y or z value is less than the min vector's x, y or z value, it is replaced by the corresponding value.
        Vector3& clamp(const Vector3& min, const Vector3& max);

        // The components of this vector are rounded down to the nearest integer value.
        Vector3& floor();

        // The x, y and z components of this vector are rounded up to the nearest integer value.
        Vector3& ceil();

        Vector3& round();

        Vector3& roundToZero();

        // Inverts this vector - i.e. sets x = -x, y = -y and z = -z.
        Vector3& negate();

        // Calculate the dot product of this vector and v.
        [[nodiscard]] float dot(const Vector3& v) const;

        // Computes the square of the Euclidean length (straight-line length) from (0, 0, 0) to (x, y, z). If you are comparing the lengths of vectors,
        // you should compare the length squared instead as it is slightly more efficient to calculate.
        [[nodiscard]] float lengthSq() const;

        // Computes the Euclidean length (straight-line length) from (0, 0, 0) to (x, y, z).
        [[nodiscard]] float length() const;

        // Computes the Manhattan length of this vector.
        [[nodiscard]] float manhattanLength() const;

        // Convert this vector to a unit vector - that is, sets it equal to a vector with the same direction as this one, but length 1.
        Vector3& normalize();

        // Set this vector to a vector with the same direction as this one, but length l.
        Vector3& setLength(float length);

        // Linearly interpolate between this vector and v, where alpha is the percent distance along the line - alpha = 0 will be this vector, and alpha = 1 will be v.
        Vector3& lerp(const Vector3& v, float alpha);

        // Sets this vector to be the vector linearly interpolated between v1 and v2 where alpha is the percent
        // distance along the line connecting the two vectors - alpha = 0 will be v1, and alpha = 1 will be v2.
        Vector3& lerpVectors(const Vector3& v1, const Vector3& v2, float alpha);

        // Sets this vector to cross product of itself and v.
        Vector3& cross(const Vector3& v);

        // Sets this vector to cross product of a and b.
        Vector3& crossVectors(const Vector3& a, const Vector3& b);

        Vector3& projectOnVector(const Vector3& v);

        Vector3& projectOnPlane(const Vector3& planeNormal);

        Vector3& reflect(const Vector3& normal);

        // Returns the angle between this vector and vector v in radians.
        [[nodiscard]] float angleTo(const Vector3& v) const;

        // Computes the distance from this vector to v.
        [[nodiscard]] float distanceTo(const Vector3& v) const;

        // Computes the squared distance from this vector to v. If you are just comparing the distance with another distance,
        // you should compare the distance squared instead as it is slightly more efficient to calculate.
        [[nodiscard]] float distanceToSquared(const Vector3& v) const;

        // Computes the Manhattan distance from this vector to v.
        [[nodiscard]] float manhattanDistanceTo(const Vector3& v) const;

        // Sets this vector from the spherical coordinates s.
        Vector3& setFromSpherical(const Spherical& s);

        // Sets this vector from the spherical coordinates radius, phi and theta.
        Vector3& setFromSphericalCoords(float radius, float phi, float theta);

        // Sets this vector to the position elements of the transformation matrix m.
        Vector3& setFromMatrixPosition(const Matrix4& m);

        // Sets this vector to the scale elements of the transformation matrix m.
        Vector3& setFromMatrixScale(const Matrix4& m);

        // Sets this vector's x, y and z components from index column of matrix.
        Vector3& setFromMatrixColumn(const Matrix4& m, unsigned int index);

        // Sets this vector's x, y and z components from index column of matrix.
        Vector3& setFromMatrix3Column(const Matrix3& m, unsigned int index);

        [[nodiscard]] Vector3 clone() const;

        [[nodiscard]] bool equals(const Vector3& v) const;

        bool operator==(const Vector3& other) const;

        bool operator!=(const Vector3& other) const;

        Vector3 operator+(const Vector3& other) const;

        Vector3& operator+=(const Vector3& other);

        Vector3 operator+(float s) const;

        Vector3& operator+=(float s);

        Vector3 operator-(const Vector3& other) const;

        Vector3& operator-=(const Vector3& other);

        Vector3 operator-(float s) const;

        Vector3& operator-=(float s);

        Vector3 operator*(const Vector3& other) const;

        Vector3& operator*=(const Vector3& other);

        Vector3 operator*(float s) const;

        Vector3& operator*=(float s);

        Vector3 operator/(const Vector3& other) const;

        Vector3& operator/=(const Vector3& other);

        Vector3 operator/(float s) const;

        Vector3& operator/=(float s);

        Vector3& makeNan();

        [[nodiscard]] bool isNan() const;

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

        inline static Vector3 X() {
            return {1, 0, 0};
        }

        inline static Vector3 Y() {
            return {0, 1, 0};
        }

        inline static Vector3 Z() {
            return {0, 0, 1};
        }

        inline static Vector3 ZEROS() {
            return {0, 0, 0};
        }

        inline static Vector3 ONES() {
            return {1, 1, 1};
        }
    };


}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
