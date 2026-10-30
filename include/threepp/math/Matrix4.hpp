// https://github.com/mrdoob/three.js/blob/r129/src/math/Matrix4.js

#ifndef THREEPP_MATRIX4_HPP
#define THREEPP_MATRIX4_HPP

#include <array>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>

namespace threepp {

    class Vector3;
    class Euler;
    class Quaternion;
    class Matrix3;

    // A class representing a 4x4 matrix.
    class Matrix4 {

    public:
        std::array<float, 16> elements{
                1.f, 0.f, 0.f, 0.f,
                0.f, 1.f, 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f,
                0.f, 0.f, 0.f, 1.f};

        Matrix4() = default;
        Matrix4(const std::array<float, 16>& elements)
            : elements(elements) {}

        Matrix4(const Matrix4&) = default;

        operator std::array<float, 16>() const {
            return elements;
        }

        float& operator[](unsigned int index) {
            if (index >= 16) throw std::runtime_error("index out of bounds: " + std::to_string(index));

            return elements[index];
        }

        const float& operator[](unsigned int index) const {
            if (index >= 16) throw std::runtime_error("index out of bounds: " + std::to_string(index));

            return elements[index];
        }

        // Set the elements of this matrix to the supplied row-major values n11, n12, ... n44.
        Matrix4& set(float n11, float n12, float n13, float n14, float n21, float n22, float n23, float n24, float n31, float n32, float n33, float n34, float n41, float n42, float n43, float n44) {
            auto& te = this->elements;

            // clang-format off
            te[ 0 ] = n11; te[ 4 ] = n12; te[ 8 ] = n13; te[ 12 ] = n14;
            te[ 1 ] = n21; te[ 5 ] = n22; te[ 9 ] = n23; te[ 13 ] = n24;
            te[ 2 ] = n31; te[ 6 ] = n32; te[ 10 ] = n33; te[ 14 ] = n34;
            te[ 3 ] = n41; te[ 7 ] = n42; te[ 11 ] = n43; te[ 15 ] = n44;
            // clang-format on

            return *this;
        }

        // Resets this matrix to the identity matrix.
        Matrix4& identity() {
            this->set(

                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1

            );

            return *this;
        }

        // Copies the elements of matrix m into this matrix.
        Matrix4& copy(const Matrix4& m) {
            auto& te = this->elements;
            const auto& me = m.elements;

            // clang-format off
            te[ 0 ] = me[ 0 ]; te[ 1 ] = me[ 1 ]; te[ 2 ] = me[ 2 ]; te[ 3 ] = me[ 3 ];
            te[ 4 ] = me[ 4 ]; te[ 5 ] = me[ 5 ]; te[ 6 ] = me[ 6 ]; te[ 7 ] = me[ 7 ];
            te[ 8 ] = me[ 8 ]; te[ 9 ] = me[ 9 ]; te[ 10 ] = me[ 10 ]; te[ 11 ] = me[ 11 ];
            te[ 12 ] = me[ 12 ]; te[ 13 ] = me[ 13 ]; te[ 14 ] = me[ 14 ]; te[ 15 ] = me[ 15 ];
            // clang-format on

            return *this;
        }

        // Copies the translation component of the supplied matrix m into this matrix's translation component.
        Matrix4& copyPosition(const Matrix4& m) {
            auto& te = this->elements;
            const auto& me = m.elements;

            te[12] = me[12];
            te[13] = me[13];
            te[14] = me[14];

            return *this;
        }

        // Set the upper 3x3 elements of this matrix to the values of the Matrix3 m.
        Matrix4& setFromMatrix3(const Matrix3& m);

        Matrix4& extractBasis(Vector3& xAxis, Vector3& yAxis, Vector3& zAxis);

        Matrix4& makeBasis(const Vector3& xAxis, const Vector3& yAxis, const Vector3& zAxis);

        Matrix4& extractRotation(const Matrix4& m);

        Matrix4& makeRotationFromEuler(const Euler& e);

        Matrix4& makeRotationFromQuaternion(const Quaternion& q);

        // Constructs a rotation matrix, looking from eye towards target oriented by the up vector.
        Matrix4& lookAt(const Vector3& eye, const Vector3& target, const Vector3& up);

        // Post-multiplies this matrix by m.
        Matrix4& multiply(const Matrix4& m) {
            return this->multiplyMatrices(*this, m);
        }

        // Pre-multiplies this matrix by m.
        Matrix4& premultiply(const Matrix4& m) {
            return this->multiplyMatrices(m, *this);
        }

        // Sets this matrix to a x b.
        Matrix4& multiplyMatrices(const Matrix4& a, const Matrix4& b);

        // Multiplies every component of the matrix by a scalar value s.
        Matrix4& multiplyScalar(float s) {
            auto& te = this->elements;

            // clang-format off
            te[ 0 ] *= s; te[ 4 ] *= s; te[ 8 ] *= s; te[ 12 ] *= s;
            te[ 1 ] *= s; te[ 5 ] *= s; te[ 9 ] *= s; te[ 13 ] *= s;
            te[ 2 ] *= s; te[ 6 ] *= s; te[ 10 ] *= s; te[ 14 ] *= s;
            te[ 3 ] *= s; te[ 7 ] *= s; te[ 11 ] *= s; te[ 15 ] *= s;
            // clang-format on

            return *this;
        }

        // Computes and returns the determinant of this matrix.
        [[nodiscard]] float determinant() const;

        Matrix4& transpose() {
            auto& te = this->elements;
            float tmp;

            // clang-format off
            tmp = te[ 1 ]; te[ 1 ] = te[ 4 ]; te[ 4 ] = tmp;
            tmp = te[ 2 ]; te[ 2 ] = te[ 8 ]; te[ 8 ] = tmp;
            tmp = te[ 6 ]; te[ 6 ] = te[ 9 ]; te[ 9 ] = tmp;

            tmp = te[ 3 ]; te[ 3 ] = te[ 12 ]; te[ 12 ] = tmp;
            tmp = te[ 7 ]; te[ 7 ] = te[ 13 ]; te[ 13 ] = tmp;
            tmp = te[ 11 ]; te[ 11 ] = te[ 14 ]; te[ 14 ] = tmp;
            // clang-format on

            return *this;
        }

        // Sets the position component for this matrix from vector v, without affecting the rest of the matrix.
        Matrix4& setPosition(const Vector3& v);

        // Sets the position component for this matrix from x, y, z, without affecting the rest of the matrix.
        Matrix4& setPosition(float x, float y, float z) {
            auto& te = this->elements;

            te[12] = x;
            te[13] = y;
            te[14] = z;


            return *this;
        }

        // Inverts this matrix, using the analytic method. You can not invert with a determinant of zero. If you attempt this, the method produces a zero matrix instead.
        Matrix4& invert();

        // Multiplies the columns of this matrix by vector v.
        Matrix4& scale(const Vector3& v);

        // Gets the maximum scale value of the 3 axes.
        [[nodiscard]] float getMaxScaleOnAxis() const;

        // Sets this matrix as a translation transform from the numbers x, y and z,
        Matrix4& makeTranslation(float x, float y, float z) {
            this->set(

                    1, 0, 0, x,
                    0, 1, 0, y,
                    0, 0, 1, z,
                    0, 0, 0, 1

            );

            return *this;
        }

        // Sets this matrix as a translation transform from vector the v,
        Matrix4& makeTranslation(const Vector3& v);

        // Sets this matrix as a rotational transformation around the X axis by theta (θ) radians.
        Matrix4& makeRotationX(float theta) {
            const float c = std::cos(theta), s = std::sin(theta);

            this->set(

                    1, 0, 0, 0,
                    0, c, -s, 0,
                    0, s, c, 0,
                    0, 0, 0, 1

            );

            return *this;
        }

        // Sets this matrix as a rotational transformation around the Y axis by theta (θ) radians.
        Matrix4& makeRotationY(float theta) {
            const float c = std::cos(theta), s = std::sin(theta);

            this->set(

                    c, 0, s, 0,
                    0, 1, 0, 0,
                    -s, 0, c, 0,
                    0, 0, 0, 1

            );

            return *this;
        }

        // Sets this matrix as a rotational transformation around the Z axis by theta (θ) radians.
        Matrix4& makeRotationZ(float theta) {
            const float c = std::cos(theta), s = std::sin(theta);

            this->set(

                    c, -s, 0, 0,
                    s, c, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1

            );

            return *this;
        }

        Matrix4& makeRotationAxis(const Vector3& axis, float angle);

        Matrix4& makeScale(float x, float y, float z) {
            this->set(

                    x, 0, 0, 0,
                    0, y, 0, 0,
                    0, 0, z, 0,
                    0, 0, 0, 1

            );

            return *this;
        }

        Matrix4& makeShear(float xy, float xz, float yx, float yz, float zx, float zy) {
            this->set(

                    1, yx, zx, 0,
                    xy, 1, zy, 0,
                    xz, yz, 1, 0,
                    0, 0, 0, 1

            );

            return *this;
        }

        // Sets this matrix to the transformation composed of position, quaternion and scale.
        Matrix4& compose(const Vector3& position, const Quaternion& quaternion, const Vector3& scale);

        void decompose(Vector3& position, Quaternion& quaternion, Vector3& scale) const;

        Matrix4& makePerspective(float left, float right, float top, float bottom, float near, float far);

        Matrix4& makeOrthographic(float left, float right, float top, float bottom, float near, float far);

        [[nodiscard]] bool equals(const Matrix4& matrix) const {
            const auto& te = this->elements;
            const auto& me = matrix.elements;

            for (int i = 0; i < 16; i++) {

                if (te[i] != me[i]) return false;
            }

            return true;
        }

        bool operator==(const Matrix4& matrix) const {
            return equals(matrix);
        }

        bool operator!=(const Matrix4& matrix) const {
            return !equals(matrix);
        }

        template<class ArrayLike>
        Matrix4& fromArray(const ArrayLike& array, size_t offset = 0) {

            for (auto i = 0; i < 16; i++) {

                this->elements[i] = array[i + offset];
            }

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike& array, size_t offset = 0) const {

            auto& te = this->elements;

            array[offset] = te[0];
            array[offset + 1] = te[1];
            array[offset + 2] = te[2];
            array[offset + 3] = te[3];

            array[offset + 4] = te[4];
            array[offset + 5] = te[5];
            array[offset + 6] = te[6];
            array[offset + 7] = te[7];

            array[offset + 8] = te[8];
            array[offset + 9] = te[9];
            array[offset + 10] = te[10];
            array[offset + 11] = te[11];

            array[offset + 12] = te[12];
            array[offset + 13] = te[13];
            array[offset + 14] = te[14];
            array[offset + 15] = te[15];
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix4& v) {
            // clang-format off
            os << "Matrix4(\n" <<
                    v.elements[0] << ", " << v.elements[1] << ", " << v.elements[2] << ", " << v.elements[3] << "\n" <<
                    v.elements[4] << ", " << v.elements[5] << ", " << v.elements[6] << ", " << v.elements[7] << "\n" <<
                    v.elements[8] << ", " << v.elements[9] << ", " << v.elements[10] << ", " << v.elements[11] << "\n" <<
                    v.elements[12] << ", " << v.elements[13] << ", " << v.elements[14] << ", " << v.elements[15] << "\n" <<
                    ")";
            // clang-format on
            return os;
        }
    };

}// namespace threepp

#endif//THREEPP_MATRIX4_HPP
