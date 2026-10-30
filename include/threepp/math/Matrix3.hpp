// https://github.com/mrdoob/three.js/blob/r129/src/math/Matrix3.js

#ifndef THREEPP_MATRIX3_HPP
#define THREEPP_MATRIX3_HPP

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace threepp {

    class Vector3;
    class Matrix4;

    class Matrix3 {

    public:
        std::array<float, 9> elements{
                1.f, 0.f, 0.f,
                0.f, 1.f, 0.f,
                0.f, 0.f, 1.f};

        float& operator[](unsigned int index) {
            if (index >= 9) throw std::runtime_error("index out of bounds: " + std::to_string(index));

            return elements[index];
        }

        Matrix3& set(float n11, float n12, float n13, float n21, float n22, float n23, float n31, float n32, float n33) {
            auto& te = this->elements;

            // clang-format off
            te[ 0 ] = n11; te[ 1 ] = n21; te[ 2 ] = n31;
            te[ 3 ] = n12; te[ 4 ] = n22; te[ 5 ] = n32;
            te[ 6 ] = n13; te[ 7 ] = n23; te[ 8 ] = n33;
            // clang-format on

            return *this;
        }

        Matrix3& identity() {
            this->set(

                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 1

            );

            return *this;
        }

        Matrix3& copy(const Matrix3& m) {
            auto& te = this->elements;
            const auto& me = m.elements;

            // clang-format off
            te[ 0 ] = me[ 0 ]; te[ 1 ] = me[ 1 ]; te[ 2 ] = me[ 2 ];
            te[ 3 ] = me[ 3 ]; te[ 4 ] = me[ 4 ]; te[ 5 ] = me[ 5 ];
            te[ 6 ] = me[ 6 ]; te[ 7 ] = me[ 7 ]; te[ 8 ] = me[ 8 ];
            // clang-format on

            return *this;
        }

        Matrix3& extractBasis(Vector3& xAxis, Vector3& yAxis, Vector3& zAxis);

        Matrix3& setFromMatrix4(const Matrix4& m);

        Matrix3& multiply(const Matrix3& m) {
            return this->multiplyMatrices(*this, m);
        }

        Matrix3& premultiply(const Matrix3& m) {
            return this->multiplyMatrices(m, *this);
        }

        Matrix3& multiplyMatrices(const Matrix3& a, const Matrix3& b);

        Matrix3& multiplyScalar(float s) {
            auto& te = this->elements;

            // clang-format off
            te[ 0 ] *= s; te[ 3 ] *= s; te[ 6 ] *= s;
            te[ 1 ] *= s; te[ 4 ] *= s; te[ 7 ] *= s;
            te[ 2 ] *= s; te[ 5 ] *= s; te[ 8 ] *= s;
            // clang-format on

            return *this;
        }

        [[nodiscard]] float determinant() const;

        Matrix3& invert();

        Matrix3& transpose() {
            float tmp;
            auto& m = this->elements;

            // clang-format off
            tmp = m[ 1 ]; m[ 1 ] = m[ 3 ]; m[ 3 ] = tmp;
            tmp = m[ 2 ]; m[ 2 ] = m[ 6 ]; m[ 6 ] = tmp;
            tmp = m[ 5 ]; m[ 5 ] = m[ 7 ]; m[ 7 ] = tmp;
            // clang-format on

            return *this;
        }

        Matrix3& getNormalMatrix(const Matrix4& m);

        template<class ArrayLike>
        Matrix3& transposeIntoArray(ArrayLike& r) {

            const auto& m = this->elements;

            r[0] = m[0];
            r[1] = m[3];
            r[2] = m[6];
            r[3] = m[1];
            r[4] = m[4];
            r[5] = m[7];
            r[6] = m[2];
            r[7] = m[5];
            r[8] = m[8];

            return *this;
        }

        Matrix3& setUvTransform(float tx, float ty, float sx, float sy, float rotation, float cx, float cy);

        Matrix3& scale(float sx, float sy) {
            auto& te = this->elements;

            // clang-format off
            te[ 0 ] *= sx; te[ 3 ] *= sx; te[ 6 ] *= sx;
            te[ 1 ] *= sy; te[ 4 ] *= sy; te[ 7 ] *= sy;
            // clang-format on

            return *this;
        }

        Matrix3& rotate(float theta) {
            const float c = std::cos(theta);
            const float s = std::sin(theta);

            auto& te = this->elements;

            const float a11 = te[0], a12 = te[3], a13 = te[6];
            const float a21 = te[1], a22 = te[4], a23 = te[7];

            te[0] = c * a11 + s * a21;
            te[3] = c * a12 + s * a22;
            te[6] = c * a13 + s * a23;

            te[1] = -s * a11 + c * a21;
            te[4] = -s * a12 + c * a22;
            te[7] = -s * a13 + c * a23;

            return *this;
        }

        Matrix3& translate(float tx, float ty) {
            auto& te = this->elements;

            te[0] += tx * te[2];
            te[3] += tx * te[5];
            te[6] += tx * te[8];
            te[1] += ty * te[2];
            te[4] += ty * te[5];
            te[7] += ty * te[8];

            return *this;
        }

        [[nodiscard]] bool equals(const Matrix3& matrix) const {
            const auto& te = this->elements;
            const auto& me = matrix.elements;

            for (int i = 0; i < 9; i++) {

                if (te[i] != me[i]) return false;
            }

            return true;
        }

        bool operator==(const Matrix3& matrix) const {
            return equals(matrix);
        }

        template<class ArrayLike>
        Matrix3& fromArray(ArrayLike& array, unsigned int offset = 0) {

            for (auto i = 0; i < 9; i++) {

                this->elements[i] = array[i + offset];
            }

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike& array, unsigned int offset = 0) const {

            const auto& te = this->elements;

            array[offset] = te[0];
            array[offset + 1] = te[1];
            array[offset + 2] = te[2];

            array[offset + 3] = te[3];
            array[offset + 4] = te[4];
            array[offset + 5] = te[5];

            array[offset + 6] = te[6];
            array[offset + 7] = te[7];
            array[offset + 8] = te[8];
        }
    };

}// namespace threepp

#endif//THREEPP_MATRIX3_HPP
