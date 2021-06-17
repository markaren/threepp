// https://github.com/mrdoob/three.js/blob/r129/src/math/Matrix3.js

#ifndef THREEPP_MATRIX3_HPP
#define THREEPP_MATRIX3_HPP

#include <array>

namespace threepp {

    class Vector3;
    class Matrix4;

    class Matrix3 {

    public:
        Matrix3() = default;

        [[nodiscard]] const auto &elements() const {
            return elements_;
        }

        float &operator[](unsigned int index);

        Matrix3 &set(float n11, float n12, float n13, float n21, float n22, float n23, float n31, float n32, float n33);

        Matrix3 &identity();

        Matrix3 &copy(const Matrix3 &m);

        Matrix3 &extractBasis(Vector3 &xAxis, Vector3 &yAxis, Vector3 &zAxis);

        Matrix3 &setFromMatrix4(const Matrix4 &m);

        Matrix3 &multiply(const Matrix3 &m);

        Matrix3 &premultiply(const Matrix3 &m);

        Matrix3 &multiplyMatrices(const Matrix3 &a, const Matrix3 &b);

        Matrix3 &multiplyScalar(float s);

        [[nodiscard]] float determinant() const;

        Matrix3 &invert();

        Matrix3 &transpose();

        Matrix3 &getNormalMatrix(const Matrix4 &m);

        template<class ArrayLike>
        Matrix3 &transposeIntoArray(ArrayLike &r) {

            const auto &m = this->elements_;

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

        Matrix3 &setUvTransform(float tx, float ty, float sx, float sy, float rotation, float cx, float cy);

        Matrix3 &scale(float sx, float sy);

        Matrix3 &rotate(float theta);

        Matrix3 &translate(float tx, float ty);

        template<class ArrayLike>
        Matrix3 &fromArray(ArrayLike &array, unsigned int offset = 0) {

            for (auto i = 0; i < 9; i++) {

                this->elements_[i] = array[i + offset];
            }

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            const auto &te = this->elements_;

            array[offset] = te[0];
            array[offset + 1] = te[1];
            array[offset + 2] = te[2];

            array[offset + 3] = te[3];
            array[offset + 4] = te[4];
            array[offset + 5] = te[5];

            array[offset + 6] = te[6];
            array[offset + 7] = te[7];
            array[offset + 8] = te[8];

            return array;
        }

    private:
        std::array<float, 16> elements_{
                1, 0, 0,
                0, 1, 0,
                0, 0, 1};
    };


}// namespace threepp

#endif//THREEPP_MATRIX3_HPP
