// https://github.com/mrdoob/three.js/blob/r129/src/math/Matrix4.js

#ifndef THREEPP_MATRIX4_HPP
#define THREEPP_MATRIX4_HPP

#include <array>
#include <ostream>

namespace threepp {

    class Vector3;
    class Euler;
    class Quaternion;
    class Matrix3;

    class Matrix4 {

    public:
        std::array<float, 16> elements{
                1.f, 0.f, 0.f, 0.f,
                0.f, 1.f, 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f,
                0.f, 0.f, 0.f, 1.f};

        Matrix4() = default;

        float& operator[](unsigned int index);

        Matrix4& set(float n11, float n12, float n13, float n14, float n21, float n22, float n23, float n24, float n31, float n32, float n33, float n34, float n41, float n42, float n43, float n44);

        Matrix4& identity();

        Matrix4& copy(const Matrix4& m);

        Matrix4& copyPosition(const Matrix4& m);

        Matrix4& setFromMatrix3(const Matrix3& m);

        Matrix4& extractBasis(Vector3& xAxis, Vector3& yAxis, Vector3& zAxis);

        Matrix4& makeBasis(const Vector3& xAxis, const Vector3& yAxis, const Vector3& zAxis);

        Matrix4& extractRotation(const Matrix4& m);

        Matrix4& makeRotationFromEuler(const Euler& e);

        Matrix4& makeRotationFromQuaternion(const Quaternion& q);

        Matrix4& lookAt(const Vector3& eye, const Vector3& target, const Vector3& up);

        Matrix4& multiply(const Matrix4& m);

        Matrix4& premultiply(const Matrix4& m);

        Matrix4& multiplyMatrices(const Matrix4& a, const Matrix4& b);

        Matrix4& multiplyScalar(float s);

        [[nodiscard]] float determinant() const;

        Matrix4& transpose();

        Matrix4& setPosition(const Vector3& v);

        Matrix4& setPosition(float x, float y, float z);

        Matrix4& invert();

        Matrix4& scale(const Vector3& v);

        [[nodiscard]] float getMaxScaleOnAxis() const;

        Matrix4& makeTranslation(float x, float y, float z);

        Matrix4& makeTranslation(const Vector3& v);

        Matrix4& makeRotationX(float theta);

        Matrix4& makeRotationY(float theta);

        Matrix4& makeRotationZ(float theta);

        Matrix4& makeRotationAxis(const Vector3& axis, float angle);

        Matrix4& makeScale(float x, float y, float z);

        Matrix4& makeShear(float xy, float xz, float yx, float yz, float zx, float zy);

        Matrix4& compose(const Vector3& position, const Quaternion& quaternion, const Vector3& scale);

        Matrix4& decompose(Vector3& position, Quaternion& quaternion, Vector3& scale);

        Matrix4& makePerspective(float left, float right, float top, float bottom, float near, float far);

        Matrix4& makeOrthographic(float left, float right, float top, float bottom, float near, float far);

        [[nodiscard]] bool equals(const Matrix4& matrix) const;

        bool operator==(const Matrix4& matrix) const;

        bool operator!=(const Matrix4& matrix) const;

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
