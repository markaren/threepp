
#include "threepp/math/Matrix3.hpp"

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace threepp;

float& Matrix3::operator[](unsigned int index) {

    if (index >= 9) throw std::runtime_error("index out of bounds: " + std::to_string(index));

    return elements[index];
}

Matrix3& Matrix3::set(float n11, float n12, float n13, float n21, float n22, float n23, float n31, float n32, float n33) {

    auto& te = this->elements;

    // clang-format off
    te[ 0 ] = n11; te[ 1 ] = n21; te[ 2 ] = n31;
    te[ 3 ] = n12; te[ 4 ] = n22; te[ 5 ] = n32;
    te[ 6 ] = n13; te[ 7 ] = n23; te[ 8 ] = n33;
    // clang-format on

    return *this;
}

Matrix3& Matrix3::identity() {

    this->set(

            1, 0, 0,
            0, 1, 0,
            0, 0, 1

    );

    return *this;
}

Matrix3& Matrix3::copy(const Matrix3& m) {

    auto& te = this->elements;
    const auto& me = m.elements;

    // clang-format off
    te[ 0 ] = me[ 0 ]; te[ 1 ] = me[ 1 ]; te[ 2 ] = me[ 2 ];
    te[ 3 ] = me[ 3 ]; te[ 4 ] = me[ 4 ]; te[ 5 ] = me[ 5 ];
    te[ 6 ] = me[ 6 ]; te[ 7 ] = me[ 7 ]; te[ 8 ] = me[ 8 ];
    // clang-format on

    return *this;
}

Matrix3& Matrix3::extractBasis(Vector3& xAxis, Vector3& yAxis, Vector3& zAxis) {

    xAxis.setFromMatrix3Column(*this, 0);
    yAxis.setFromMatrix3Column(*this, 1);
    zAxis.setFromMatrix3Column(*this, 2);

    return *this;
}

Matrix3& Matrix3::setFromMatrix4(const Matrix4& m) {

    auto& me = m.elements;

    this->set(

            me[0], me[4], me[8],
            me[1], me[5], me[9],
            me[2], me[6], me[10]

    );

    return *this;
}

Matrix3& Matrix3::multiply(const Matrix3& m) {

    return this->multiplyMatrices(*this, m);
}

Matrix3& Matrix3::premultiply(const Matrix3& m) {

    return this->multiplyMatrices(m, *this);
}

Matrix3& Matrix3::multiplyMatrices(const Matrix3& a, const Matrix3& b) {

    const auto& ae = a.elements;
    const auto& be = b.elements;
    auto& te = this->elements;

    const auto a11 = ae[0], a12 = ae[3], a13 = ae[6];
    const auto a21 = ae[1], a22 = ae[4], a23 = ae[7];
    const auto a31 = ae[2], a32 = ae[5], a33 = ae[8];

    const auto b11 = be[0], b12 = be[3], b13 = be[6];
    const auto b21 = be[1], b22 = be[4], b23 = be[7];
    const auto b31 = be[2], b32 = be[5], b33 = be[8];

    te[0] = a11 * b11 + a12 * b21 + a13 * b31;
    te[3] = a11 * b12 + a12 * b22 + a13 * b32;
    te[6] = a11 * b13 + a12 * b23 + a13 * b33;

    te[1] = a21 * b11 + a22 * b21 + a23 * b31;
    te[4] = a21 * b12 + a22 * b22 + a23 * b32;
    te[7] = a21 * b13 + a22 * b23 + a23 * b33;

    te[2] = a31 * b11 + a32 * b21 + a33 * b31;
    te[5] = a31 * b12 + a32 * b22 + a33 * b32;
    te[8] = a31 * b13 + a32 * b23 + a33 * b33;

    return *this;
}

Matrix3& Matrix3::multiplyScalar(float s) {

    auto& te = this->elements;

    // clang-format off
    te[ 0 ] *= s; te[ 3 ] *= s; te[ 6 ] *= s;
    te[ 1 ] *= s; te[ 4 ] *= s; te[ 7 ] *= s;
    te[ 2 ] *= s; te[ 5 ] *= s; te[ 8 ] *= s;
    // clang-format on

    return *this;
}

float Matrix3::determinant() const {

    auto& te = this->elements;

    const auto a = te[0], b = te[1], c = te[2],
               d = te[3], e = te[4], f = te[5],
               g = te[6], h = te[7], i = te[8];

    return a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g;
}

Matrix3& Matrix3::invert() {

    auto& te = this->elements;

    const auto n11 = te[0], n21 = te[1], n31 = te[2],
               n12 = te[3], n22 = te[4], n32 = te[5],
               n13 = te[6], n23 = te[7], n33 = te[8],

               t11 = n33 * n22 - n32 * n23,
               t12 = n32 * n13 - n33 * n12,
               t13 = n23 * n12 - n22 * n13,

               det = n11 * t11 + n21 * t12 + n31 * t13;

    if (det == 0) {
        return this->set(0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    const auto detInv = 1.0f / det;

    te[0] = t11 * detInv;
    te[1] = (n31 * n23 - n33 * n21) * detInv;
    te[2] = (n32 * n21 - n31 * n22) * detInv;

    te[3] = t12 * detInv;
    te[4] = (n33 * n11 - n31 * n13) * detInv;
    te[5] = (n31 * n12 - n32 * n11) * detInv;

    te[6] = t13 * detInv;
    te[7] = (n21 * n13 - n23 * n11) * detInv;
    te[8] = (n22 * n11 - n21 * n12) * detInv;

    return *this;
}

Matrix3& Matrix3::transpose() {

    float tmp;
    auto& m = this->elements;

    // clang-format off
    tmp = m[ 1 ]; m[ 1 ] = m[ 3 ]; m[ 3 ] = tmp;
    tmp = m[ 2 ]; m[ 2 ] = m[ 6 ]; m[ 6 ] = tmp;
    tmp = m[ 5 ]; m[ 5 ] = m[ 7 ]; m[ 7 ] = tmp;
    // clang-format on

    return *this;
}

Matrix3& Matrix3::getNormalMatrix(const Matrix4& m) {

    return this->setFromMatrix4(m).invert().transpose();
}

Matrix3& Matrix3::setUvTransform(float tx, float ty, float sx, float sy, float rotation, float cx, float cy) {

    const float c = std::cos(rotation);
    const float s = std::sin(rotation);

    this->set(
            sx * c, sx * s, -sx * (c * cx + s * cy) + cx + tx,
            -sy * s, sy * c, -sy * (-s * cx + c * cy) + cy + ty,
            0, 0, 1);

    return *this;
}

Matrix3& Matrix3::scale(float sx, float sy) {

    auto& te = this->elements;

    // clang-format off
    te[ 0 ] *= sx; te[ 3 ] *= sx; te[ 6 ] *= sx;
    te[ 1 ] *= sy; te[ 4 ] *= sy; te[ 7 ] *= sy;
    // clang-format on

    return *this;
}

Matrix3& Matrix3::rotate(float theta) {

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

Matrix3& Matrix3::translate(float tx, float ty) {

    auto& te = this->elements;

    te[0] += tx * te[2];
    te[3] += tx * te[5];
    te[6] += tx * te[8];
    te[1] += ty * te[2];
    te[4] += ty * te[5];
    te[7] += ty * te[8];

    return *this;
}

bool Matrix3::equals(const Matrix3& matrix) const {

    const auto& te = this->elements;
    const auto& me = matrix.elements;

    for (int i = 0; i < 9; i++) {

        if (te[i] != me[i]) return false;
    }

    return true;
}

bool Matrix3::operator==(const Matrix3& matrix) const {

    return equals(matrix);
}
