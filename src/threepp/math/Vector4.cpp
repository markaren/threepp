
#include "threepp/math/Vector4.hpp"

#include "threepp/math/Matrix4.hpp"

#include <cmath>
#include <string>

using namespace threepp;


Vector4::Vector4(int x, int y, int z, int w)
    : x((float) x), y((float) y), z((float) z), w((float) w) {}

Vector4::Vector4(float x, float y, float z, float w)
    : x(x), y(y), z(z), w(w) {}

float& Vector4::operator[](unsigned int index) {
    switch (index) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
        default:
            throw std::runtime_error("index out of bound: " + std::to_string(index));
    }
}

Vector4& Vector4::set(float x, float y, float z, float w) {

    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;

    return *this;
}

Vector4& Vector4::setScalar(float value) {

    this->x = value;
    this->y = value;
    this->z = value;
    this->w = value;

    return *this;
}

Vector4& Vector4::copy(const Vector4& v) {

    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->w = v.w;

    return *this;
}

Vector4& Vector4::add(const Vector4& v) {

    this->x += v.x;
    this->y += v.y;
    this->z += v.z;
    this->w += v.w;

    return *this;
}
Vector4& Vector4::addScalar(float s) {

    this->x += s;
    this->y += s;
    this->z += s;
    this->w += s;

    return *this;
}

Vector4& Vector4::addVectors(const Vector4& a, const Vector4& b) {

    this->x = a.x + b.x;
    this->y = a.y + b.y;
    this->z = a.z + b.z;
    this->w = a.w + b.w;

    return *this;
}
Vector4& Vector4::addScaledVector(const Vector4& v, float s) {

    this->x += v.x * s;
    this->y += v.y * s;
    this->z += v.z * s;
    this->w += v.w * s;

    return *this;
}

Vector4& Vector4::multiply(const Vector4& v) {

    this->x *= v.x;
    this->y *= v.y;
    this->z *= v.z;
    this->w *= v.w;

    return *this;
}
Vector4& Vector4::multiplyScalar(float scalar) {

    this->x *= scalar;
    this->y *= scalar;
    this->z *= scalar;
    this->w *= scalar;

    return *this;
}

Vector4& Vector4::applyMatrix4(const Matrix4& m) {

    const auto x_ = this->x, y_ = this->y, z_ = this->z, w_ = this->w;
    const auto& e = m.elements;

    this->x = e[0] * x_ + e[4] * y_ + e[8] * z_ + e[12] * w_;
    this->y = e[1] * x_ + e[5] * y_ + e[9] * z_ + e[13] * w_;
    this->z = e[2] * x_ + e[6] * y_ + e[10] * z_ + e[14] * w_;
    this->w = e[3] * x_ + e[7] * y_ + e[11] * z_ + e[15] * w_;

    return *this;
}

Vector4& Vector4::divideScalar(float scalar) {

    return this->multiplyScalar(1.f / scalar);
}

Vector4& Vector4::floor() {

    this->x = std::floor(this->x);
    this->y = std::floor(this->y);
    this->z = std::floor(this->z);
    this->w = std::floor(this->w);

    return *this;
}
Vector4& Vector4::ceil() {

    this->x = std::ceil(this->x);
    this->y = std::ceil(this->y);
    this->z = std::ceil(this->z);
    this->w = std::ceil(this->w);

    return *this;
}
Vector4& Vector4::round() {

    this->x = std::round(this->x);
    this->y = std::round(this->y);
    this->z = std::round(this->z);
    this->w = std::round(this->w);

    return *this;
}
Vector4& Vector4::roundToZero() {

    this->x = (this->x < 0) ? std::ceil(this->x) : std::floor(this->x);
    this->y = (this->y < 0) ? std::ceil(this->y) : std::floor(this->y);
    this->z = (this->z < 0) ? std::ceil(this->z) : std::floor(this->z);
    this->w = (this->w < 0) ? std::ceil(this->w) : std::floor(this->w);

    return *this;
}
Vector4& Vector4::negate() {

    this->x = -this->x;
    this->y = -this->y;
    this->z = -this->z;
    this->w = -this->w;

    return *this;
}

float Vector4::dot(const Vector4& v) const {

    return this->x * v.x + this->y * v.y + this->z * v.z + this->w * v.w;
}

float Vector4::lengthSq() const {

    return this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w;
}

float Vector4::length() const {

    return std::sqrt(this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w);
}

float Vector4::manhattanLength() const {

    return std::abs(this->x) + std::abs(this->y) + std::abs(this->z) + std::abs(this->w);
}

Vector4& Vector4::normalize() {

    const auto len = this->length();
    return this->divideScalar(std::isnan(len) ? 1 : len);
}

Vector4& Vector4::setLength(float length) {

    return this->normalize().multiplyScalar(length);
}

Vector4 Vector4::clone() const {

    return Vector4{x, y, z, w};
}

bool Vector4::equals(const Vector4& v) const {

    return ((v.x == this->x) && (v.y == this->y) && (v.z == this->z) && (v.w == this->w));
}

bool Vector4::operator==(const Vector4& other) const {

    return equals(other);
}

bool Vector4::operator!=(const Vector4& other) const {

    return !equals(other);
}
