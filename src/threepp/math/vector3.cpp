
#include "threepp/math/vector3.hpp"

#include "threepp/math/matrix3.hpp"
#include "threepp/math/matrix4.hpp"

#include "threepp/math/math_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace threepp;

vector3::vector3(double x, double y, double z) : x(x), y(y), z(z) {}

vector3 &vector3::set(double x, double y, double z) {

    this->x = x;
    this->y = y;
    this->z = z;

    return *this;
}

vector3 &vector3::setScalar(double value) {

    this->x = value;
    this->y = value;
    this->z = value;

    return *this;
}

vector3 &vector3::setX(double value) {

    this->x = value;

    return *this;
}

vector3 &vector3::setY(double value) {

    y = value;

    return *this;
}

vector3 &vector3::setZ(double value) {

    z = value;

    return *this;
}

double &vector3::operator[](unsigned int index) {
    if (index >= 3) throw std::runtime_error("index out of bounds: " + std::to_string(index));
    switch (index) {
        case 0: return x;
        case 1: return y;
        case 2: return z;
        default: throw std::runtime_error("index out of bound: " + std::to_string(index));
    }
}

vector3 &vector3::add(const vector3 &v) {

    this->x += v.x;
    this->y += v.y;
    this->z += v.z;

    return *this;
}

vector3 &vector3::add(double s) {

    this->x += s;
    this->y += s;
    this->z += s;

    return *this;
}

vector3 &vector3::addVectors(const vector3 &a, const vector3 &b) {

    this->x = a.x + b.x;
    this->y = a.y + b.y;
    this->z = a.z + b.z;

    return *this;
}

vector3 &vector3::addScaledVector(const vector3 &v, double s) {

    this->x += v.x * s;
    this->y += v.y * s;
    this->z += v.z * s;

    return *this;
}

vector3 &vector3::sub(const vector3 &v) {

    this->x -= v.x;
    this->y -= v.y;
    this->z -= v.z;

    return *this;
}

vector3 &vector3::sub(double s) {

    this->x -= s;
    this->y -= s;
    this->z -= s;

    return *this;
}

vector3 &vector3::subVectors(const vector3 &a, const vector3 &b) {

    this->x = a.x - b.x;
    this->y = a.y - b.y;
    this->z = a.z - b.z;

    return *this;
}

vector3 &vector3::multiply(const vector3 &v) {

    this->x *= v.x;
    this->y *= v.y;
    this->z *= v.z;

    return *this;
}

vector3 &vector3::multiply(double scalar) {

    this->x *= scalar;
    this->y *= scalar;
    this->z *= scalar;

    return *this;
}

vector3 &vector3::multiplyVectors(const vector3 &a, const vector3 &b) {

    this->x = a.x * b.x;
    this->y = a.y * b.y;
    this->z = a.z * b.z;

    return *this;
}

vector3 &vector3::applyMatrix3(const matrix3 &m) {

//    const auto x_ = this->x, y_ = this->y, z_ = this->z;
//    const auto e = m.elements_;
//
//    this->x = e[0] * x_ + e[3] * y_ + e[6] * z_;
//    this->y = e[1] * x_ + e[4] * y_ + e[7] * z_;
//    this->z = e[2] * x_ + e[5] * y_ + e[8] * z_;

    return *this;
}

vector3 &vector3::applyNormalMatrix(const matrix3 &m) {

    return applyMatrix3(m).normalize();
}

vector3 &vector3::applyMatrix4(const matrix4 &m) {

    const auto x_ = this->x, y_ = this->y, z_ = this->z;
    const auto e = m.elements_;

    const auto w = 1.0 / (e[3] * x + e[7] * y + e[11] * z + e[15]);

    this->x = (e[0] * x_ + e[4] * y_ + e[8] * z_ + e[12]) * w;
    this->y = (e[1] * x_ + e[5] * y_ + e[9] * z_ + e[13]) * w;
    this->z = (e[2] * x_ + e[6] * y_ + e[10] * z_ + e[14]) * w;

    return *this;
}

vector3 &vector3::divide(const vector3 &v) {
    this->x /= v.x;
    this->y /= v.y;
    this->z /= v.z;

    return *this;
}

vector3 &vector3::divide(const double &v) {
    this->x /= v;
    this->y /= v;
    this->z /= v;

    return *this;
}

vector3 &vector3::min(const vector3 &v) {

    this->x = std::min(this->x, v.x);
    this->y = std::min(this->y, v.y);
    this->z = std::min(this->z, v.z);

    return *this;
}

vector3 &vector3::max(const vector3 &v) {

    this->x = std::max(this->x, v.x);
    this->y = std::max(this->y, v.y);
    this->z = std::max(this->z, v.z);

    return *this;
}

vector3 &vector3::clamp(const vector3 &min, const vector3 &max) {

    // assumes min < max, componentwise

    this->x = std::max(min.x, std::min(max.x, this->x));
    this->y = std::max(min.y, std::min(max.y, this->y));
    this->z = std::max(min.z, std::min(max.z, this->z));

    return *this;
}

vector3 &vector3::floor() {

    this->x = std::floor(this->x);
    this->y = std::floor(this->y);
    this->z = std::floor(this->z);

    return *this;
}

vector3 &vector3::ceil() {

    this->x = std::ceil(this->x);
    this->y = std::ceil(this->y);
    this->z = std::ceil(this->z);

    return *this;
}

vector3 &vector3::round() {

    this->x = std::round(this->x);
    this->y = std::round(this->y);
    this->z = std::round(this->z);

    return *this;
}

vector3 &vector3::roundToZero() {

    this->x = (x < 0) ? std::ceil(this->x) : std::floor(this->x);
    this->y = (y < 0) ? std::ceil(this->y) : std::floor(this->y);
    this->z = (z < 0) ? std::ceil(this->z) : std::floor(this->z);

    return *this;
}

vector3 &vector3::negate() {

    x = -x;
    y = -y;
    z = -z;

    return *this;
}

double vector3::dot(const vector3 &v) const {

    return x * v.x + y * v.y + z * v.z;
}

double vector3::lengthSq() const {

    return x * x + y * y + z * z;
}

double vector3::length() const {

    return std::sqrt(x * x + y * y + z * z);
}

double vector3::manhattanLength() const {

    return std::abs(x) + std::abs(y) + std::abs(z);
}

vector3 &vector3::normalize() {

    auto l = length();
    return divide(isnan(l) ? 0 : l);
}

vector3 &vector3::setLength(double length) {

    return normalize().multiply(length);
}

vector3 &vector3::lerp(const vector3 &v, double alpha) {

    this->x += (v.x - x) * alpha;
    this->y += (v.y - y) * alpha;
    this->z += (v.z - z) * alpha;

    return *this;
}

vector3 &vector3::lerpVectors(const vector3 &v1, const vector3 &v2, double alpha) {

    this->x = v1.x + (v2.x - v1.x) * alpha;
    this->y = v1.y + (v2.y - v1.y) * alpha;
    this->z = v1.z + (v2.z - v1.z) * alpha;

    return *this;
}

vector3 &vector3::cross(const vector3 &v) {

    return crossVectors(*this, v);
}

vector3 &vector3::crossVectors(const vector3 &a, const vector3 &b) {

    const auto ax = a.x, ay = a.y, az = a.z;
    const auto bx = b.x, by = b.y, bz = b.z;

    this->x = ay * bz - az * by;
    this->y = az * bx - ax * bz;
    this->z = ax * by - ay * bx;

    return *this;
}

double vector3::angleTo(const vector3 &v) const {

    const auto denominator = std::sqrt(lengthSq() * v.lengthSq());

    if (denominator == 0) return PI / 2;

    const auto theta = dot(v) / denominator;

    // clamp, to handle numerical problems

    return std::acos(std::clamp(theta, -1.0, 1.0));
}

double vector3::distanceTo(const vector3 &v) const {

    return std::sqrt(distanceToSquared(v));
}

double vector3::distanceToSquared(const vector3 &v) const {

    const auto dx = this->x - v.x, dy = this->y - v.y, dz = this->z - v.z;

    return dx * dx + dy * dy + dz * dz;
}

double vector3::manhattanDistanceTo(const vector3 &v) const {

    return std::abs(this->x - v.x) + std::abs(this->y - v.y) + std::abs(this->z - v.z);
}

vector3 &vector3::setFromMatrixPosition(const matrix4 &m) {

    const auto e = m.elements_;

    this->x = e[12];
    this->y = e[13];
    this->z = e[14];

    return *this;
}

vector3 &vector3::setFromMatrixScale(const matrix4 &m) {

    const auto sx = this->setFromMatrixColumn(m, 0).length();
    const auto sy = this->setFromMatrixColumn(m, 1).length();
    const auto sz = this->setFromMatrixColumn(m, 2).length();

    this->x = sx;
    this->y = sy;
    this->z = sz;

    return *this;
}

vector3 &vector3::setFromMatrixColumn(const matrix4 &m, unsigned int index) {

    return this->fromArray(m.elements_, index * 4);
}

vector3 &vector3::setFromMatrix3Column(const matrix3 &m, unsigned int index) {

    return this->fromArray(m.elements_, index * 3);
}

//std::ostream &operator<<(std::ostream &os, const vector3 &v) {
//    os << "vector3(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ", z=" + std::to_string(v.z) +
//          ")";
//    return os;
//}
