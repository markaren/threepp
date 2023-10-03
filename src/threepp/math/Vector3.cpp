
#include "threepp/math/Vector3.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Spherical.hpp"

#include "threepp/cameras/Camera.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace threepp;

namespace {
    thread_local Quaternion _quaternion;
}

Vector3::Vector3(): Vector3(0, 0, 0) {}

Vector3::Vector3(float x, float y, float z)
    : x(x), y(y), z(z) {}

Vector3& Vector3::set(float x, float y, float z) {

    this->x = x;
    this->y = y;
    this->z = z;

    return *this;
}

Vector3& Vector3::setScalar(float value) {

    this->x = value;
    this->y = value;
    this->z = value;

    return *this;
}

Vector3& Vector3::setX(float value) {

    this->x = value;

    return *this;
}

Vector3& Vector3::setY(float value) {

    this->y = value;

    return *this;
}

Vector3& Vector3::setZ(float value) {

    this->z = value;

    return *this;
}

float& Vector3::operator[](size_t index) {
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

Vector3& Vector3::copy(const Vector3& v) {

    this->x = v.x;
    this->y = v.y;
    this->z = v.z;

    return *this;
}

Vector3& Vector3::add(const Vector3& v) {

    this->x += v.x;
    this->y += v.y;
    this->z += v.z;

    return *this;
}

Vector3& Vector3::addScalar(float s) {

    this->x += s;
    this->y += s;
    this->z += s;

    return *this;
}

Vector3& Vector3::addVectors(const Vector3& a, const Vector3& b) {

    this->x = a.x + b.x;
    this->y = a.y + b.y;
    this->z = a.z + b.z;

    return *this;
}

Vector3& Vector3::addScaledVector(const Vector3& v, float s) {

    this->x += v.x * s;
    this->y += v.y * s;
    this->z += v.z * s;

    return *this;
}

Vector3& Vector3::sub(const Vector3& v) {

    this->x -= v.x;
    this->y -= v.y;
    this->z -= v.z;

    return *this;
}

Vector3& Vector3::subScalar(float s) {

    this->x -= s;
    this->y -= s;
    this->z -= s;

    return *this;
}

Vector3& Vector3::subVectors(const Vector3& a, const Vector3& b) {

    this->x = a.x - b.x;
    this->y = a.y - b.y;
    this->z = a.z - b.z;

    return *this;
}

Vector3& Vector3::multiply(const Vector3& v) {

    this->x *= v.x;
    this->y *= v.y;
    this->z *= v.z;

    return *this;
}

Vector3& Vector3::multiplyScalar(float scalar) {

    this->x *= scalar;
    this->y *= scalar;
    this->z *= scalar;

    return *this;
}

Vector3& Vector3::multiplyVectors(const Vector3& a, const Vector3& b) {

    this->x = a.x * b.x;
    this->y = a.y * b.y;
    this->z = a.z * b.z;

    return *this;
}

Vector3& Vector3::applyAxisAngle(const Vector3& axis, float angle) {

    return this->applyQuaternion(_quaternion.setFromAxisAngle(axis, angle));
}

Vector3& Vector3::applyEuler(const Euler& euler) {

    return this->applyQuaternion(_quaternion.setFromEuler(euler));
}

Vector3& Vector3::applyMatrix3(const Matrix3& m) {

    const auto x_ = this->x, y_ = this->y, z_ = this->z;
    const auto& e = m.elements;

    this->x = e[0] * x_ + e[3] * y_ + e[6] * z_;
    this->y = e[1] * x_ + e[4] * y_ + e[7] * z_;
    this->z = e[2] * x_ + e[5] * y_ + e[8] * z_;

    return *this;
}

Vector3& Vector3::applyNormalMatrix(const Matrix3& m) {

    return applyMatrix3(m).normalize();
}

Vector3& Vector3::applyMatrix4(const Matrix4& m) {

    const auto x_ = this->x, y_ = this->y, z_ = this->z;
    const auto& e = m.elements;

    const auto w = 1.0f / (e[3] * x + e[7] * y + e[11] * z + e[15]);

    this->x = (e[0] * x_ + e[4] * y_ + e[8] * z_ + e[12]) * w;
    this->y = (e[1] * x_ + e[5] * y_ + e[9] * z_ + e[13]) * w;
    this->z = (e[2] * x_ + e[6] * y_ + e[10] * z_ + e[14]) * w;

    return *this;
}

Vector3& Vector3::applyQuaternion(const Quaternion& q) {

    const auto x = this->x, y = this->y, z = this->z;
    const auto qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();

    // calculate quat * vector

    const auto ix = qw * x + qy * z - qz * y;
    const auto iy = qw * y + qz * x - qx * z;
    const auto iz = qw * z + qx * y - qy * x;
    const auto iw = -qx * x - qy * y - qz * z;

    // calculate result * inverse quat

    this->x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
    this->y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
    this->z = iz * qw + iw * -qz + ix * -qy - iy * -qx;

    return *this;
}

Vector3& Vector3::project(const Camera& camera) {

    return this->applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
}

Vector3& Vector3::unproject(const Camera& camera) {

    return this->applyMatrix4(camera.projectionMatrixInverse).applyMatrix4(*camera.matrixWorld);
}

Vector3& Vector3::transformDirection(const Matrix4& m) {

    // input: THREE.Matrix4 affine matrix
    // vector interpreted as a direction

    const auto x = this->x, y = this->y, z = this->z;
    const auto& e = m.elements;

    this->x = e[0] * x + e[4] * y + e[8] * z;
    this->y = e[1] * x + e[5] * y + e[9] * z;
    this->z = e[2] * x + e[6] * y + e[10] * z;

    return this->normalize();
}

Vector3& Vector3::divide(const Vector3& v) {
    this->x /= v.x;
    this->y /= v.y;
    this->z /= v.z;

    return *this;
}

Vector3& Vector3::divideScalar(float v) {
    this->x /= v;
    this->y /= v;
    this->z /= v;

    return *this;
}

Vector3& Vector3::min(const Vector3& v) {

    this->x = std::min(this->x, v.x);
    this->y = std::min(this->y, v.y);
    this->z = std::min(this->z, v.z);

    return *this;
}

Vector3& Vector3::max(const Vector3& v) {

    this->x = std::max(this->x, v.x);
    this->y = std::max(this->y, v.y);
    this->z = std::max(this->z, v.z);

    return *this;
}

Vector3& Vector3::clamp(const Vector3& min, const Vector3& max) {

    // assumes min < max, componentwise

    this->x = std::max(min.x, std::min(max.x, this->x));
    this->y = std::max(min.y, std::min(max.y, this->y));
    this->z = std::max(min.z, std::min(max.z, this->z));

    return *this;
}

Vector3& Vector3::floor() {

    this->x = std::floor(this->x);
    this->y = std::floor(this->y);
    this->z = std::floor(this->z);

    return *this;
}

Vector3& Vector3::ceil() {

    this->x = std::ceil(this->x);
    this->y = std::ceil(this->y);
    this->z = std::ceil(this->z);

    return *this;
}

Vector3& Vector3::round() {

    this->x = std::round(this->x);
    this->y = std::round(this->y);
    this->z = std::round(this->z);

    return *this;
}

Vector3& Vector3::roundToZero() {

    this->x = (x < 0) ? std::ceil(this->x) : std::floor(this->x);
    this->y = (y < 0) ? std::ceil(this->y) : std::floor(this->y);
    this->z = (z < 0) ? std::ceil(this->z) : std::floor(this->z);

    return *this;
}

Vector3& Vector3::negate() {

    x = -x;
    y = -y;
    z = -z;

    return *this;
}

float Vector3::dot(const Vector3& v) const {

    return x * v.x + y * v.y + z * v.z;
}

float Vector3::lengthSq() const {

    return x * x + y * y + z * z;
}

float Vector3::length() const {

    return std::sqrt(x * x + y * y + z * z);
}

float Vector3::manhattanLength() const {

    return std::abs(x) + std::abs(y) + std::abs(z);
}

Vector3& Vector3::normalize() {

    auto l = length();
    this->divideScalar(std::isnan(l) ? 1 : l);

    return *this;
}

Vector3& Vector3::setLength(float length) {

    return normalize().multiplyScalar(length);
}

Vector3& Vector3::lerp(const Vector3& v, float alpha) {

    this->x += (v.x - x) * alpha;
    this->y += (v.y - y) * alpha;
    this->z += (v.z - z) * alpha;

    return *this;
}

Vector3& Vector3::lerpVectors(const Vector3& v1, const Vector3& v2, float alpha) {

    this->x = v1.x + (v2.x - v1.x) * alpha;
    this->y = v1.y + (v2.y - v1.y) * alpha;
    this->z = v1.z + (v2.z - v1.z) * alpha;

    return *this;
}

Vector3& Vector3::cross(const Vector3& v) {

    return crossVectors(*this, v);
}

Vector3& Vector3::crossVectors(const Vector3& a, const Vector3& b) {

    const auto ax = a.x, ay = a.y, az = a.z;
    const auto bx = b.x, by = b.y, bz = b.z;

    this->x = ay * bz - az * by;
    this->y = az * bx - ax * bz;
    this->z = ax * by - ay * bx;

    return *this;
}

Vector3& Vector3::projectOnVector(const Vector3& v) {

    const auto denominator = v.lengthSq();

    if (denominator == 0) return this->set(0, 0, 0);

    const auto scalar = v.dot(*this) / denominator;

    return this->copy(v).multiplyScalar(scalar);
}

Vector3& Vector3::projectOnPlane(const Vector3& planeNormal) {

    Vector3 _vector;
    _vector.copy(*this).projectOnVector(planeNormal);

    return this->sub(_vector);
}

Vector3& Vector3::reflect(const Vector3& normal) {

    // reflect incident vector off plane orthogonal to normal
    // normal is assumed to have unit length

    Vector3 _vector;
    return this->sub(_vector.copy(normal).multiplyScalar(2 * this->dot(normal)));
}

float Vector3::angleTo(const Vector3& v) const {

    const auto denominator = std::sqrt(lengthSq() * v.lengthSq());

    if (denominator == 0) return math::PI / 2;

    const auto theta = dot(v) / denominator;

    // clamp, to handle numerical problems

    return std::acos(std::clamp(theta, -1.0f, 1.0f));
}

float Vector3::distanceTo(const Vector3& v) const {

    return std::sqrt(distanceToSquared(v));
}

float Vector3::distanceToSquared(const Vector3& v) const {

    const auto dx = this->x - v.x, dy = this->y - v.y, dz = this->z - v.z;

    return dx * dx + dy * dy + dz * dz;
}

float Vector3::manhattanDistanceTo(const Vector3& v) const {

    return std::abs(this->x - v.x) + std::abs(this->y - v.y) + std::abs(this->z - v.z);
}

Vector3& Vector3::setFromSpherical(const Spherical& s) {

    return this->setFromSphericalCoords(s.radius, s.phi, s.theta);
}

Vector3& Vector3::setFromSphericalCoords(float radius, float phi, float theta) {

    const auto sinPhiRadius = std::sin(phi) * radius;

    this->x = sinPhiRadius * std::sin(theta);
    this->y = std::cos(phi) * radius;
    this->z = sinPhiRadius * std::cos(theta);

    return *this;
}

Vector3& Vector3::setFromMatrixPosition(const Matrix4& m) {

    const auto& e = m.elements;

    this->x = e[12];
    this->y = e[13];
    this->z = e[14];

    return *this;
}

Vector3& Vector3::setFromMatrixScale(const Matrix4& m) {

    const auto sx = this->setFromMatrixColumn(m, 0).length();
    const auto sy = this->setFromMatrixColumn(m, 1).length();
    const auto sz = this->setFromMatrixColumn(m, 2).length();

    this->x = sx;
    this->y = sy;
    this->z = sz;

    return *this;
}

Vector3& Vector3::setFromMatrixColumn(const Matrix4& m, unsigned int index) {

    return this->fromArray(m.elements, index * 4);
}

Vector3& Vector3::setFromMatrix3Column(const Matrix3& m, unsigned int index) {

    return this->fromArray(m.elements, index * 3);
}

Vector3 Vector3::clone() const {

    return Vector3{x, y, z};
}

bool Vector3::equals(const Vector3& v) const {

    return ((v.x == this->x) && (v.y == this->y) && (v.z == this->z));
}

bool Vector3::operator!=(const Vector3& other) const {

    return !equals(other);
}
bool Vector3::operator==(const Vector3& other) const {

    return equals(other);
}

bool Vector3::isNan() const {

    return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

Vector3& Vector3::makeNan() {

    return set(NAN, NAN, NAN);
}

Vector3& Vector3::operator/=(float s) {

    return divideScalar(s);
}

Vector3 Vector3::operator/(float s) const {

    return clone().divideScalar(s);
}

Vector3& Vector3::operator/=(const Vector3& other) {

    return divide(other);
}

Vector3 Vector3::operator/(const Vector3& other) const {

    return clone().divide(other);
}

Vector3& Vector3::operator*=(float s) {

    return multiplyScalar(s);
}

Vector3 Vector3::operator*(float s) const {

    return clone().multiplyScalar(s);
}

Vector3& Vector3::operator*=(const Vector3& other) {

    return multiply(other);
}

Vector3 Vector3::operator*(const Vector3& other) const {

    return clone().multiply(other);
}

Vector3& Vector3::operator-=(float s) {

    return subScalar(s);
}

Vector3 Vector3::operator-(float s) const {

    return clone().subScalar(s);
}

Vector3& Vector3::operator-=(const Vector3& other) {

    return sub(other);
}

Vector3 Vector3::operator-(const Vector3& other) const {

    return clone().sub(other);
}
Vector3& Vector3::operator+=(float s) {

    return addScalar(s);
}

Vector3 Vector3::operator+(float s) const {

    return clone().addScalar(s);
}

Vector3& Vector3::operator+=(const Vector3& other) {

    return add(other);
}

Vector3 Vector3::operator+(const Vector3& other) const {

    return clone().add(other);
}
