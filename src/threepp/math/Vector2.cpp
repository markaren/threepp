
#include "threepp/math/Vector2.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

Vector2::Vector2(int x, int y): Vector2(static_cast<float>(x), static_cast<float>(y)) {}

Vector2::Vector2(float x, float y): x(x), y(y) {}

Vector2::Vector2(double x, double y): x(x), y(y) {}

Vector2& Vector2::set(float x, float y) {

    this->x = x;
    this->y = y;

    return *this;
}

Vector2& Vector2::setScalar(float value) {

    this->x = value;
    this->y = value;

    return *this;
}

Vector2& Vector2::setX(float value) {

    this->x = value;

    return *this;
}

Vector2& Vector2::setY(float value) {

    this->y = value;

    return *this;
}

float& Vector2::operator[](unsigned int index) {
    if (index >= 2) throw std::runtime_error("index out of bounds: " + std::to_string(index));
    switch (index) {
        case 0:
            return x;
        case 1:
            return y;
        default:
            throw std::runtime_error("index out of bound: " + std::to_string(index));
    }
}

float Vector2::operator[](unsigned int index) const {
    if (index >= 2) throw std::runtime_error("index out of bounds: " + std::to_string(index));
    switch (index) {
        case 0:
            return x;
        case 1:
            return y;
        default:
            throw std::runtime_error("index out of bound: " + std::to_string(index));
    }
}

Vector2& Vector2::copy(const Vector2& v) {

    this->x = v.x;
    this->y = v.y;

    return *this;
}

Vector2& Vector2::add(const Vector2& v) {

    this->x += v.x;
    this->y += v.y;

    return *this;
}

Vector2& Vector2::addScalar(float s) {

    this->x += s;
    this->y += s;

    return *this;
}

Vector2& Vector2::addVectors(const Vector2& a, const Vector2& b) {

    this->x = a.x + b.x;
    this->y = a.y + b.y;

    return *this;
}

Vector2& Vector2::addScaledVector(const Vector2& v, float s) {

    this->x += v.x * s;
    this->y += v.y * s;

    return *this;
}

Vector2& Vector2::sub(const Vector2& v) {

    this->x -= v.x;
    this->y -= v.y;

    return *this;
}

Vector2& Vector2::subScalar(float s) {

    this->x -= s;
    this->y -= s;

    return *this;
}

Vector2& Vector2::multiplyScalar(float scalar) {

    this->x *= scalar;
    this->y *= scalar;

    return *this;
}

Vector2& Vector2::divide(const Vector2& v) {

    this->x /= v.x;
    this->y /= v.y;

    return *this;
}

Vector2& Vector2::divideScalar(float scalar) {

    return this->multiplyScalar(1.0f / scalar);
}

Vector2& Vector2::applyMatrix3(const Matrix3& m) {

    const auto x_ = this->x, y_ = this->y;
    const auto& e = m.elements;

    this->x = e[0] * x_ + e[3] * y_ + e[6];
    this->y = e[1] * x_ + e[4] * y_ + e[7];

    return *this;
}

Vector2& Vector2::min(const Vector2& v) {

    this->x = std::min(this->x, v.x);
    this->y = std::min(this->y, v.y);

    return *this;
}

Vector2& Vector2::max(const Vector2& v) {

    this->x = std::max(this->x, v.x);
    this->y = std::max(this->y, v.y);

    return *this;
}

Vector2& Vector2::clamp(const Vector2& min, const Vector2& max) {

    // assumes min < max, componentwise

    this->x = std::max(min.x, std::min(max.x, this->x));
    this->y = std::max(min.y, std::min(max.y, this->y));

    return *this;
}

Vector2& Vector2::clampScalar(float minVal, float maxVal) {

    this->x = std::max(minVal, std::min(maxVal, this->x));
    this->y = std::max(minVal, std::min(maxVal, this->y));

    return *this;
}

Vector2& Vector2::clampLength(float min, float max) {

    const auto length = this->length();

    return this->divideScalar(std::isnan(length) ? 1 : length).multiplyScalar(std::max(min, std::min(max, length)));
}

Vector2& Vector2::floor() {

    this->x = std::floor(this->x);
    this->y = std::floor(this->y);

    return *this;
}

Vector2& Vector2::ceil() {

    this->x = std::ceil(this->x);
    this->y = std::ceil(this->y);

    return *this;
}

Vector2& Vector2::round() {

    this->x = std::round(this->x);
    this->y = std::round(this->y);

    return *this;
}

Vector2& Vector2::roundToZero() {

    this->x = (this->x < 0) ? std::ceil(this->x) : std::floor(this->x);
    this->y = (this->y < 0) ? std::ceil(this->y) : std::floor(this->y);

    return *this;
}

Vector2& Vector2::negate() {

    this->x = -this->x;
    this->y = -this->y;

    return *this;
}

float Vector2::dot(const Vector2& v) const {

    return this->x * v.x + this->y * v.y;
}

float Vector2::cross(const Vector2& v) const {

    return this->x * v.y - this->y * v.x;
}

float Vector2::lengthSq() const {

    return this->x * this->x + this->y * this->y;
}

float Vector2::length() const {

    return std::sqrt(this->x * this->x + this->y * this->y);
}

float Vector2::manhattanLength() const {

    return std::abs(this->x) + std::abs(this->y);
}

Vector2& Vector2::normalize() {

    const auto len = this->length();
    return this->divideScalar(std::isnan(len) ? 1 : len);
}

float Vector2::angle() const {

    // computes the angle in radians with respect to the positive x-axis

    const auto angle = std::atan2(-this->y, -this->x) + math::PI;

    return angle;
}

float Vector2::angleTo(const Vector2& v) const {

    const auto denominator = std::sqrt(this->lengthSq() * v.lengthSq());

    if (denominator == 0) return math::PI / 2;

    const auto theta = this->dot(v) / denominator;

    // clamp, to handle numerical problems

    return std::acos(std::clamp(theta, -1.f, 1.f));
}

float Vector2::distanceTo(const Vector2& v) const {

    return std::sqrt(this->distanceToSquared(v));
}

float Vector2::distanceToSquared(const Vector2& v) const {

    const auto dx = this->x - v.x, dy = this->y - v.y;
    return dx * dx + dy * dy;
}

float Vector2::manhattanDistanceTo(const Vector2& v) const {

    return std::abs(this->x - v.x) + std::abs(this->y - v.y);
}

Vector2& Vector2::setLength(float length) {

    return this->normalize().multiplyScalar(length);
}

Vector2& Vector2::lerp(const Vector2& v, float alpha) {

    this->x += (v.x - this->x) * alpha;
    this->y += (v.y - this->y) * alpha;

    return *this;
}

Vector2& Vector2::lerpVectors(const Vector2& v1, const Vector2& v2, float alpha) {

    this->x = v1.x + (v2.x - v1.x) * alpha;
    this->y = v1.y + (v2.y - v1.y) * alpha;

    return *this;
}

bool Vector2::isNan() const {

    return std::isnan(x) || std::isnan(y);
}

Vector2& Vector2::makeNan() {

    return set(NAN, NAN);
}

Vector2 Vector2::clone() const {

    return {x, y};
}

bool Vector2::equals(const Vector2& v) const {

    return ((v.x == this->x) && (v.y == this->y));
}

bool Vector2::operator==(const Vector2& other) const {

    return equals(other);
}

bool Vector2::operator!=(const Vector2& other) const {

    return !equals(other);
}

Vector2 Vector2::operator+(const Vector2& other) const {

    return clone().add(other);
}

Vector2& Vector2::operator+=(const Vector2& other) {

    return add(other);
}

Vector2 Vector2::operator-(const Vector2& other) const {

    return clone().sub(other);
}

Vector2& Vector2::operator-=(const Vector2& other) {

    return sub(other);
}

Vector2& Vector2::rotateAround(const Vector2& center, float angle) {

    float c = std::cos(angle), s = std::sin(angle);

    float x = this->x - center.x;
    float y = this->y - center.y;

    this->x = x * c - y * s + center.x;
    this->y = x * s + y * c + center.y;

    return *this;
}

Vector2 Vector2::operator+(float value) const {

    return Vector2(x, y).addScalar(value);
}

Vector2 Vector2::operator-(float value) const {

    return Vector2(x, y).subScalar(value);
}

Vector2 Vector2::operator*(float value) const {

    return Vector2(x, y).multiplyScalar(value);
}

Vector2 Vector2::operator/(float value) const {

    return Vector2(x, y).divideScalar(value);
}

Vector2::operator std::pair<int, int>() const {

    return {static_cast<int>(x), static_cast<int>(y)};
}

Vector2::operator std::pair<float, float>() const {

    return {x, y};
}
