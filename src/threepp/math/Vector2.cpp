
#include "threepp/math/Vector2.hpp"

#include "threepp/math/Matrix3.hpp"

#include <algorithm>
#include <iostream>
#include <string>

using namespace threepp;

Vector2::Vector2(float x, float y) : x(x), y(y) {}

Vector2 &Vector2::set(float x, float y) {

    this->x = x;
    this->y = y;

    return *this;
}

Vector2 &Vector2::setScalar(float value) {

    this->x = value;
    this->y = value;

    return *this;
}

Vector2 &Vector2::setX(float value) {

    this->x = value;

    return *this;
}

Vector2 &Vector2::setY(float value) {

    y = value;

    return *this;
}

float &Vector2::operator[](unsigned int index) {
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

Vector2 &Vector2::copy(const Vector2 &v) {

    this->x = v.x;
    this->y = v.y;

    return *this;
}

Vector2 &Vector2::add(const Vector2 &v) {

    this->x += v.x;
    this->y += v.y;

    return *this;
}

Vector2 &Vector2::add(float s) {

    this->x += s;
    this->y += s;

    return *this;
}

Vector2 &Vector2::addVectors(const Vector2 &a, const Vector2 &b) {

    this->x = a.x + b.x;
    this->y = a.y + b.y;

    return *this;
}

Vector2 &Vector2::addScaledVector(const Vector2 &v, float s) {

    this->x += v.x * s;
    this->y += v.y * s;

    return *this;
}

Vector2 &Vector2::sub(const Vector2 &v) {

    this->x -= v.x;
    this->y -= v.y;

    return *this;
}

Vector2 &Vector2::sub(float s) {

    this->x -= s;
    this->y -= s;

    return *this;
}

Vector2 &Vector2::subVectors(const Vector2 &a, const Vector2 &b) {

    this->x = a.x - b.x;
    this->y = a.y - b.y;

    return *this;
}

Vector2 &Vector2::multiply(const Vector2 &v) {

    this->x *= v.x;
    this->y *= v.y;

    return *this;
}

Vector2 &Vector2::multiply(float scalar) {

    this->x *= scalar;
    this->y *= scalar;

    return *this;
}

Vector2 &Vector2::divide(const Vector2 &v) {

    this->x /= v.x;
    this->y /= v.y;

    return *this;
}

Vector2 &Vector2::divide(float scalar) {

    return this->multiply(1.0f / scalar);
}

Vector2 &Vector2::applyMatrix3(const Matrix3 &m) {

    const auto x_ = this->x, y_ = this->y;
    const auto& e = m.elements();

    this->x = e[0] * x_ + e[3] * y_ + e[6];
    this->y = e[1] * x_ + e[4] * y_ + e[7];

    return *this;
}

Vector2 &Vector2::min(const Vector2 &v) {

    this->x = std::min(this->x, v.x);
    this->y = std::min(this->y, v.y);

    return *this;
}

Vector2 &Vector2::max(const Vector2 &v) {

    this->x = std::max(this->x, v.x);
    this->y = std::max(this->y, v.y);

    return *this;
}
