
#include "threepp/math/Vector2.hpp"

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
        case 0: return x;
        case 1: return y;
        default: throw std::runtime_error("index out of bound: " + std::to_string(index));
    }
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
