
#include "threepp/math/vector2.hpp"

#include <iostream>
#include <string>

using namespace threepp;

vector2::vector2(float x, float y) : x(x), y(y) {}

vector2 &vector2::set(float x, float y) {

    this->x = x;
    this->y = y;

    return *this;
}

vector2 &vector2::setScalar(float value) {

    this->x = value;
    this->y = value;

    return *this;
}

vector2 &vector2::setX(float value) {

    this->x = value;

    return *this;
}

vector2 &vector2::setY(float value) {

    y = value;

    return *this;
}

float &vector2::operator[](unsigned int index) {
    if (index >= 2) throw std::runtime_error("index out of bounds: " + std::to_string(index));
    switch (index) {
        case 0: return x;
        case 1: return y;
        default: throw std::runtime_error("index out of bound: " + std::to_string(index));
    }
}

vector2 &vector2::add(const vector2 &v) {

    this->x += v.x;
    this->y += v.y;

    return *this;
}

vector2 &vector2::add(float s) {

    this->x += s;
    this->y += s;

    return *this;
}

vector2 &vector2::addVectors(const vector2 &a, const vector2 &b) {

    this->x = a.x + b.x;
    this->y = a.y + b.y;

    return *this;
}

vector2 &vector2::addScaledVector(const vector2 &v, float s) {

    this->x += v.x * s;
    this->y += v.y * s;

    return *this;
}
