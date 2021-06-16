
#include "threepp/math/Color.hpp"

#include <cmath>

using namespace threepp;

Color::Color(float r, float g, float b) : r(r), g(g), b(b) {}

Color::Color(int hex) {
    setHex(hex);
}

Color &Color::setScalar(float scalar) {

    this->r = scalar;
    this->g = scalar;
    this->b = scalar;

    return *this;
}

Color &Color::setHex(int hex) {

    this->r = (float)( hex >> 16 & 255 ) / 255;
    this->g = (float)( hex >> 8 & 255 ) / 255;
    this->b = (float)( hex & 255 ) / 255;

    return *this;
}

Color &Color::setRGB(float r, float g, float b) {

    this->r = r;
    this->g = g;
    this->b = b;

    return *this;
}

