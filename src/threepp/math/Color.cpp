
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
    hex = std::floor( hex );

    this->r = ( hex >> 16 & 255 ) / 255;
    this->g = ( hex >> 8 & 255 ) / 255;
    this->b = ( hex & 255 ) / 255;

    return *this;
}

Color &Color::setRGB(float r, float g, float b) {

    this->r = r;
    this->g = g;
    this->b = b;

    return *this;
}

