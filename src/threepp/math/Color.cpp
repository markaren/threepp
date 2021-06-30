
#include "threepp/math/Color.hpp"

#include <cmath>
#include <algorithm>

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

    this->r = (float) (hex >> 16 & 255) / 255;
    this->g = (float) (hex >> 8 & 255) / 255;
    this->b = (float) (hex & 255) / 255;

    return *this;
}

Color &Color::setRGB(float r, float g, float b) {

    this->r = r;
    this->g = g;
    this->b = b;

    return *this;
}

Color &Color::copy(const Color &color) {

    this->r = color.r;
    this->g = color.g;
    this->b = color.b;

    return *this;
}

Color &Color::add(const Color &color) {

    this->r += color.r;
    this->g += color.g;
    this->b += color.b;

    return *this;
}

Color &Color::addColors(const Color &color1, const Color &color2) {

    this->r = color1.r + color2.r;
    this->g = color1.g + color2.g;
    this->b = color1.b + color2.b;

    return *this;
}

Color &Color::addScalar(float s) {

    this->r += s;
    this->g += s;
    this->b += s;

    return *this;
}

Color &Color::sub(const Color &color) {

    this->r = std::max(0.f, this->r - color.r);
    this->g = std::max(0.f, this->g - color.g);
    this->b = std::max(0.f, this->b - color.b);

    return *this;
}

Color &Color::multiply(const Color &color) {

    this->r *= color.r;
    this->g *= color.g;
    this->b *= color.b;

    return *this;
}

Color &Color::multiplyScalar(float s) {

    this->r *= s;
    this->g *= s;
    this->b *= s;

    return *this;
}

Color &Color::lerp(const Color &color, float alpha) {

    this->r += (color.r - this->r) * alpha;
    this->g += (color.g - this->g) * alpha;
    this->b += (color.b - this->b) * alpha;

    return *this;
}

Color &Color::lerpColors(const Color &color1, const Color &color2, float alpha) {

    this->r = color1.r + (color2.r - color1.r) * alpha;
    this->g = color1.g + (color2.g - color1.g) * alpha;
    this->b = color1.b + (color2.b - color1.b) * alpha;

    return *this;
}

bool Color::equals(const Color &c) const {

    return (c.r == this->r) && (c.g == this->g) && (c.b == this->b);
}
