
#include "threepp/math/Color.hpp"

using namespace threepp;

Color::Color(float r, float g, float b) : r(r), g(g), b(b) {}

Color &Color::setRGB(float r, float g, float b) {

    this->r = r;
    this->g = g;
    this->b = b;

    return *this;
}
