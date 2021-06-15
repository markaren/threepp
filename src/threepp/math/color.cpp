
#include "threepp/math/color.hpp"

using namespace threepp;

color::color(float r, float g, float b) : r(r), g(g), b(b) {}

color &color::setRGB(float r, float g, float b) {

    this->r = r;
    this->g = g;
    this->b = b;

    return *this;
}
