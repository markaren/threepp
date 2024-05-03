
#include "threepp/scenes/Fog.hpp"

using namespace threepp;


Fog::Fog(const Color& color, float near, float far)
    : color(color), near(near), far(far) {}

bool Fog::operator==(const Fog& f) const {

    return f.color == this->color && f.near == this->near && f.far == this->far;
}
