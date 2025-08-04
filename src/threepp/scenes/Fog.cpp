
#include "threepp/scenes/Fog.hpp"

using namespace threepp;


Fog::Fog(const Color& color, float _near, float _far)
    : color(color), nearPlane(_near), farPlane(_far) {}

bool Fog::operator==(const Fog& f) const {

    return f.color == this->color && f.nearPlane == this->nearPlane && f.farPlane == this->farPlane;
}
