
#include "threepp/math/Spherical.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

Spherical::Spherical(float radius, float phi, float theta)
    : radius(radius), phi(phi), theta(theta) {}


Spherical& Spherical::set(float radius, float phi, float theta) {

    this->radius = radius;
    this->phi = phi;
    this->theta = theta;

    return *this;
}

Spherical& Spherical::copy(const Spherical& other) {

    this->radius = other.radius;
    this->phi = other.phi;
    this->theta = other.theta;

    return *this;
}

Spherical& Spherical::makeSafe() {

    const auto EPS = 0.000001f;
    this->phi = std::max(EPS, std::min(math::PI - EPS, this->phi));

    return *this;
}

Spherical& Spherical::setFromVector3(const Vector3& v) {

    return this->setFromCartesianCoords(v.x, v.y, v.z);
}

Spherical& Spherical::setFromCartesianCoords(float x, float y, float z) {

    this->radius = std::sqrt(x * x + y * y + z * z);

    if (this->radius == 0) {

        this->theta = 0;
        this->phi = 0;

    } else {

        this->theta = std::atan2(x, z);
        this->phi = std::acos(std::clamp(y / this->radius, -1.f, 1.f));
    }

    return *this;
}
