
#include "threepp/math/Cylindrical.hpp"

#include "threepp/math/Vector3.hpp"

#include <cmath>

using namespace threepp;

Cylindrical::Cylindrical(float radius, float theta, float y): radius_(radius), theta_(theta), y_(y) {}

Cylindrical& Cylindrical::set(float radius, float theta, float y) {

    this->radius_ = radius;
    this->theta_ = theta;
    this->y_ = y;

    return *this;
}

Cylindrical& Cylindrical::copy(const Cylindrical& other) {

    this->radius_ = other.radius_;
    this->theta_ = other.theta_;
    this->y_ = other.y_;

    return *this;
}

Cylindrical& Cylindrical::setFromVector3(const Vector3& v) {

    return this->setFromCartesianCoords(v.x, v.y, v.z);
}

Cylindrical& Cylindrical::setFromCartesianCoords(float x, float y, float z) {

    this->radius_ = std::sqrt(x * x + z * z);
    this->theta_ = std::atan2(x, z);
    this->y_ = y;

    return *this;
}
