
#include "threepp/math/Plane.hpp"


using namespace threepp;

Vector3 Plane::_vector1 = Vector3();
Vector3 Plane::_vector2 = Vector3();
Matrix3 Plane::_normalMatrix = Matrix3();


Plane::Plane() : normal(Vector3(1, 0, 0)), constant(0){}

Plane::Plane(Vector3 normal, float constant) : normal(normal), constant(constant) {}

Plane &Plane::set(const Vector3 &normal, float constant) {

    this->normal = normal;
    this->constant = constant;

    return *this;
}

Plane &Plane::setComponents(float x, float y, float z, float w) {

    this->normal.set( x, y, z );
    this->constant = w;

    return *this;

};
