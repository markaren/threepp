
#include "threepp/math/plane.hpp"


using namespace threepp;

vector3 plane::_vector1 = vector3();
vector3 plane::_vector2 = vector3();
matrix3 plane::_normalMatrix = matrix3();


plane::plane() : normal(vector3(1, 0, 0)), constant(0){}

plane::plane(vector3 normal, float constant) : normal(normal), constant(constant) {}

plane &plane::set(const vector3 &normal, float constant) {

    this->normal = normal;
    this->constant = constant;

    return *this;
}

plane &plane::setComponents(float x, float y, float z, float w) {

    this->normal.set( x, y, z );
    this->constant = w;

    return *this;

};
