
#include "threepp/math/Plane.hpp"
#include "threepp/math/Sphere.hpp"

using namespace threepp;

Vector3 Plane::_vector1 = Vector3();
Vector3 Plane::_vector2 = Vector3();
Matrix3 Plane::_normalMatrix = Matrix3();


Plane::Plane() : normal(Vector3(1, 0, 0)), constant(0) {}

Plane::Plane(Vector3 normal, float constant) : normal(normal), constant(constant) {}

Plane &Plane::set(const Vector3 &normal, float constant) {

    this->normal = normal;
    this->constant = constant;

    return *this;
}

Plane &Plane::setComponents(float x, float y, float z, float w) {

    this->normal.set(x, y, z);
    this->constant = w;

    return *this;
}

Plane &Plane::setFromNormalAndCoplanarPoint(const Vector3 &normal, const Vector3 &point) {

    this->normal.copy(normal);
    this->constant = -point.dot(this->normal);

    return *this;
}

Plane &Plane::setFromCoplanarPoints(const Vector3 &a, const Vector3 &b, const Vector3 &c) {

    const auto normal = _vector1.subVectors(c, b).cross(_vector2.subVectors(a, b)).normalize();

    // Q: should an error be thrown if normal is zero (e.g. degenerate plane)?

    this->setFromNormalAndCoplanarPoint(normal, a);

    return *this;
}

Plane &Plane::copy(const Plane &plane) {

    this->normal.copy(plane.normal);
    this->constant = plane.constant;

    return *this;
}

Plane &Plane::normalize() {

    // Note: will lead to a divide by zero if the plane is invalid.

    const auto inverseNormalLength = 1.0f / this->normal.length();
    this->normal.multiply(inverseNormalLength);
    this->constant *= inverseNormalLength;

    return *this;
}

Plane &Plane::negate() {

    this->constant *= -1;
    this->normal.negate();

    return *this;
}

float Plane::distanceToPoint(const Vector3 &point) const {

    return this->normal.dot(point) + this->constant;
}

float Plane::distanceToSphere(const Sphere &sphere) const {

    return this->distanceToPoint(sphere.center()) - sphere.radius();
};
