
#include "threepp/math/Plane.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Line3.hpp"
#include "threepp/math/Sphere.hpp"
#include "threepp/math/Matrix3.hpp"

using namespace threepp;

Plane::Plane(): normal({1, 0, 0}), constant(0.f) {}

Plane::Plane(Vector3 normal, float constant): normal(normal), constant(constant) {}

Plane& Plane::set(const Vector3& normal, float constant) {

    this->normal = normal;
    this->constant = constant;

    return *this;
}

Plane& Plane::setComponents(float x, float y, float z, float w) {

    this->normal.set(x, y, z);
    this->constant = w;

    return *this;
}

Plane& Plane::setFromNormalAndCoplanarPoint(const Vector3& normal, const Vector3& point) {

    this->normal.copy(normal);
    this->constant = -point.dot(this->normal);

    return *this;
}

Plane& Plane::setFromCoplanarPoints(const Vector3& a, const Vector3& b, const Vector3& c) {

    Vector3 _vector1;
    Vector3 _vector2;
    const auto& normal = _vector1.subVectors(c, b).cross(_vector2.subVectors(a, b)).normalize();

    // Q: should an error be thrown if normal is zero (e.g. degenerate plane)?

    this->setFromNormalAndCoplanarPoint(normal, a);

    return *this;
}

Plane& Plane::copy(const Plane& plane) {

    this->normal.copy(plane.normal);
    this->constant = plane.constant;

    return *this;
}

Plane& Plane::normalize() {

    // Note: will lead to a divide by zero if the plane is invalid.

    const auto inverseNormalLength = 1.0f / this->normal.length();
    this->normal.multiplyScalar(inverseNormalLength);
    this->constant *= inverseNormalLength;

    return *this;
}

Plane& Plane::negate() {

    this->constant *= -1;
    this->normal.negate();

    return *this;
}

float Plane::distanceToPoint(const Vector3& point) const {

    return this->normal.dot(point) + this->constant;
}

float Plane::distanceToSphere(const Sphere& sphere) const {

    return this->distanceToPoint(sphere.center) - sphere.radius;
}

void Plane::projectPoint(const Vector3& point, Vector3& target) const {

    target.copy(this->normal).multiplyScalar(-this->distanceToPoint(point)).add(point);
}

void Plane::intersectLine(const Line3& line, Vector3& target) const {

    Vector3 _vector1;
    line.delta(_vector1);
    const auto& direction = _vector1;

    const auto denominator = this->normal.dot(direction);

    if (denominator == 0) {

        // line is coplanar, return origin
        if (this->distanceToPoint(line.start()) == 0) {

            target.copy(line.end());
        }

        // Unsure if this is the correct method to handle this case.
        return;
    }

    const auto t = -(line.start().dot(this->normal) + this->constant) / denominator;

    if (t < 0 || t > 1) {

        return;
    }

    target.copy(direction).multiplyScalar(t).add(line.start());
}

bool Plane::intersectsLine(const Line3& line) const {

    // Note: this tests if a line intersects the plane, not whether it (or its end-points) are coplanar with it.

    const auto startSign = this->distanceToPoint(line.start());
    const auto endSign = this->distanceToPoint(line.end());

    return (startSign < 0 && endSign > 0) || (endSign < 0 && startSign > 0);
}

bool Plane::intersectsBox(const Box3& box) const {

    return box.intersectsPlane(*this);
}

bool Plane::intersectsSphere(const Sphere& sphere) const {

    return sphere.intersectsPlane(*this);
}

void Plane::coplanarPoint(Vector3& target) const {

    target.copy(this->normal).multiplyScalar(-this->constant);
}

Plane& Plane::applyMatrix4(const Matrix4& matrix) {

    Vector3 _vector1;
    Matrix3 _normalMatrix;

    const auto& normalMatrix = _normalMatrix.getNormalMatrix(matrix);

    this->coplanarPoint(_vector1);
    const auto& referencePoint = _vector1.applyMatrix4(matrix);

    const auto& normal = this->normal.applyMatrix3(normalMatrix).normalize();

    this->constant = -referencePoint.dot(normal);

    return *this;
}

Plane& Plane::applyMatrix4(const Matrix4& matrix, Matrix3& normalMatrix) {

    Vector3 _vector1;

    this->coplanarPoint(_vector1);
    const auto& referencePoint = _vector1.applyMatrix4(matrix);

    const auto& normal = this->normal.applyMatrix3(normalMatrix).normalize();

    this->constant = -referencePoint.dot(normal);

    return *this;
}

Plane& Plane::translate(const Vector3& offset) {

    this->constant -= offset.dot(this->normal);

    return *this;
}

bool Plane::equals(const Plane& plane) const {

    return plane.normal.equals(this->normal) && (plane.constant == this->constant);
}

bool Plane::operator==(const Plane& plane) const {

    return equals(plane);
};
