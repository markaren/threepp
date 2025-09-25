
#include "threepp/math/Sphere.hpp"

#include "threepp/math/Box3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Plane.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

Sphere::Sphere(const Vector3& center, float radius): center(center), radius(radius) {}

Sphere& Sphere::set(const Vector3& center, float radius) {

    this->center = (center);
    this->radius = radius;

    return *this;
}

Sphere& Sphere::setFromPoints(const std::vector<Vector3>& points, Vector3* optionalCenter) {

    if (optionalCenter) {

        center.copy(*optionalCenter);

    } else {

        Box3 _box{};
        _box.setFromPoints(points).getCenter(center);
    }

    float maxRadiusSq = 0;

    for (auto& point : points) {

        maxRadiusSq = std::max(maxRadiusSq, center.distanceToSquared(point));
    }

    this->radius = std::sqrt(maxRadiusSq);

    return *this;
}

Sphere& Sphere::copy(const Sphere& sphere) {

    this->center.copy(sphere.center);
    this->radius = sphere.radius;

    return *this;
}

bool Sphere::isEmpty() const {

    return (this->radius < 0);
}

Sphere& Sphere::makeEmpty() {

    this->center.set(0, 0, 0);
    this->radius = -1;

    return *this;
}

bool Sphere::containsPoint(const Vector3& point) const {

    return (point.distanceToSquared(this->center) <= (this->radius * this->radius));
}

float Sphere::distanceToPoint(const Vector3& point) const {

    return (point.distanceTo(this->center) - this->radius);
}

bool Sphere::intersectsSphere(const Sphere& sphere) const {

    const auto radiusSum = this->radius + sphere.radius;

    return sphere.center.distanceToSquared(this->center) <= (radiusSum * radiusSum);
}

bool Sphere::intersectsBox(const Box3& box) const {

    return box.intersectsSphere(*this);
}

bool Sphere::intersectsPlane(const Plane& plane) const {

    return std::abs(plane.distanceToPoint(this->center)) <= this->radius;
}

void Sphere::clampPoint(const Vector3& point, Vector3& target) const {

    const auto deltaLengthSq = this->center.distanceToSquared(point);

    target.copy(point);

    if (deltaLengthSq > (this->radius * this->radius)) {

        target.sub(this->center).normalize();
        target.multiplyScalar(this->radius).add(this->center);
    }
}

void Sphere::getBoundingBox(Box3& target) const {

    if (this->isEmpty()) {

        // Empty sphere produces empty bounding box
        target.makeEmpty();
        return;
    }

    target.set(this->center, this->center);
    target.expandByScalar(this->radius);
}

Sphere& Sphere::applyMatrix4(const Matrix4& matrix) {

    this->center.applyMatrix4(matrix);
    this->radius = this->radius * matrix.getMaxScaleOnAxis();

    return *this;
}

Sphere& Sphere::translate(const Vector3& offset) {

    this->center.add(offset);

    return *this;
}

Sphere& Sphere::expandByPoint(const Vector3& point) {

    if (this->isEmpty()) {

        this->center.copy(point);
        this->radius = 0;

        return *this;
    }

    Vector3 _v1;
    _v1.subVectors(point, this->center);

    const auto lengthSq = _v1.lengthSq();

    if (lengthSq > (this->radius * this->radius)) {

        // calculate the minimal sphere

        const auto length = std::sqrt(lengthSq);

        const auto delta = (length - this->radius) * 0.5f;

        this->center.addScaledVector(_v1, delta / length);

        this->radius += delta;
    }

    return *this;
}

Sphere& Sphere::union_(const Sphere& sphere) {

    if (sphere.isEmpty()) {

        return *this;
    }

    if (this->isEmpty()) {

        this->copy(sphere);

        return *this;
    }

    if (this->center.equals(sphere.center) == true) {

        this->radius = std::max(this->radius, sphere.radius);

    } else {

        Vector3 _v1, _v2;

        _v2.subVectors(sphere.center, this->center).setLength(sphere.radius);

        this->expandByPoint(_v1.copy(sphere.center).add(_v2));

        this->expandByPoint(_v1.copy(sphere.center).sub(_v2));
    }

    return *this;
}

bool Sphere::equals(const Sphere& sphere) const {

    return sphere.center.equals(this->center) && (sphere.radius == this->radius);
}

Sphere Sphere::clone() const {

    return Sphere().copy(*this);
}
