
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

    // from https://github.com/juj/MathGeoLib/blob/2940b99b99cfe575dd45103ef20f4019dee15b54/src/Geometry/Sphere.cpp#L649-L671

    Vector3 _toPoint{};

    _toPoint.subVectors(point, this->center);

    const float lengthSq = _toPoint.lengthSq();

    if (lengthSq > (this->radius * this->radius)) {

        const float length = std::sqrt(lengthSq);
        const float missingRadiusHalf = (length - this->radius) * 0.5f;

        // Nudge this sphere towards the target point. Add half the missing distance to radius,
        // and the other half to position. This gives a tighter enclosure, instead of if
        // the whole missing distance were just added to radius.

        this->center.add(_toPoint.multiplyScalar(missingRadiusHalf / length));
        this->radius += missingRadiusHalf;
    }

    return *this;
}

Sphere& Sphere::union_(const Sphere& sphere) {

    // from https://github.com/juj/MathGeoLib/blob/2940b99b99cfe575dd45103ef20f4019dee15b54/src/Geometry/Sphere.cpp#L759-L769

    // To enclose another sphere into this sphere, we only need to enclose two points:
    // 1) Enclose the farthest point on the other sphere into this sphere.
    // 2) Enclose the opposite point of the farthest point into this sphere.

    Vector3 _v1{};
    Vector3 _toFarthestPoint{};
    _toFarthestPoint.subVectors(sphere.center, this->center).normalize().multiplyScalar(sphere.radius);

    this->expandByPoint(_v1.copy(sphere.center).add(_toFarthestPoint));
    this->expandByPoint(_v1.copy(sphere.center).sub(_toFarthestPoint));

    return *this;
}

bool Sphere::equals(const Sphere& sphere) const {

    return sphere.center.equals(this->center) && (sphere.radius == this->radius);
}

Sphere Sphere::clone() const {

    return Sphere().copy(*this);
}
