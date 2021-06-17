
#include "threepp/math/Sphere.hpp"

#include "threepp/math/Box3.hpp"

using namespace threepp;

Box3 Sphere::_box = Box3();
Vector3 Sphere::_v1 = Vector3();

Sphere::Sphere(Vector3 center, float radius) : center(center), radius(radius) {}

Sphere &Sphere::set(const Vector3 &center, float radius) {

    this->center = (center);
    this->radius = radius;

    return *this;
}

Sphere &Sphere::setFromPoints(const std::vector<Vector3> &points, Vector3 *optionalCenter) {

    if (optionalCenter) {

        center.copy(*optionalCenter);

    } else {

        _box.setFromPoints(points).getCenter(center);
    }

    float maxRadiusSq = 0;

    for (auto &point : points) {

        maxRadiusSq = std::max(maxRadiusSq, center.distanceToSquared(point));
    }

    this->radius = std::sqrt(maxRadiusSq);

    return *this;
}
