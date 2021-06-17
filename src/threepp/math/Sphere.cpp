
#include "threepp/math/Sphere.hpp"

#include "threepp/math/Box3.hpp"

using namespace threepp;

Box3 Sphere::_box = Box3();
Vector3 Sphere::_v1 = Vector3();

Sphere::Sphere(Vector3 center, float radius) : center_(center), radius_(radius) {}

float Sphere::radius() const {
    return radius_;
}

const Vector3 &Sphere::center() const {
    return center_;
}

Sphere &Sphere::set(const Vector3 &center, float radius) {

    this->center_ = (center);
    this->radius_ = radius;

    return *this;
}

Sphere &Sphere::setFromPoints(const std::vector<Vector3> &points, Vector3 *optionalCenter) {

    if (optionalCenter) {

        center_ = (*optionalCenter);

    } else {

        _box.setFromPoints(points).getCenter(center_);
    }

    float maxRadiusSq = 0;

    for (auto &point : points) {

        maxRadiusSq = std::max(maxRadiusSq, center_.distanceToSquared(point));
    }

    this->radius_ = std::sqrt(maxRadiusSq);

    return *this;
}
