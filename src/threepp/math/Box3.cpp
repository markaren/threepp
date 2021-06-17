
#include "threepp/math/Box3.hpp"

#include <limits>

using namespace threepp;

namespace {

    constexpr float Infinity = std::numeric_limits<float>::infinity();

}

std::array<Vector3, 9> Box3::_points = {
        Vector3(), Vector3(), Vector3(),
        Vector3(), Vector3(), Vector3(),
        Vector3(), Vector3(), Vector3()};

Vector3 Box3::_vector = Vector3();

Box3 Box3::_box = Box3();

Vector3 Box3::_v0 = Vector3();
Vector3 Box3::_v1 = Vector3();
Vector3 Box3::_v2 = Vector3();

Vector3 Box3::_f0 = Vector3();
Vector3 Box3::_f1 = Vector3();
Vector3 Box3::_f2 = Vector3();

Vector3 Box3::_center = Vector3();
Vector3 Box3::_extents = Vector3();
Vector3 Box3::_triangleNormal = Vector3();
Vector3 Box3::_testAxis = Vector3();

Box3::Box3() : min_(Vector3(+Infinity, +Infinity, +Infinity)), max_(Vector3(-Infinity, -Infinity, -Infinity)) {}

Box3::Box3(Vector3 min, Vector3 max) : min_(min), max_(max) {}

Box3 &Box3::set(const Vector3 &min, const Vector3 &max) {

    this->min_ = min;
    this->max_ = max;

    return *this;
}

Box3 &Box3::setFromPoints(const std::vector<Vector3> &points) {

    this->makeEmpty();

    for (auto &point : points) {

        this->expandByPoint(point);
    }

    return *this;
}

Box3 &Box3::setFromCenterAndSize(const Vector3 &center, const Vector3 &size) {

    const auto halfSize = _vector.copy(size).multiply(0.5f);

    this->min_.copy(center).sub(halfSize);
    this->max_.copy(center).add(halfSize);

    return *this;
}

Box3 &Box3::copy(const Box3 &box) {

    this->min_.copy(box.min_);
    this->max_.copy(box.max_);

    return *this;
}

Box3 &Box3::makeEmpty() {

    this->min_.x = this->min_.y = this->min_.z = +Infinity;
    this->max_.x = this->max_.y = this->max_.z = -Infinity;

    return *this;
}
bool Box3::isEmpty() const {

    // this is a more robust check for empty than ( volume <= 0 ) because volume can get positive with two negative axes

    return (this->max_.x < this->min_.x) || (this->max_.y < this->min_.y) || (this->max_.z < this->min_.z);
}

void Box3::getCenter(Vector3 &target) const {

    this->isEmpty() ? target.set(0, 0, 0) : target.addVectors(this->min_, this->max_).multiply(0.5f);
}

void Box3::getSize(Vector3 &target) const {

    this->isEmpty() ? target.set(0, 0, 0) : target.subVectors(this->max_, this->min_);
}

Box3 &Box3::expandByPoint(const Vector3 &point) {

    this->min_.min(point);
    this->max_.max(point);

    return *this;
}

Box3 &Box3::expandByVector(const Vector3 &vector) {

    this->min_.sub(vector);
    this->max_.add(vector);

    return *this;
}

Box3 &Box3::expandByScalar(float scalar) {

    this->min_.add(-scalar);
    this->max_.add(scalar);

    return *this;
}

bool Box3::containsPoint(const Vector3 &point) const {

    return point.x < this->min_.x || point.x > this->max_.x ||
                           point.y < this->min_.y || point.y > this->max_.y ||
                           point.z < this->min_.z || point.z > this->max_.z
                   ? false
                   : true;
}

bool Box3::containsBox(const Box3 &box) const {

    return this->min_.x <= box.min_.x && box.max_.x <= this->max_.x &&
           this->min_.y <= box.min_.y && box.max_.y <= this->max_.y &&
           this->min_.z <= box.min_.z && box.max_.z <= this->max_.z;
}

void Box3::getParameter(const Vector3 &point, Vector3 &target) const {

    // This can potentially have a divide by zero if the box
    // has a size dimension of 0.

    target.set(
            (point.x - this->min_.x) / (this->max_.x - this->min_.x),
            (point.y - this->min_.y) / (this->max_.y - this->min_.y),
            (point.z - this->min_.z) / (this->max_.z - this->min_.z));
}

bool Box3::intersectsBox(const Box3 &box) const {

    // using 6 splitting planes to rule out intersections.
    return box.max_.x < this->min_.x || box.min_.x > this->max_.x ||
                           box.max_.y < this->min_.y || box.min_.y > this->max_.y ||
                           box.max_.z < this->min_.z || box.min_.z > this->max_.z
                   ? false
                   : true;
}

bool Box3::intersectsSphere(const Sphere &sphere) {

    // Find the point on the AABB closest to the sphere center.
    this->clampPoint( sphere.center(), _vector );

    // If that point is inside the sphere, the AABB and sphere intersect.
    const auto radius = sphere.radius();
    return _vector.distanceToSquared( sphere.center() ) <= (radius * radius);

}

bool Box3::intersectsPlane(const Plane &plane) {

    // We compute the minimum and maximum dot product values. If those values
    // are on the same side (back or front) of the plane, then there is no intersection.

    float min, max;

    if (plane.normal.x > 0) {

        min = plane.normal.x * this->min_.x;
        max = plane.normal.x * this->max_.x;

    } else {

        min = plane.normal.x * this->max_.x;
        max = plane.normal.x * this->min_.x;
    }

    if (plane.normal.y > 0) {

        min += plane.normal.y * this->min_.y;
        max += plane.normal.y * this->max_.y;

    } else {

        min += plane.normal.y * this->max_.y;
        max += plane.normal.y * this->min_.y;
    }

    if (plane.normal.z > 0) {

        min += plane.normal.z * this->min_.z;
        max += plane.normal.z * this->max_.z;

    } else {

        min += plane.normal.z * this->max_.z;
        max += plane.normal.z * this->min_.z;
    }

    return (min <= -plane.constant && max >= -plane.constant);
}

bool Box3::intersectsTriangle(const Triangle &triangle) {

    if (this->isEmpty()) {

        return false;
    }

    // compute box center and extents
    this->getCenter(_center);
    _extents.subVectors(this->max_, _center);

    // translate triangle to aabb origin
    _v0.subVectors(triangle.a(), _center);
    _v1.subVectors(triangle.b(), _center);
    _v2.subVectors(triangle.c(), _center);

    // compute edge vectors for triangle
    _f0.subVectors(_v1, _v0);
    _f1.subVectors(_v2, _v1);
    _f2.subVectors(_v0, _v2);

    // test against axes that are given by cross product combinations of the edges of the triangle and the edges of the aabb
    // make an axis testing of each of the 3 sides of the aabb against each of the 3 sides of the triangle = 9 axis of separation
    // axis_ij = u_i x f_j (u0, u1, u2 = face normals of aabb = x,y,z axes vectors since aabb is axis aligned)
    std::vector<float> axes = {
            0, -_f0.z, _f0.y, 0, -_f1.z, _f1.y, 0, -_f2.z, _f2.y,
            _f0.z, 0, -_f0.x, _f1.z, 0, -_f1.x, _f2.z, 0, -_f2.x,
            -_f0.y, _f0.x, 0, -_f1.y, _f1.x, 0, -_f2.y, _f2.x, 0};
    if (!satForAxes(axes, _v0, _v1, _v2, _extents)) {

        return false;
    }

    // test 3 face normals from the aabb
    axes = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (!satForAxes(axes, _v0, _v1, _v2, _extents)) {

        return false;
    }

    // finally testing the face normal of the triangle
    // use already existing triangle edge vectors here
    _triangleNormal.crossVectors(_f0, _f1);
    axes = {_triangleNormal.x, _triangleNormal.y, _triangleNormal.z};

    return satForAxes(axes, _v0, _v1, _v2, _extents);
}


bool Box3::satForAxes(const std::vector<float> &axes, const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, const Vector3 &extents) {

    for (size_t i = 0, j = axes.size() - 3; i <= j; i += 3) {

        _testAxis.fromArray(axes, i);
        // project the aabb onto the seperating axis
        const auto r = extents.x * std::abs(_testAxis.x) + extents.y * std::abs(_testAxis.y) + extents.z * std::abs(_testAxis.z);
        // project all 3 vertices of the triangle onto the seperating axis
        const auto p0 = v0.dot(_testAxis);
        const auto p1 = v1.dot(_testAxis);
        const auto p2 = v2.dot(_testAxis);
        // actual test, basically see if either of the most extreme of the triangle points intersects r
        if (std::max(-std::max(p0, std::max(p1, p2)), std::min(p0, std::min(p1, p2))) > r) {

            // points of the projected triangle are outside the projected half-length of the aabb
            // the axis is seperating and we can exit
            return false;
        }
    }

    return true;
}

Vector3 &Box3::clampPoint(const Vector3 &point, Vector3 &target) const {

    return target.copy(point).clamp(this->min_, this->max_);
}

float Box3::distanceToPoint(const Vector3 &point) const {

    auto clampedPoint = _vector.copy(point).clamp(this->min_, this->max_);

    return clampedPoint.sub(point).length();
}
