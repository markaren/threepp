
#include "threepp/math/Box3.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/objects/InstancedMesh.hpp"

#include "threepp/math/Triangle.hpp"


using namespace threepp;

namespace {

    Vector3 _vector;

    Vector3 _v0;
    Vector3 _v1;
    Vector3 _v2;

    Vector3 _f0;
    Vector3 _f1;
    Vector3 _f2;

    Vector3 _center;
    Vector3 _extents;

    Vector3 _triangleNormal;
    Vector3 _testAxis;

    std::array<Vector3, 8> _points;


    bool satForAxes(const std::vector<float>& axes, const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& extents) {

        for (unsigned i = 0, j = axes.size() - 3; i <= j; i += 3) {

            _testAxis.fromArray(axes, i);
            // project the aabb onto the separating axis
            const float r = extents.x * std::abs(_testAxis.x) + extents.y * std::abs(_testAxis.y) + extents.z * std::abs(_testAxis.z);
            // project all 3 vertices of the triangle onto the seperating axis
            const float p0 = v0.dot(_testAxis);
            const float p1 = v1.dot(_testAxis);
            const float p2 = v2.dot(_testAxis);
            // actual test, basically see if either of the most extreme of the triangle points intersects r
            if (std::max(-std::max(p0, std::max(p1, p2)), std::min(p0, std::min(p1, p2))) > r) {

                // points of the projected triangle are outside the projected half-length of the aabb
                // the axis is separating and we can exit
                return false;
            }
        }

        return true;
    }

}// namespace

Box3::Box3()
    : min_(Vector3(+Infinity<float>, +Infinity<float>, +Infinity<float>)),
      max_(Vector3(-Infinity<float>, -Infinity<float>, -Infinity<float>)) {}

Box3::Box3(const Vector3& min, const Vector3& max)
    : min_(min), max_(max) {}

const Vector3& Box3::min() const {

    return min_;
}

const Vector3& Box3::max() const {

    return max_;
}

Box3& Box3::set(const Vector3& min, const Vector3& max) {

    this->min_.copy(min);
    this->max_.copy(max);

    return *this;
}

Box3& Box3::set(float minX, float minY, float minZ, float maxX, float maxY, float maxZ) {

    this->min_.set(minX, minY, minZ);
    this->max_.set(maxX, maxY, maxZ);

    return *this;
}

Box3& Box3::setFromCenterAndSize(const Vector3& center, const Vector3& size) {

    const auto halfSize = _vector.copy(size).multiplyScalar(0.5f);

    this->min_.copy(center).sub(halfSize);
    this->max_.copy(center).add(halfSize);

    return *this;
}

Box3& Box3::setFromObject(Object3D& object, bool precise) {

    this->makeEmpty();

    return this->expandByObject(object, precise);
}

Box3 Box3::clone() const {

    return Box3().copy(*this);
}

Box3& Box3::copy(const Box3& box) {

    this->min_.copy(box.min_);
    this->max_.copy(box.max_);

    return *this;
}

Box3& Box3::makeEmpty() {

    this->min_.x = this->min_.y = this->min_.z = +Infinity<float>;
    this->max_.x = this->max_.y = this->max_.z = -Infinity<float>;

    return *this;
}

bool Box3::isEmpty() const {

    // this is a more robust check for empty than ( volume <= 0 ) because volume can get positive with two negative axes

    return (this->max_.x < this->min_.x) || (this->max_.y < this->min_.y) || (this->max_.z < this->min_.z);
}

void Box3::getCenter(Vector3& target) const {

    this->isEmpty() ? target.set(0, 0, 0) : target.addVectors(this->min_, this->max_).multiplyScalar(0.5f);
}

Vector3 Box3::getCenter() const {

    Vector3 target;
    getCenter(target);

    return target;
}

void Box3::getSize(Vector3& target) const {

    this->isEmpty() ? target.set(0, 0, 0) : target.subVectors(this->max_, this->min_);
}

Vector3 Box3::getSize() const {

    Vector3 target;
    getSize(target);

    return target;
}

Box3& Box3::expandByPoint(const Vector3& point) {

    this->min_.min(point);
    this->max_.max(point);

    return *this;
}

Box3& Box3::expandByVector(const Vector3& vector) {

    this->min_.sub(vector);
    this->max_.add(vector);

    return *this;
}

Box3& Box3::expandByScalar(float scalar) {

    this->min_.addScalar(-scalar);
    this->max_.addScalar(scalar);

    return *this;
}

bool Box3::containsPoint(const Vector3& point) const {

    return point.x < this->min_.x || point.x > this->max_.x ||
                           point.y < this->min_.y || point.y > this->max_.y ||
                           point.z < this->min_.z || point.z > this->max_.z
                   ? false
                   : true;
}

Box3& Box3::expandByObject(Object3D& object, bool presice) {

    // Computes the world-axis-aligned bounding box of an object (including its children),
    // accounting for both the object's, and children's, world transforms

    object.updateWorldMatrix(false, false);

    if (auto instancedMesh = object.as<InstancedMesh>()) {

        Box3 _box{};

        if (!instancedMesh->boundingBox) instancedMesh->computeBoundingBox();

        _box.copy(*instancedMesh->boundingBox);

        _box.applyMatrix4(*object.matrixWorld);

        this->union_(_box);

    } else {

        const auto geometry = object.geometry();

        if (geometry) {

        if (presice && geometry->getAttributes().contains("position")) {

                const auto position = geometry->getAttribute<float>("position");
                for (unsigned i = 0, l = position->count(); i < l; i++) {

                    position->setFromBufferAttribute(_vector, i);
                    _vector.applyMatrix4(*object.matrixWorld);

                    this->expandByPoint(_vector);
                }

            } else {

                if (!geometry->boundingBox) {

                    geometry->computeBoundingBox();
                }

                Box3 _box{};

                _box.copy(geometry->boundingBox.value());
                _box.applyMatrix4(*object.matrixWorld);

                this->union_(_box);
            }
        }
    }

    for (auto& child : object.children) {

        this->expandByObject(*child, presice);
    }

    return *this;
}

bool Box3::containsBox(const Box3& box) const {

    return this->min_.x <= box.min_.x && box.max_.x <= this->max_.x &&
           this->min_.y <= box.min_.y && box.max_.y <= this->max_.y &&
           this->min_.z <= box.min_.z && box.max_.z <= this->max_.z;
}

void Box3::getParameter(const Vector3& point, Vector3& target) const {

    // This can potentially have a divide by zero if the box
    // has a size dimension of 0.

    target.set(
            (point.x - this->min_.x) / (this->max_.x - this->min_.x),
            (point.y - this->min_.y) / (this->max_.y - this->min_.y),
            (point.z - this->min_.z) / (this->max_.z - this->min_.z));
}

bool Box3::intersectsBox(const Box3& box) const {

    // using 6 splitting planes to rule out intersections.
    return box.max_.x < this->min_.x || box.min_.x > this->max_.x ||
                           box.max_.y < this->min_.y || box.min_.y > this->max_.y ||
                           box.max_.z < this->min_.z || box.min_.z > this->max_.z
                   ? false
                   : true;
}

bool Box3::intersectsSphere(const Sphere& sphere) const {

    // Find the point on the AABB closest to the sphere center.
    this->clampPoint(sphere.center, _vector);

    // If that point is inside the sphere, the AABB and sphere intersect.
    const float radius = sphere.radius;
    return _vector.distanceToSquared(sphere.center) <= (radius * radius);
}

bool Box3::intersectsPlane(const Plane& plane) const {

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

bool Box3::intersectsTriangle(const Triangle& triangle) const {

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

Vector3& Box3::clampPoint(const Vector3& point, Vector3& target) const {

    return target.copy(point).clamp(this->min_, this->max_);
}

float Box3::distanceToPoint(const Vector3& point) const {

    auto clampedPoint = _vector.copy(point).clamp(this->min_, this->max_);

    return clampedPoint.sub(point).length();
}

void Box3::getBoundingSphere(Sphere& target) const {

    this->getCenter(target.center);

    this->getSize(_vector);
    target.radius = _vector.length() * 0.5f;
}

Box3& Box3::intersect(const Box3& box) {

    this->min_.max(box.min_);
    this->max_.min(box.max_);

    // ensure that if there is no overlap, the result is fully empty, not slightly empty with non-inf/+inf values that will cause subsequence intersects to erroneously return valid values.
    if (this->isEmpty()) this->makeEmpty();

    return *this;
}

Box3& Box3::union_(const Box3& box) {

    this->min_.min(box.min_);
    this->max_.max(box.max_);

    return *this;
}

Box3& Box3::applyMatrix4(const Matrix4& matrix) {

    // transform of empty box is an empty box.
    if (this->isEmpty()) return *this;

    // NOTE: I am using a binary pattern to specify all 2^3 combinations below
    _points[0].set(this->min_.x, this->min_.y, this->min_.z).applyMatrix4(matrix);// 000
    _points[1].set(this->min_.x, this->min_.y, this->max_.z).applyMatrix4(matrix);// 001
    _points[2].set(this->min_.x, this->max_.y, this->min_.z).applyMatrix4(matrix);// 010
    _points[3].set(this->min_.x, this->max_.y, this->max_.z).applyMatrix4(matrix);// 011
    _points[4].set(this->max_.x, this->min_.y, this->min_.z).applyMatrix4(matrix);// 100
    _points[5].set(this->max_.x, this->min_.y, this->max_.z).applyMatrix4(matrix);// 101
    _points[6].set(this->max_.x, this->max_.y, this->min_.z).applyMatrix4(matrix);// 110
    _points[7].set(this->max_.x, this->max_.y, this->max_.z).applyMatrix4(matrix);// 111

    this->setFromPoints(_points);

    return *this;
}

Box3& Box3::translate(const Vector3& offset) {

    this->min_.add(offset);
    this->max_.add(offset);

    return *this;
}

bool Box3::equals(const Box3& box) const {

    return box.min_.equals(this->min_) && box.max_.equals(this->max_);
}

bool Box3::operator==(const Box3& other) const {

    return equals(other);
}
