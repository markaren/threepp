
#include "threepp/math/Frustum.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Sphere.hpp"

using namespace threepp;

namespace {

    Sphere _sphere;
    Vector3 _vector;

}

Frustum::Frustum(Plane p0, Plane p1, Plane p2, Plane p3, Plane p4, Plane p5)
    : planes_{p0, p1, p2, p3, p4, p5} {}

Frustum &Frustum::copy(const Frustum &frustum) {

    for (int i = 0; i < 6; i++) {

        planes_[i].copy(frustum.planes_[i]);
    }

    return *this;
}

Frustum &Frustum::setFromProjectionMatrix(const Matrix4 &m) {

    const auto &me = m.elements();
    const auto me0 = me[0], me1 = me[1], me2 = me[2], me3 = me[3];
    const auto me4 = me[4], me5 = me[5], me6 = me[6], me7 = me[7];
    const auto me8 = me[8], me9 = me[9], me10 = me[10], me11 = me[11];
    const auto me12 = me[12], me13 = me[13], me14 = me[14], me15 = me[15];

    planes_[0].setComponents(me3 - me0, me7 - me4, me11 - me8, me15 - me12).normalize();
    planes_[1].setComponents(me3 + me0, me7 + me4, me11 + me8, me15 + me12).normalize();
    planes_[2].setComponents(me3 + me1, me7 + me5, me11 + me9, me15 + me13).normalize();
    planes_[3].setComponents(me3 - me1, me7 - me5, me11 - me9, me15 - me13).normalize();
    planes_[4].setComponents(me3 - me2, me7 - me6, me11 - me10, me15 - me14).normalize();
    planes_[5].setComponents(me3 + me2, me7 + me6, me11 + me10, me15 + me14).normalize();

    return *this;
}

bool Frustum::intersectsObject(Object3D &object) {

    auto geometry = object.geometry();

    if (!geometry->boundingSphere) geometry->computeBoundingSphere();

    _sphere.copy(geometry->boundingSphere.value()).applyMatrix4(object.matrixWorld);

    return this->intersectsSphere(_sphere);
}

bool Frustum::intersectsSphere(const Sphere &sphere) {

    const auto &center = sphere.center;
    const auto negRadius = -sphere.radius;

    for (int i = 0; i < 6; i++) {

        const auto distance = planes_[i].distanceToPoint(center);

        if (distance < negRadius) {

            return false;
        }
    }

    return true;
}

bool Frustum::intersectsBox(const Box3 &box) {

    for (int i = 0; i < 6; i++) {

        const auto &plane = planes_[i];

        // corner at max distance

        _vector.x = plane.normal.x > 0 ? box.max().x : box.min().x;
        _vector.y = plane.normal.y > 0 ? box.max().y : box.min().y;
        _vector.z = plane.normal.z > 0 ? box.max().z : box.min().z;

        if (plane.distanceToPoint(_vector) < 0) {

            return false;
        }
    }

    return true;
}

bool Frustum::containsPoint(const Vector3 &point) {

    for (int i = 0; i < 6; i++) {

        if (planes_[i].distanceToPoint(point) < 0) {

            return false;
        }
    }

    return true;
}
