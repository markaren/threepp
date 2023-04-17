
#include "threepp/math/Ray.hpp"

#include "threepp/math/Box3.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Sphere.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

namespace {

    Vector3 _vector;

    Vector3 _segCenter;
    Vector3 _segDir;
    Vector3 _diff;

    Vector3 _edge1;
    Vector3 _edge2;
    Vector3 _normal;

}// namespace

Ray::Ray(const Vector3& origin, const Vector3& direction): origin(origin), direction(direction) {}

Ray& Ray::set(const Vector3& origin, const Vector3& direction) {

    this->origin.copy(origin);
    this->direction.copy(direction);

    return *this;
}

Ray& Ray::copy(const Ray& ray) {

    this->origin.copy(ray.origin);
    this->direction.copy(ray.direction);

    return *this;
}

Vector3& Ray::at(float t, Vector3& target) const {

    return target.copy(this->direction).multiplyScalar(t).add(this->origin);
}

Ray& Ray::lookAt(const Vector3& v) {

    this->direction.copy(v).sub(this->origin).normalize();

    return *this;
}

Ray& Ray::recast(float t) {

    this->at(t, this->origin);

    return *this;
}

void Ray::closestPointToPoint(const Vector3& point, Vector3& target) const {

    target.subVectors(point, this->origin);

    const auto directionDistance = target.dot(this->direction);

    if (directionDistance < 0) {

        target.copy(this->origin);

    } else {

        target.copy(this->direction).multiplyScalar(directionDistance).add(this->origin);
    }
}

float Ray::distanceToPoint(const Vector3& point) const {

    return std::sqrt(this->distanceSqToPoint(point));
}

float Ray::distanceSqToPoint(const Vector3& point) const {

    const auto directionDistance = _vector.subVectors(point, this->origin).dot(this->direction);

    // point behind the ray

    if (directionDistance < 0) {

        return this->origin.distanceToSquared(point);
    }


    _vector.copy(this->direction).multiplyScalar(directionDistance).add(this->origin);

    return _vector.distanceToSquared(point);
}

float Ray::distanceSqToSegment(const Vector3& v0, const Vector3& v1, Vector3* optionalPointOnRay, Vector3* optionalPointOnSegment) const {

    // from http://www.geometrictools.com/GTEngine/Include/Mathematics/GteDistRaySegment.h
    // It returns the min distance between the ray and the segment
    // defined by v0 and v1
    // It can also set two optional targets :
    // - The closest point on the ray
    // - The closest point on the segment

    _segCenter.copy(v0).add(v1).multiplyScalar(0.5f);
    _segDir.copy(v1).sub(v0).normalize();
    _diff.copy(this->origin).sub(_segCenter);

    const float segExtent = v0.distanceTo(v1) * 0.5f;
    const float a01 = -this->direction.dot(_segDir);
    const float b0 = _diff.dot(this->direction);
    const float b1 = -_diff.dot(_segDir);
    const float c = _diff.lengthSq();
    const float det = std::abs(1 - a01 * a01);
    float s0, s1, sqrDist, extDet;

    if (det > 0) {

        // The ray and segment are not parallel.

        s0 = a01 * b1 - b0;
        s1 = a01 * b0 - b1;
        extDet = segExtent * det;

        if (s0 >= 0) {

            if (s1 >= -extDet) {

                if (s1 <= extDet) {

                    // region 0
                    // Minimum at interior points of ray and segment.

                    const float invDet = 1.f / det;
                    s0 *= invDet;
                    s1 *= invDet;
                    sqrDist = s0 * (s0 + a01 * s1 + 2 * b0) + s1 * (a01 * s0 + s1 + 2 * b1) + c;

                } else {

                    // region 1

                    s1 = segExtent;
                    s0 = std::max(0.f, -(a01 * s1 + b0));
                    sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
                }

            } else {

                // region 5

                s1 = -segExtent;
                s0 = std::max(0.f, -(a01 * s1 + b0));
                sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
            }

        } else {

            if (s1 <= -extDet) {

                // region 4

                s0 = std::max(0.f, -(-a01 * segExtent + b0));
                s1 = (s0 > 0) ? -segExtent : std::min(std::max(-segExtent, -b1), segExtent);
                sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;

            } else if (s1 <= extDet) {

                // region 3

                s0 = 0;
                s1 = std::min(std::max(-segExtent, -b1), segExtent);
                sqrDist = s1 * (s1 + 2 * b1) + c;

            } else {

                // region 2

                s0 = std::max(0.f, -(a01 * segExtent + b0));
                s1 = (s0 > 0) ? segExtent : std::min(std::max(-segExtent, -b1), segExtent);
                sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
            }
        }

    } else {

        // Ray and segment are parallel.

        s1 = (a01 > 0) ? -segExtent : segExtent;
        s0 = std::max(0.f, -(a01 * s1 + b0));
        sqrDist = -s0 * s0 + s1 * (s1 + 2 * b1) + c;
    }

    if (optionalPointOnRay) {

        optionalPointOnRay->copy(this->origin).addScaledVector(this->direction, s0);
    }

    if (optionalPointOnSegment) {

        optionalPointOnSegment->copy(_segCenter).addScaledVector(_segDir, s1);
    }

    return sqrDist;
}

void Ray::intersectSphere(const Sphere& sphere, Vector3& target) const {

    _vector.subVectors(sphere.center, this->origin);
    const auto tca = _vector.dot(this->direction);
    const auto d2 = _vector.dot(_vector) - tca * tca;
    const auto radius2 = sphere.radius * sphere.radius;

    if (d2 > radius2) {
        target.set(NAN, NAN, NAN);
        return;
    }

    const auto thc = std::sqrt(radius2 - d2);

    // t0 = first intersect point - entrance on front of sphere
    const auto t0 = tca - thc;

    // t1 = second intersect point - exit point on back of sphere
    const auto t1 = tca + thc;

    // test to see if both t0 and t1 are behind the ray - if so, return null
    if (t0 < 0 && t1 < 0) {
        target.set(NAN, NAN, NAN);
        return;
    }

    // test to see if t0 is behind the ray:
    // if it is, the ray is inside the sphere, so return the second exit point scaled by t1,
    // in order to always return an intersect point that is in front of the ray.
    if (t0 < 0) {

        this->at(t1, target);

    } else {

        // else t0 is in front of the ray, so return the first collision point scaled by t0
        this->at(t0, target);
    }
}
bool Ray::intersectsSphere(const Sphere& sphere) const {

    return this->distanceSqToPoint(sphere.center) <= (sphere.radius * sphere.radius);
}

float Ray::distanceToPlane(const Plane& plane) const {

    const auto denominator = plane.normal.dot(this->direction);

    if (denominator == 0) {

        // line is coplanar, return origin
        if (plane.distanceToPoint(this->origin) == 0) {

            return 0;
        }

        // Null is preferable to undefined since undefined means.... it is undefined

        return NAN;
    }

    const auto t = -(this->origin.dot(plane.normal) + plane.constant) / denominator;

    // Return if the ray never intersects the plane

    return t >= 0 ? t : NAN;
}

void Ray::intersectPlane(const Plane& plane, Vector3& target) const {

    const auto t = this->distanceToPlane(plane);

    if (std::isnan(t)) {

        target.set(NAN, NAN, NAN);
        return;
    }

    this->at(t, target);
}

bool Ray::intersectsPlane(const Plane& plane) const {

    // check if the ray lies on the plane first

    const auto distToPoint = plane.distanceToPoint(this->origin);

    if (distToPoint == 0) {

        return true;
    }

    const auto denominator = plane.normal.dot(this->direction);

    if (denominator * distToPoint < 0) {

        return true;
    }

    // ray origin is behind the plane (and is pointing behind it)

    return false;
}

void Ray::intersectBox(const Box3& box, Vector3& target) const {

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    const auto invdirx = 1.f / this->direction.x,
               invdiry = 1.f / this->direction.y,
               invdirz = 1.f / this->direction.z;

    if (invdirx >= 0) {

        tmin = (box.min().x - origin.x) * invdirx;
        tmax = (box.max().x - origin.x) * invdirx;

    } else {

        tmin = (box.max().x - origin.x) * invdirx;
        tmax = (box.min().x - origin.x) * invdirx;
    }

    if (invdiry >= 0) {

        tymin = (box.min().y - origin.y) * invdiry;
        tymax = (box.max().y - origin.y) * invdiry;

    } else {

        tymin = (box.max().y - origin.y) * invdiry;
        tymax = (box.min().y - origin.y) * invdiry;
    }

    if ((tmin > tymax) || (tymin > tmax)) {

        target.set(NAN, NAN, NAN);

        return;
    };

    // These lines also handle the case where tmin or tmax is NaN
    // (result of 0 * Infinity). x !== x returns true if x is NaN

    if (tymin > tmin || tmin != tmin) tmin = tymin;

    if (tymax < tmax || tmax != tmax) tmax = tymax;

    if (invdirz >= 0) {

        tzmin = (box.min().z - origin.z) * invdirz;
        tzmax = (box.max().z - origin.z) * invdirz;

    } else {

        tzmin = (box.max().z - origin.z) * invdirz;
        tzmax = (box.min().z - origin.z) * invdirz;
    }

    if ((tmin > tzmax) || (tzmin > tmax)) {

        target.set(NAN, NAN, NAN);
        return;
    }

    if (tzmin > tmin || tmin != tmin) tmin = tzmin;

    if (tzmax < tmax || tmax != tmax) tmax = tzmax;

    //return point closest to the ray (positive side)

    if (tmax < 0) {

        target.set(NAN, NAN, NAN);
        return;
    }

    this->at(tmin >= 0 ? tmin : tmax, target);
}

bool Ray::intersectsBox(const Box3& box) const {

    this->intersectBox(box, _vector);

    return !_vector.isNan();
}

std::optional<Vector3> Ray::intersectTriangle(const Vector3& a, const Vector3& b, const Vector3& c, bool backfaceCulling, Vector3& target) const {

    // Compute the offset origin, edges, and normal.

    // from http://www.geometrictools.com/GTEngine/Include/Mathematics/GteIntrRay3Triangle3.h

    _edge1.subVectors(b, a);
    _edge2.subVectors(c, a);
    _normal.crossVectors(_edge1, _edge2);

    // Solve Q + t*D = b1*E1 + b2*E2 (Q = kDiff, D = ray direction,
    // E1 = kEdge1, E2 = kEdge2, N = Cross(E1,E2)) by
    //   |Dot(D,N)|*b1 = sign(Dot(D,N))*Dot(D,Cross(Q,E2))
    //   |Dot(D,N)|*b2 = sign(Dot(D,N))*Dot(D,Cross(E1,Q))
    //   |Dot(D,N)|*t = -sign(Dot(D,N))*Dot(Q,N)
    float DdN = this->direction.dot(_normal);
    float sign;

    if (DdN > 0) {

        if (backfaceCulling) {

            target.set(NAN, NAN, NAN);
            return std::nullopt;
        }
        sign = 1.f;

    } else if (DdN < 0) {

        sign = -1.f;
        DdN = -DdN;

    } else {

        target.set(NAN, NAN, NAN);
        return std::nullopt;
    }

    _diff.subVectors(this->origin, a);
    const float DdQxE2 = sign * this->direction.dot(_edge2.crossVectors(_diff, _edge2));

    // b1 < 0, no intersection
    if (DdQxE2 < 0) {

        target.set(NAN, NAN, NAN);
        return std::nullopt;
    }

    const float DdE1xQ = sign * this->direction.dot(_edge1.cross(_diff));

    // b2 < 0, no intersection
    if (DdE1xQ < 0) {

        target.set(NAN, NAN, NAN);
        return std::nullopt;
    }

    // b1+b2 > 1, no intersection
    if (DdQxE2 + DdE1xQ > DdN) {

        target.set(NAN, NAN, NAN);
        return std::nullopt;
    }

    // Line intersects triangle, check if ray does.
    const float QdN = -sign * _diff.dot(_normal);

    // t < 0, no intersection
    if (QdN < 0) {

        target.set(NAN, NAN, NAN);
        return std::nullopt;
    }

    // Ray intersects triangle.
    return this->at(QdN / DdN, target);
}

Ray& Ray::applyMatrix4(const Matrix4& matrix4) {

    this->origin.applyMatrix4(matrix4);
    this->direction.transformDirection(matrix4);

    return *this;
}
