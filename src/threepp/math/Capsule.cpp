
#include "threepp/math/Capsule.hpp"

#include "threepp/math/Box3.hpp"
#include "threepp/math/Line3.hpp"

#include <cmath>

using namespace threepp;

namespace {

    Vector3 _v1;
    Vector3 _v2;
    Vector3 _v3;

    const float EPS = 1e-10;

}// namespace

Capsule::Capsule(const Vector3& start, const Vector3& end, float radius)
    : start(start), end(end), radius(radius) {}

Capsule Capsule::clone() const {

    return Capsule(start, end, radius);
}

void Capsule::set(const Vector3& start, const Vector3& end, float radius) {

    this->start.copy(start);
    this->end.copy(end);
    this->radius = radius;
}

void Capsule::copy(const Capsule& capsule) {

    this->start.copy(capsule.start);
    this->end.copy(capsule.end);
    this->radius = capsule.radius;
}

void Capsule::getCenter(Vector3& target) const {

    target.copy(this->end).add(this->start).multiplyScalar(0.5f);
}

void Capsule::translate(const Vector3& v) {

    this->start.add(v);
    this->end.add(v);
}

bool Capsule::checkAABBAxis(float p1x, float p1y, float p2x, float p2y, float minx, float maxx, float miny, float maxy, float radius) {

    return (
            (minx - p1x < radius || minx - p2x < radius) &&
            (p1x - maxx < radius || p2x - maxx < radius) &&
            (miny - p1y < radius || miny - p2y < radius) &&
            (p1y - maxy < radius || p2y - maxy < radius));
}

bool Capsule::intersectsBox(const Box3& box) const {

    return (
            this->checkAABBAxis(
                    this->start.x, this->start.y, this->end.x, this->end.y,
                    box.min().x, box.max().x, box.min().y, box.max().y,
                    this->radius) &&
            this->checkAABBAxis(
                    this->start.x, this->start.z, this->end.x, this->end.z,
                    box.min().x, box.max().x, box.min().z, box.max().z,
                    this->radius) &&
            this->checkAABBAxis(
                    this->start.y, this->start.z, this->end.y, this->end.z,
                    box.min().y, box.max().y, box.min().z, box.max().z,
                    this->radius));
}

std::pair<Vector3, Vector3> Capsule::lineLineMinimumPoints(const Line3& line1, const Line3& line2) const {

    auto r = _v1.copy(line1.end()).sub(line1.start());
    auto s = _v2.copy(line2.end()).sub(line2.start());
    const auto w = _v3.copy(line2.start()).sub(line1.start());

    const auto a = r.dot(s),
               b = r.dot(r),
               c = s.dot(s),
               d = s.dot(w),
               e = r.dot(w);

    float t1, t2;
    const auto divisor = b * c - a * a;

    if (std::abs(divisor) < EPS) {

        const auto d1 = -d / c;
        const auto d2 = (a - d) / c;

        if (std::abs(d1 - 0.5) < std::abs(d2 - 0.5)) {

            t1 = 0;
            t2 = d1;

        } else {

            t1 = 1;
            t2 = d2;
        }

    } else {

        t1 = (d * a + e * c) / divisor;
        t2 = (t1 * a - d) / c;
    }

    t2 = std::max(0.f, std::min(1.f, t2));
    t1 = std::max(0.f, std::min(1.f, t1));

    auto point1 = r.multiplyScalar(t1).add(line1.start());
    auto point2 = s.multiplyScalar(t2).add(line2.start());

    return {point1, point2};
}
