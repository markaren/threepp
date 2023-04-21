
#include "threepp/math/Line3.hpp"

#include "threepp/math/Matrix4.hpp"

#include <algorithm>

using namespace threepp;

Line3::Line3(const Vector3& start, const Vector3& end)
    : start_(start), end_(end) {}

const Vector3& Line3::start() const {

    return start_;
}

const Vector3& Line3::end() const {

    return end_;
}

Line3 Line3::set(const Vector3& start, const Vector3& end) {

    this->start_.copy(start);
    this->end_.copy(end);

    return *this;
}

Line3& Line3::copy(const Line3& line) {

    this->start_.copy(line.start_);
    this->end_.copy(line.end_);

    return *this;
}

void Line3::getCenter(Vector3& target) const {

    target.addVectors(this->start_, this->end_).multiplyScalar(0.5f);
}

void Line3::delta(Vector3& target) const {

    target.subVectors(this->end_, this->start_);
}

float Line3::distanceSq() const {

    return this->start_.distanceToSquared(this->end_);
}

float Line3::distance() const {

    return this->start_.distanceTo(this->end_);
}

void Line3::at(float t, Vector3& target) const {

    this->delta(target);
    target.multiplyScalar(t).add(this->start_);
}

float Line3::closestPointToPointParameter(const Vector3& point, bool clampToLine) {

    Vector3 _startP;
    Vector3 _startEnd;

    _startP.subVectors(point, this->start_);
    _startEnd.subVectors(this->end_, this->start_);

    const float startEnd2 = _startEnd.dot(_startEnd);
    const float startEnd_startP = _startEnd.dot(_startP);

    float t = startEnd_startP / startEnd2;

    if (clampToLine) {

        t = std::clamp(t, 0.f, 1.f);
    }

    return t;
}

void Line3::closestPointToPoint(const Vector3& point, bool clampToLine, Vector3& target) {

    const auto t = this->closestPointToPointParameter(point, clampToLine);

    this->delta(target);
    target.multiplyScalar(t).add(this->start_);
}

Line3& Line3::applyMatrix4(const Matrix4& matrix) {

    this->start_.applyMatrix4(matrix);
    this->end_.applyMatrix4(matrix);

    return *this;
}
