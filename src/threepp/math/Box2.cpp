
#include "threepp/math/Box2.hpp"

#include "threepp/math/infinity.hpp"

using namespace threepp;

namespace {

    Vector2 _vector;

}

Box2::Box2()
    : min_(+Infinity<float>, +Infinity<float>),
      max_(-Infinity<float>, -Infinity<float>) {}

Box2::Box2(const Vector2& min, const Vector2& max)
    : min_(min), max_(max) {}

const Vector2& Box2::getMin() const {
    return min_;
}

const Vector2& Box2::getMax() const {
    return max_;
}

Box2& Box2::set(const Vector2& min, const Vector2& max) {

    this->min_ = min;
    this->max_ = max;

    return *this;
}

Box2& Box2::setFromPoints(const std::vector<Vector2>& points) {

    this->makeEmpty();

    for (const auto& point : points) {

        this->expandByPoint(point);
    }

    return *this;
}

Box2& Box2::setFromCenterAndSize(const Vector2& center, const Vector2& size) {

    const auto halfSize = _vector.copy(size).multiplyScalar(0.5);
    this->min_.copy(center).sub(halfSize);
    this->max_.copy(center).add(halfSize);

    return *this;
}

Box2& Box2::copy(const Box2& box) {

    this->min_.copy(box.min_);
    this->max_.copy(box.max_);

    return *this;
}

Box2& Box2::makeEmpty() {
    this->min_.x = +Infinity<float>;
    this->min_.y = +Infinity<float>;
    this->max_.x = -Infinity<float>;
    this->max_.y = -Infinity<float>;

    return *this;
}

bool Box2::isEmpty() const {
    // this is a more robust check for empty than ( volume <= 0 ) because volume can get positive with two negative axes

    return (this->max_.x < this->min_.x) || (this->max_.y < this->min_.y);
}

void Box2::getCenter(Vector2& target) const {

    this->isEmpty() ? target.set(0, 0) : target.addVectors(this->min_, this->max_).multiplyScalar(0.5f);
}

void Box2::getSize(Vector2& target) const {

    this->isEmpty() ? target.set(0, 0) : target.subVectors(this->max_, this->min_);
}

Box2& Box2::expandByPoint(const Vector2& point) {

    this->min_.min(point);
    this->max_.max(point);

    return *this;
}

Box2& Box2::expandByVector(const Vector2& vector) {

    this->min_.sub(vector);
    this->max_.add(vector);

    return *this;
}

Box2& Box2::expandByScalar(float scalar) {

    this->min_.addScalar(-scalar);
    this->max_.addScalar(scalar);

    return *this;
}

bool Box2::containsPoint(const Vector2& point) const {
    // clang-format off
    return point.x < this->min_.x || point.x > this->max_.x ||
                           point.y < this->min_.y || point.y > this->max_.y ? false : true;
    // clang-format on
}

bool Box2::containsBox(const Box2& box) const {

    return this->min_.x <= box.min_.x && box.max_.x <= this->max_.x &&
           this->min_.y <= box.min_.y && box.max_.y <= this->max_.y;
}

bool Box2::intersectsBox(const Box2& box) const {

    // using 4 splitting planes to rule out intersections

    return box.max_.x < this->min_.x || box.min_.x > this->max_.x ||
                           box.max_.y < this->min_.y || box.min_.y > this->max_.y
                   ? false
                   : true;
}

Vector2& Box2::clampPoint(const Vector2& point, Vector2& target) const {

    return target.copy(point).clamp(this->min_, this->max_);
}

float Box2::distanceToPoint(const Vector2& point) const {

    return this->clampPoint(point, _vector).distanceTo(point);
}

Box2& Box2::intersect(const Box2& box) {

    this->min_.max(box.min_);
    this->max_.min(box.max_);

    if (this->isEmpty()) this->makeEmpty();

    return *this;
}

Box2& Box2::_union(const Box2& box) {

    this->min_.min(box.min_);
    this->max_.max(box.max_);

    return *this;
}

Box2& Box2::translate(const Vector2& offset) {

    this->min_.add(offset);
    this->max_.add(offset);

    return *this;
}
