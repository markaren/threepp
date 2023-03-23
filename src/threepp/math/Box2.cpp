
#include "threepp/math/Box2.hpp"

#include "threepp/math/infinity.hpp"

using namespace threepp;

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

void Box2::getCenter(Vector2& target) {

    this->isEmpty() ? target.set(0, 0) : target.addVectors(this->min_, this->max_).multiplyScalar(0.5f);
}

void Box2::getSize(Vector2& target) {

    this->isEmpty() ? target.set(0, 0) : target.subVectors(this->max_, this->min_);
}

Box2& Box2::expandByPoint(const Vector2& point) {

    this->min_.min(point);
    this->max_.max(point);

    return *this;
}
