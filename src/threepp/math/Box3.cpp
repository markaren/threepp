
#include "threepp/math/Box3.hpp"

#include <limits>

using namespace threepp;

constexpr float Infinity = std::numeric_limits<float>::infinity();

Vector3 Box3::_vector = Vector3();

Box3::Box3() : min_(Vector3(+Infinity, +Infinity, +Infinity)), max_(Vector3(-Infinity, -Infinity, -Infinity)) {}

Box3::Box3(Vector3 min, Vector3 max) : min_(min), max_(max) {}

Box3 &Box3::set(const Vector3 &min, const Vector3 &max) {

    this->min_ = min;
    this->max_ = max;

    return *this;
}

Box3 &Box3::setFromPoints(const std::vector<Vector3> &points) {

    this->makeEmpty();

    for (auto& point : points) {

        this->expandByPoint( point );

    }

    return *this;

}

Box3 &Box3::setFromCenterAndSize(const Vector3 &center, const Vector3 &size) {

    const auto halfSize = _vector.copy( size ).multiply( 0.5f );

    this->min_.copy( center ).sub( halfSize );
    this->max_.copy( center ).add( halfSize );

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
