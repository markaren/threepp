
#include "threepp/math/box2.hpp"

#include <limits>

using namespace threepp;

constexpr float Infinity = std::numeric_limits<float>::infinity();

vector2 box2::_vector = vector2();

box2::box2() : min_(+Infinity, +Infinity), max_(-Infinity, -Infinity) {}

box2::box2(const vector2 &min, const vector2 &max) : min_(min), max_(max) {}

box2 &box2::set(const vector2 &min, const vector2 &max) {

    this->min_ = min;
    this->max_ = max;

    return *this;
}

box2 &box2::makeEmpty() {
    this->min_.x = +Infinity;
    this->min_.y = +Infinity;
    this->max_.x = -Infinity;
    this->max_.y = -Infinity;

    return *this;
}

bool box2::isEmpty() const {
    // this is a more robust check for empty than ( volume <= 0 ) because volume can get positive with two negative axes

    return ( this->max_.x < this->min_.x ) || ( this->max_.y < this->min_.y );
}
