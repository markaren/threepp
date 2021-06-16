
#include "threepp/math/Box2.hpp"

#include <limits>

using namespace threepp;

constexpr float Infinity = std::numeric_limits<float>::infinity();

Vector2 Box2::_vector = Vector2();

Box2::Box2() : min_(+Infinity, +Infinity), max_(-Infinity, -Infinity) {}

Box2::Box2(const Vector2 &min, const Vector2 &max) : min_(min), max_(max) {}

Box2 &Box2::set(const Vector2 &min, const Vector2 &max) {

    this->min_ = min;
    this->max_ = max;

    return *this;
}

Box2 &Box2::setFromPoints(const std::vector<Vector2> &points) {

    this->makeEmpty();

    for ( size_t i = 0, il = points.size(); i < il; i ++ ) {

        this->expandByPoint( points[ i ] );

    }

    return *this;

}

Box2 &Box2::makeEmpty() {
    this->min_.x = +Infinity;
    this->min_.y = +Infinity;
    this->max_.x = -Infinity;
    this->max_.y = -Infinity;

    return *this;
}

bool Box2::isEmpty() const {
    // this is a more robust check for empty than ( volume <= 0 ) because volume can get positive with two negative axes

    return ( this->max_.x < this->min_.x ) || ( this->max_.y < this->min_.y );
}

void Box2::getCenter(Vector2 &target) {

    this->isEmpty() ? target.set( 0, 0 ) : target.addVectors( this->min_, this->max_ ).multiply( 0.5f );

}

void Box2::getSize(Vector2 &target) {

    this->isEmpty() ? target.set( 0, 0 ) : target.subVectors( this->max_, this->min_ );

}

Box2 &Box2::expandByPoint(const Vector2 &point) {

    this->min_.min( point );
    this->max_.max( point );

    return *this;

}
