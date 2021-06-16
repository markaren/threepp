
#include "threepp/math/Box3.hpp"

#include <limits>

using namespace threepp;

constexpr float Infinity = std::numeric_limits<float>::infinity();

Box3::Box3() : min_(Vector3(+Infinity, +Infinity, +Infinity)), max_(Vector3(-Infinity, -Infinity, -Infinity)) {}

Box3::Box3(Vector3 min, Vector3 max) : min_(min), max_(max) {}

Box3 &Box3::set(const Vector3 &min, const Vector3 &max) {

    this->min_ = min;
    this->max_ = max;

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

Box3& Box3::expandByObject( Object3D &object ) {
    // Computes the world-axis-aligned bounding box of an object (including its children),
    // accounting for both the object's, and children's, world transforms

    object.updateWorldMatrix( false, false );

//    auto geometry = object.geometry;
//
//    if ( geometry !== undefined ) {
//
//        if ( geometry.boundingBox === null ) {
//
//            geometry.computeBoundingBox();
//
//        }
//
//        _box.copy( geometry.boundingBox );
//        _box.applyMatrix4( object.matrixWorld );
//
//        this.union( _box );
//
//    }

    for ( const auto &child : object.children ) {

        this->expandByObject( *child );

    }

    return *this;
}
