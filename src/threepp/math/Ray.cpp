
#include "threepp/math/Ray.hpp"

using namespace threepp;

namespace {

    Vector3 _vector;
    Vector3 _segCenter ;
    Vector3 _segDir ;
    Vector3 _diff ;

    Vector3 _edge1 ;
    Vector3 _edge2 ;
    Vector3 _normal ;

}

Ray::Ray(Vector3 origin, Vector3 direction) : origin_(origin), direction_(direction){}

Ray &Ray::set(const Vector3 &origin, const Vector3 &direction) {

    this->origin_.copy( origin );
    this->direction_.copy( direction );

    return *this;

}

Ray &Ray::copy(const Ray &ray) {

    this->origin_.copy( ray.origin_ );
    this->direction_.copy( ray.direction_ );

    return *this;

}

Vector3 &Ray::at(float t, Vector3 &target) {

    return target.copy(this->direction_).multiplyScalar(t).add( this->origin_ );

}

Ray &Ray::lookAt(const Vector3 &v) {

    this->direction_.copy( v ).sub( this->origin_ ).normalize();

    return *this;

}

Ray &Ray::recast(float t) {

    this->origin_.copy( this->at( t, _vector ) );

    return *this;

}
