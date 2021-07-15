
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

Ray::Ray(Vector3 origin, Vector3 direction) : origin(origin), direction(direction){}

Ray &Ray::set(const Vector3 &origin, const Vector3 &direction) {

    this->origin.copy( origin );
    this->direction.copy( direction );

    return *this;

}

Ray &Ray::copy(const Ray &ray) {

    this->origin.copy( ray.origin);
    this->direction.copy( ray.direction);

    return *this;

}

Vector3 &Ray::at(float t, Vector3 &target) {

    return target.copy(this->direction).multiplyScalar(t).add( this->origin);

}

Ray &Ray::lookAt(const Vector3 &v) {

    this->direction.copy( v ).sub( this->origin).normalize();

    return *this;

}

Ray &Ray::recast(float t) {

    this->origin.copy( this->at( t, _vector ) );

    return *this;

}
