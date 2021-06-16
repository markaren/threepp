// https://github.com/mrdoob/three.js/blob/r129/src/math/Sphere.js

#ifndef THREEPP_SPHERE_HPP
#define THREEPP_SPHERE_HPP

#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Sphere {

    public:
        explicit Sphere(Vector3 center, float radius = -1): center_(center), radius_(radius) {}

        Sphere &set( const Vector3 &center, float radius ) {

            this.center = ( center );
            this.radius = radius;

            return *this;

        }

    private:

        Vector3 center_;
        float radius_;

    };

}// namespace threepp

#endif//THREEPP_SPHERE_HPP
