// https://github.com/mrdoob/three.js/blob/r129/src/math/Ray.js

#ifndef THREEPP_RAY_HPP
#define THREEPP_RAY_HPP

#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Ray {

    public:
        explicit Ray(Vector3 origin = Vector3(), Vector3 direction = Vector3());

        Ray &set( const Vector3 &origin, const Vector3 &direction );

        Ray &copy( const Ray &ray );

        Vector3 &at( float t, Vector3 &target );

        Ray &lookAt( const Vector3 &v );

        Ray &recast( float t );

    private:

        Vector3 origin_;
        Vector3 direction_;

        static Vector3 _vector;
        static Vector3 _segCenter ;
        static Vector3 _segDir ;
        static Vector3 _diff ;

        static Vector3 _edge1 ;
        static Vector3 _edge2 ;
        static Vector3 _normal ;

    };

}

#endif//THREEPP_RAY_HPP
