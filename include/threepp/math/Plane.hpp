// https://github.com/mrdoob/three.js/blob/r129/src/math/Plane.js

#ifndef THREEPP_PLANE_HPP
#define THREEPP_PLANE_HPP

#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Plane {

    public:
        Vector3 normal;
        float constant;

        Plane();
        Plane(Vector3 normal, float constant);

        Plane &set(const Vector3 &normal, float constant);

        Plane &setComponents( float x, float y, float z, float w );

    private:
        static Vector3 _vector1;
        static Vector3 _vector2;
        static Matrix3 _normalMatrix;
    };

}// namespace threepp

#endif//THREEPP_PLANE_HPP
