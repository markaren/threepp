// https://github.com/mrdoob/three.js/blob/r129/src/math/Plane.js

#ifndef THREEPP_PLANE_HPP
#define THREEPP_PLANE_HPP

#include "threepp/math/matrix3.hpp"
#include "threepp/math/vector3.hpp"

namespace threepp {

    class plane {

    public:
        vector3 normal;
        float constant;

        plane();
        plane(vector3 normal, float constant);

        plane &set(const vector3 &normal, float constant);

        plane &setComponents( float x, float y, float z, float w );

    private:
        static vector3 _vector1;
        static vector3 _vector2;
        static matrix3 _normalMatrix;
    };

}// namespace threepp

#endif//THREEPP_PLANE_HPP
