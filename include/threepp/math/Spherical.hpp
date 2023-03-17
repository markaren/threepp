// https://github.com/mrdoob/three.js/blob/r129/src/math/Spherical.js

#ifndef THREEPP_SPHERICAL_HPP
#define THREEPP_SPHERICAL_HPP

#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Spherical {

    public:
        float radius;
        float phi;
        float theta;

        explicit Spherical(float radius = 1, float phi = 0, float theta = 0);

        Spherical& set(float radius, float phi, float theta);

        Spherical& copy(const Spherical& other);

        // restrict phi to be between EPS and PI-EPS
        Spherical& makeSafe();

        Spherical& setFromVector3(const Vector3& v);

        Spherical& setFromCartesianCoords(float x, float y, float z);
    };

}// namespace threepp

#endif//THREEPP_SPHERICAL_HPP
