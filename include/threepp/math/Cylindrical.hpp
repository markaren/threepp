// https://github.com/mrdoob/three.js/blob/r129/src/math/Cylindrical.js

#ifndef THREEPP_CYLINDRICAL_HPP
#define THREEPP_CYLINDRICAL_HPP

namespace threepp {

    class Vector3;

    class Cylindrical {

    public:
        float radius;
        float theta;
        float y;

        explicit Cylindrical(float radius = 1, float theta = 0, float y = 0);

        Cylindrical& set(float radius, float theta, float y);

        Cylindrical& copy(const Cylindrical& other);

        Cylindrical& setFromVector3(const Vector3& v);

        Cylindrical& setFromCartesianCoords(float x, float y, float z);
    };

}// namespace threepp

#endif//THREEPP_CYLINDRICAL_HPP
