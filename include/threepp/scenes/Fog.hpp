// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Fog.js

#ifndef THREEPP_FOG_HPP
#define THREEPP_FOG_HPP

#include "threepp/math/Color.hpp"

namespace threepp {

    class Fog {

    public:
        Color color;
        float near;
        float far;

        template<class T>
        explicit Fog(T color, float near = 1, float far = 1000)
            : color(color), near(near), far(far) {}

        bool operator==(const Fog &f) const {

            return f.color == this->color && f.near == this->near && f.far == this->far;
        }

    };

}// namespace threepp

#endif//THREEPP_FOG_HPP
