// https://github.com/mrdoob/three.js/blob/r129/src/scenes/FogExp2.js

#ifndef THREEPP_FOGEXP2_HPP
#define THREEPP_FOGEXP2_HPP

#include "threepp/math/Color.hpp"

namespace threepp {

    class FogExp2 {

    public:

        Color color;
        float density;

        template<class T>
        explicit FogExp2(T hex, float density = 0.00025f): color(hex), density(density){}

        bool operator==(const FogExp2 &f) const {

            return f.color == this->color && f.density == this->density;
        }

    };

}

#endif//THREEPP_FOGEXP2_HPP
