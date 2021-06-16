// https://github.com/mrdoob/three.js/blob/r129/src/scenes/FogExp2.js

#ifndef THREEPP_FOGEXP2_HPP
#define THREEPP_FOGEXP2_HPP

#include "threepp/math/Color.hpp"

namespace threepp {

    class FogExp2 {

    public:

        std::string name;

        Color color;
        float density;

        explicit FogExp2(int hex, float density = 0.00025): color(hex), density(density){}
        explicit FogExp2(Color color, float density = 0.00025): color(color), density(density){}

    };

}

#endif//THREEPP_FOGEXP2_HPP
