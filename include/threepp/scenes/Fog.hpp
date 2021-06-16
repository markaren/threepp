// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Fog.js

#ifndef THREEPP_FOG_HPP
#define THREEPP_FOG_HPP

#include "threepp/math/Color.hpp"

namespace threepp {

    class Fog  {

    public:
        std::string name;

        Color color;
        float near;
        float far;

        explicit Fog(int hex, float near = 1, float far = 1000): color(hex), near(near), far(far) {}
        explicit Fog(Color color, float near = 1, float far = 1000): color(color), near(near), far(far) {}

    };

}

#endif//THREEPP_FOG_HPP
