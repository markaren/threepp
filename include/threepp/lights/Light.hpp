// https://github.com/mrdoob/three.js/blob/r129/src/lights/Light.js

#ifndef THREEPP_LIGHT_HPP
#define THREEPP_LIGHT_HPP

#include "threepp/core/Object3d.hpp"
#include "threepp/math/Color.hpp"

#include <optional>

namespace threepp {

    class Light: public Object3d {

    public:

        std::string type = "Light";

        Color color;
        float intensity;

        Light(int color, std::optional<float> intensity): color(color), intensity(intensity.value_or(1)){}
        Light(Color color, std::optional<float> intensity): color(color), intensity(intensity.value_or(1)){}

        void dispose() {}

    };

}

#endif//THREEPP_LIGHT_HPP
