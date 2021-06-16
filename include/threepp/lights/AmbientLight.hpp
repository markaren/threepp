// https://github.com/mrdoob/three.js/blob/r129/src/lights/AmbientLight.js

#ifndef THREEPP_AMBIENTLIGHT_HPP
#define THREEPP_AMBIENTLIGHT_HPP

#include "threepp/lights/Light.hpp"

#include <optional>

namespace threepp {

    class AmbientLight: public Light {

    public:

        std::string type = "AmbientLight";

        AmbientLight(int color, std::optional<float> intensity = std::nullopt): Light(color, intensity){}

    };

}

#endif//THREEPP_AMBIENTLIGHT_HPP
