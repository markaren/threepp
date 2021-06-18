// https://github.com/mrdoob/three.js/blob/r129/src/lights/AmbientLight.js

#ifndef THREEPP_AMBIENTLIGHT_HPP
#define THREEPP_AMBIENTLIGHT_HPP

#include "threepp/lights/Light.hpp"

#include <optional>

namespace threepp {

    class AmbientLight : public Light {

    public:
        explicit AmbientLight(int color, std::optional<float> intensity = std::nullopt) : Light(color, intensity) {}
        explicit AmbientLight(Color color, std::optional<float> intensity = std::nullopt) : Light(color, intensity) {}

        std::string type() const override {
            return "AmbientLight";
        }
    };

}// namespace threepp

#endif//THREEPP_AMBIENTLIGHT_HPP
