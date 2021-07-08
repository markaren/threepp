// https://github.com/mrdoob/three.js/blob/r129/src/lights/AmbientLight.js

#ifndef THREEPP_AMBIENTLIGHT_HPP
#define THREEPP_AMBIENTLIGHT_HPP

#include "threepp/lights/Light.hpp"

#include <optional>

namespace threepp {

    class AmbientLight : public Light {

    public:
        [[nodiscard]] std::string type() const override {
            return "AmbientLight";
        }

        template<class T>
        static std::shared_ptr<AmbientLight> create(T color, std::optional<float> intensity = std::nullopt) {
            return std::shared_ptr<AmbientLight>(new AmbientLight(color, intensity));
        }

    protected:
        template<class T>
        explicit AmbientLight(T color, std::optional<float> intensity) : Light(color, intensity) {}
    };

}// namespace threepp

#endif//THREEPP_AMBIENTLIGHT_HPP
