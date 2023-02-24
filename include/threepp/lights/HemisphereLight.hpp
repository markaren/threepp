// https://github.com/mrdoob/three.js/blob/r129/src/lights/HemisphereLight.js

#ifndef THREEPP_HEMISPHERELIGHT_HPP
#define THREEPP_HEMISPHERELIGHT_HPP

#include "threepp/lights/Light.hpp"

namespace threepp {

    class HemisphereLight: public Light {

    public:
        Color groundColor;

        [[nodiscard]] virtual std::string type() const override {

            return "HemisphereLight";
        }

        static std::shared_ptr<HemisphereLight> create(const Color& skyColor = 0xffffff, const Color& groundColor = 0xffffff, std::optional<float> intensity = std::nullopt) {

            return std::shared_ptr<HemisphereLight>(new HemisphereLight(skyColor, groundColor, intensity));
        }

    protected:
        HemisphereLight(const Color& skyColor, const Color& groundColor, std::optional<float> intensity)
            : Light(skyColor, intensity),
              groundColor(groundColor) {

            position.copy(Object3D::defaultUp);
            updateMatrix();
        }
    };

}// namespace threepp

#endif//THREEPP_HEMISPHERELIGHT_HPP
