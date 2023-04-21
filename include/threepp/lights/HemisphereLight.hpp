// https://github.com/mrdoob/three.js/blob/r129/src/lights/HemisphereLight.js

#ifndef THREEPP_HEMISPHERELIGHT_HPP
#define THREEPP_HEMISPHERELIGHT_HPP

#include "threepp/lights/Light.hpp"

namespace threepp {

    class HemisphereLight: public Light {

    public:
        Color groundColor;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<HemisphereLight> create(const Color& skyColor = 0xffffff, const Color& groundColor = 0xffffff, std::optional<float> intensity = std::nullopt);

    protected:
        HemisphereLight(const Color& skyColor, const Color& groundColor, std::optional<float> intensity);
    };

}// namespace threepp

#endif//THREEPP_HEMISPHERELIGHT_HPP
