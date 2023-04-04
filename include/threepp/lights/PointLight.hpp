// https://github.com/mrdoob/three.js/blob/r129/src/lights/PointLight.js

#ifndef THREEPP_POINTLIGHT_HPP
#define THREEPP_POINTLIGHT_HPP

#include "threepp/lights/Light.hpp"
#include "threepp/lights/light_interfaces.hpp"

namespace threepp {

    class PointLight: public Light, public LightWithShadow {

    public:
        float distance;
        float decay;

        [[nodiscard]] std::string type() const override;

        [[nodiscard]] float getPower() const;

        void setPower(float power);

        void dispose() override;

        static std::shared_ptr<PointLight> create(const Color& color = 0xffffff, std::optional<float> intensity = std::nullopt, float distance = 0, float decay = 1);

    protected:
        PointLight(const Color& color, std::optional<float> intensity, float distance, float decay);
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHT_HPP
