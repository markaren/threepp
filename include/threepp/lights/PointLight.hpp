// https://github.com/mrdoob/three.js/blob/r129/src/lights/PointLight.js

#ifndef THREEPP_POINTLIGHT_HPP
#define THREEPP_POINTLIGHT_HPP

#include "threepp/lights/Light.hpp"
#include "threepp/lights/LightWithShadow.hpp"
#include "threepp/lights/PointLightShadow.hpp"

namespace threepp {

    class PointLight : public Light, public LightWithShadow {

    public:
        float distance;
        float decay;

        std::shared_ptr<PointLightShadow> shadow;

        [[nodiscard]] float getPower() const;

        void setPower(float power);

        LightShadow *getLightShadow() override {

            return shadow.get();
        }

        void dispose() override;

        [[nodiscard]] std::string type() const override {

            return "PointLight";
        }

        template<class T>
        static std::shared_ptr<PointLight> create(T color, std::optional<float> intensity = std::nullopt, float distance = 0, float decay = 1) {

            return std::shared_ptr<PointLight>(new PointLight(color, intensity, distance, decay));
        }

    protected:
        template<class T>
        PointLight(T color, std::optional<float> intensity, float distance, float decay)
            : Light(color, intensity), distance(distance), decay(decay) {}
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHT_HPP
