// // https://github.com/mrdoob/three.js/blob/r129/src/lights/SpotLight.js

#ifndef THREEPP_SPOTLIGHT_HPP
#define THREEPP_SPOTLIGHT_HPP

#include "threepp/lights/Light.hpp"
#include "threepp/lights/SpotLightShadow.hpp"
#include "threepp/lights/light_interfaces.hpp"

namespace threepp {

    class SpotLight: public Light, public LightWithShadow, public LightWithTarget {

    public:
        float distance;
        float angle;
        float penumbra;
        float decay;

        float getPower();

        void setPower(float power);

        void dispose() override;

        [[nodiscard]] std::string type() const override {

            return "SpotLight";
        }

        static std::shared_ptr<SpotLight> create(const Color& color = 0xffffff, std::optional<float> intensity = std::nullopt, float distance = 0, float angle = math::PI / 3, float penumbra = 0, float decay = 1) {

            return std::shared_ptr<SpotLight>(new SpotLight(color, intensity, distance, angle, penumbra, decay));
        }

    protected:
        SpotLight(const Color& color, std::optional<float> intensity, float distance, float angle, float penumbra, float decay)
            : Light(color, intensity), LightWithShadow(SpotLightShadow::create()), distance(distance), angle(angle), penumbra(penumbra), decay(decay) {

            this->position.copy(Object3D::defaultUp);
            this->updateMatrix();
        }
    };

}// namespace threepp

#endif//THREEPP_SPOTLIGHT_HPP
