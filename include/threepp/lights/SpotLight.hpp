// // https://github.com/mrdoob/three.js/blob/r129/src/lights/SpotLight.js

#ifndef THREEPP_SPOTLIGHT_HPP
#define THREEPP_SPOTLIGHT_HPP

#include "threepp/lights/Light.hpp"
#include "threepp/lights/SpotLightShadow.hpp"
#include "threepp/lights/light_interfaces.hpp"

namespace threepp {

    class SpotLight : public Light, public LightWithShadow<SpotLightShadow> {

    public:
        float distance;
        float angle;
        float penumbra;
        float decay;

        std::shared_ptr<Object3D> target{Object3D::create()};

        std::shared_ptr<SpotLightShadow> shadow{SpotLightShadow::create()};

        float getPower();

        void setPower(float power);

        void dispose() override;

        [[nodiscard]] std::string type() const override {

            return "SpotLight";
        }

        template<class T>
        static std::shared_ptr<SpotLight> create(T color, std::optional<float> intensity = std::nullopt, float distance = 0, float angle = math::PI / 3, float penumbra = 0, float decay = 1) {

            return std::shared_ptr<SpotLight>(new SpotLight(color, intensity, distance, angle, penumbra, decay));
        }

    protected:
        template<class T>
        SpotLight(T color, std::optional<float> intensity, float distance, float angle, float penumbra, float decay)
            : Light(color, intensity), LightWithShadow<SpotLightShadow>(SpotLightShadow::create()), distance(distance), angle(angle), penumbra(penumbra), decay(decay) {

            this->position.copy(Object3D::defaultUp);
            this->updateMatrix();
        }
    };

}// namespace threepp

#endif//THREEPP_SPOTLIGHT_HPP
