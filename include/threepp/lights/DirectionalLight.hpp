// https://github.com/mrdoob/three.js/blob/r129/src/lights/DirectionalLight.js

#ifndef THREEPP_DIRECTIONALLIGHT_HPP
#define THREEPP_DIRECTIONALLIGHT_HPP

#include "threepp/lights/DirectionalLightShadow.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/lights/light_interfaces.hpp"

namespace threepp {

    class DirectionalLight: public Light, public LightWithShadow, public LightWithTarget {

    public:
        [[nodiscard]] std::string type() const override {

            return "DirectionalLight";
        }

        void dispose() override {

            this->shadow->dispose();
        }

        static std::shared_ptr<DirectionalLight> create(const Color& color = 0xffffff, std::optional<float> intensity = std::nullopt) {

            return std::shared_ptr<DirectionalLight>(new DirectionalLight(color, intensity));
        }

    protected:
        DirectionalLight(const Color& color, std::optional<float> intensity)
            : Light(color, intensity), LightWithShadow(DirectionalLightShadow::create()) {

            this->position.copy(Object3D::defaultUp);
            this->updateMatrix();
        }
    };

}// namespace threepp

#endif//THREEPP_DIRECTIONALLIGHT_HPP
