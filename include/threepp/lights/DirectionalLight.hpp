// https://github.com/mrdoob/three.js/blob/r129/src/lights/DirectionalLight.js

#ifndef THREEPP_DIRECTIONALLIGHT_HPP
#define THREEPP_DIRECTIONALLIGHT_HPP

#include "threepp/lights/DirectionalLightShadow.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/lights/LightWithShadow.hpp"

namespace threepp {

    class DirectionalLight : public Light, public LightWithShadow {

    public:
        std::shared_ptr<Object3D> target = Object3D::create();

        std::shared_ptr<DirectionalLightShadow> shadow;

        LightShadow *getLightShadow() override {

            return shadow.get();
        }

        void dispose() override {

            this->shadow->dispose();
        }

        [[nodiscard]] std::string type() const override {

            return "DirectionalLight";
        }

        template<class T>
        static std::shared_ptr<DirectionalLight> create(T color, std::optional<float> intensity = std::nullopt) {

            return std::shared_ptr<DirectionalLight>(new DirectionalLight(color, intensity));
        }

    protected:
        template<class T>
        DirectionalLight(T color, std::optional<float> intensity) : Light(color, intensity) {

            this->position.copy(Object3D::defaultUp);
            this->updateMatrix();
        }
    };

}// namespace threepp

#endif//THREEPP_DIRECTIONALLIGHT_HPP
