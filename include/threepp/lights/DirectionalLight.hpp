// https://github.com/mrdoob/three.js/blob/r129/src/lights/DirectionalLight.js

#ifndef THREEPP_DIRECTIONALLIGHT_HPP
#define THREEPP_DIRECTIONALLIGHT_HPP

#include "threepp/lights/Light.hpp"
#include "threepp/lights/LightShadow.hpp"

namespace threepp {

    class DirectionalLight : Light {

    public:
        DirectionalLight(const DirectionalLight &) = delete;

        template<class T>
        static std::shared_ptr<DirectionalLight> create(T color, std::optional<float> intensity = std::nullopt) {
            return std::shared_ptr<DirectionalLight>(new DirectionalLight(color, intensity));
        }

        std::optional<Object3D> target() override {
            return target_;
        }

        void dispose() override {

            this->shadow_.dispose();
        }
        
    protected:
        template<class T>
        DirectionalLight(T color, std::optional<float> intensity) : Light(color, intensity) {

            this->position_.copy(Object3D::defaultUp);
            this->updateMatrix();
        }

    private:
        Vector3 position_;
        Object3D target_;

        LightShadow shadow_;
    };

}// namespace threepp

#endif//THREEPP_DIRECTIONALLIGHT_HPP
