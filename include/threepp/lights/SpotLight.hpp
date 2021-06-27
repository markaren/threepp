// // https://github.com/mrdoob/three.js/blob/r129/src/lights/SpotLight.js

#ifndef THREEPP_SPOTLIGHT_HPP
#define THREEPP_SPOTLIGHT_HPP

#include "threepp/lights/Light.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class SpotLight : public Light {

    public:
        float distance;
        float angle;
        float penumbra;
        float decay;

        Object3D* target = nullptr;

        float getPower() {

            // intensity = power per solid angle.
            // ref: equation (17) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
            return this->intensity * math::PI;
        }

        void setPower(float power) {

            // intensity = power per solid angle.
            // ref: equation (17) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
            this->intensity = power / math::PI;
        }

        void dispose() {

            //            this->shadow.dispose();
        }

        std::string type() const override {
            return "SpotLight";
        }

        SpotLight &copy(SpotLight &source) {

            //                this->distance = source.distance;
            //                this->angle = source.angle;
            //                this->penumbra = source.penumbra;
            //                this->decay = source.decay;
            //
            //                this->target = source.target.clone();
            //
            //                this->shadow = source.shadow.clone();

            return *this;
        }

        template<class T>
        static std::shared_ptr<SpotLight> create(T color, std::optional<float> intensity = std::nullopt, float distance = 0, float angle = math::PI / 3, float penumbra = 0, float decay = 1) {
            return std::shared_ptr<SpotLight>(new SpotLight(color, intensity));
        }

    protected:
        template<class T>
        SpotLight(T color, std::optional<float> intensity, float distance, float angle, float penumbra, float decay)
            : Light(color, intensity), distance(distance), angle(angle), penumbra(penumbra), decay(decay) {

            this->position.copy(Object3D::defaultUp);
            this->updateMatrix();
        }

    private:
        //SpotLightShadow shadow;
    };

}// namespace threepp

#endif//THREEPP_SPOTLIGHT_HPP
