// https://github.com/mrdoob/three.js/blob/r129/src/lights/PointLight.js

#ifndef THREEPP_POINTLIGHT_HPP
#define THREEPP_POINTLIGHT_HPP

#include "threepp/lights/Light.hpp"
#include "threepp/lights/PointLightShadow.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class PointLight : public Light {

    public:
        float getPower() const {

            // intensity = power per solid angle.
            // ref: equation (15) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
            return this->intensity * 4.f * math::PI;
        }

        void setPower(float power) {

            // intensity = power per solid angle.
            // ref: equation (15) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
            this->intensity = power / (4.f * math::PI);
        }

        void dispose() override {

            this->shadow_.dispose();
        }

        template<class T>
        static std::shared_ptr<PointLight> create(T color, std::optional<float> intensity = std::nullopt, float distance = 0, float decay = 1) {
            return std::shared_ptr<PointLight>(new PointLight(color, intensity, distance, decay));
        }

    protected:
        template<class T>
        PointLight(T color, std::optional<float> intensity, float distance, float decay)
            : Light(color, intensity), shadow_(PerspectiveCamera::create(90, 1, 0.5, 500)) {}

    private:
        PointLightShadow shadow_;
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHT_HPP
