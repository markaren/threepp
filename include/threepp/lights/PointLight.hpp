// https://github.com/mrdoob/three.js/blob/r129/src/lights/PointLight.js

#ifndef THREEPP_POINTLIGHT_HPP
#define THREEPP_POINTLIGHT_HPP

#include "threepp/lights/Light.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class PointLight : public Light {

    public:
        PointLight(const PointLight &) = delete;

        float getPower() const {

            // intensity = power per solid angle.
            // ref: equation (15) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
            return this->intensity * 4f * PI;
        }

        void setPower(float power) {

            // intensity = power per solid angle.
            // ref: equation (15) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
            this->intensity = power / (4 * PI);
        }

        dispose() {

            this->shadow.dispose();
        }

        template<class T>
        static std::shared_ptr<PointLight> create(T color, std::optional<float> intensity = std::nullopt, float distance = 0, float decay = 1) {
            return std::shared_ptr<PointLight>(new PointLight(color, intensity, distance, decay));
        }

    protected:
        PointLight(T color, std::optional<float> intensity, float distance, float decay) : Light(color, intensity) {}
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHT_HPP
