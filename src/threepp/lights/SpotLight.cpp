
#include "threepp/lights/SpotLight.hpp"

#include "threepp/lights/SpotLightShadow.hpp"

using namespace threepp;


SpotLight::SpotLight(const Color& color, std::optional<float> intensity, float distance, float angle, float penumbra, float decay)
    : Light(color, intensity), LightWithShadow(SpotLightShadow::create()), distance(distance), angle(angle), penumbra(penumbra), decay(decay) {

    this->position.copy(Object3D::defaultUp);
    this->updateMatrix();
}


std::string SpotLight::type() const {

    return "SpotLight";
}

float SpotLight::getPower() {

    // intensity = power per solid angle.
    // ref: equation (17) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    return this->intensity * math::PI;
}

void SpotLight::setPower(float power) {

    // intensity = power per solid angle.
    // ref: equation (17) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    this->intensity = power / math::PI;
}

void SpotLight::dispose() {

    this->shadow->dispose();
}

std::shared_ptr<SpotLight> SpotLight::create(const Color& color, std::optional<float> intensity, float distance, float angle, float penumbra, float decay) {

    return std::shared_ptr<SpotLight>(new SpotLight(color, intensity, distance, angle, penumbra, decay));
}
