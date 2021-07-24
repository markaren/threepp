
#include "threepp/lights/SpotLight.hpp"

using namespace threepp;

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
