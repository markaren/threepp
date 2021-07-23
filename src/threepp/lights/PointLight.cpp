
#include "threepp/lights/PointLight.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"

using namespace threepp;

float PointLight::getPower() const {

    // intensity = power per solid angle.
    // ref: equation (15) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    return this->intensity * 4.f * math::PI;
}

void PointLight::setPower(float power) {

    // intensity = power per solid angle.
    // ref: equation (15) from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    this->intensity = power / (4.f * math::PI);
}

void PointLight::dispose() {

    this->shadow.dispose();
}
