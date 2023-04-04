
#include "threepp/lights/AmbientLight.hpp"

using namespace threepp;


AmbientLight::AmbientLight(const Color& color, std::optional<float> intensity)
    : Light(color, intensity) {}


std::string AmbientLight::type() const {

    return "AmbientLight";
}

std::shared_ptr<AmbientLight> AmbientLight::create(const Color& color, std::optional<float> intensity) {

    return std::shared_ptr<AmbientLight>(new AmbientLight(color, intensity));
}
