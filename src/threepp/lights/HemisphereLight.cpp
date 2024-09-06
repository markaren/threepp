
#include "threepp/lights/HemisphereLight.hpp"

using namespace threepp;


HemisphereLight::HemisphereLight(const Color& skyColor, const Color& groundColor, std::optional<float> intensity)
    : Light(skyColor, intensity),
      groundColor(groundColor) {

    position.copy(Object3D::defaultUp);
    updateMatrix();
}


std::string HemisphereLight::type() const {

    return "HemisphereLight";
}

void HemisphereLight::copy(const Object3D& source, bool recursive) {
    Light::copy(source, recursive);

    if (auto l = source.as<HemisphereLight>()) {

        groundColor.copy(l->groundColor);
    }
}

std::shared_ptr<HemisphereLight> HemisphereLight::create(const Color& skyColor, const Color& groundColor, std::optional<float> intensity) {

    return std::shared_ptr<HemisphereLight>(new HemisphereLight(skyColor, groundColor, intensity));
}
