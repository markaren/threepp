
#include "threepp/lights/Light.hpp"

using namespace threepp;


Light::Light(const Color& color, std::optional<float> intensity)
    : color(color), intensity(intensity.value_or(1)) {}


const std::string& Light::type() const {

    static const std::string typeName = "Light";
    return typeName;
}

void Light::copy(const Object3D& source, bool recursive) {
    Object3D::copy(source, recursive);

    if (const auto l = source.as<Light>()) {

        color.copy(l->color);
        intensity = l->intensity;
    }
}

void Light::dispose() {}
