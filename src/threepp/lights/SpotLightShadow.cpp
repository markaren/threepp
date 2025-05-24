
#include "threepp/lights/SpotLightShadow.hpp"
#include "threepp/lights/SpotLight.hpp"

using namespace threepp;

SpotLightShadow::SpotLightShadow()
    : LightShadow(std::make_unique<PerspectiveCamera>(50.f, 1.f, 0.5f, 500.f)) {}

void SpotLightShadow::updateMatrices(Light& _light) {

    auto light = _light.as<SpotLight>();

    const auto fov = math::RAD2DEG * 2 * light->angle * this->focus;
    const auto aspect = this->mapSize.x / this->mapSize.y;
    const auto far = (light->distance > 0) ? light->distance : camera->farPlane;

    auto c = camera->as<PerspectiveCamera>();

    if (fov != c->fov || aspect != c->aspect || far != camera->farPlane) {

        c->fov = fov;
        c->aspect = aspect;
        c->farPlane = far;
        c->updateProjectionMatrix();
    }

    LightShadow::updateMatrices(_light);
}

std::shared_ptr<SpotLightShadow> SpotLightShadow::create() {

    return std::shared_ptr<SpotLightShadow>(new SpotLightShadow());
}
