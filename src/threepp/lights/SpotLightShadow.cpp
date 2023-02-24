
#include "threepp/lights/SpotLightShadow.hpp"
#include "threepp/lights/SpotLight.hpp"

using namespace threepp;

void SpotLightShadow::updateMatrices(SpotLight* light) {

    const auto fov = math::RAD2DEG * 2 * light->angle * this->focus;
    const auto aspect = this->mapSize.x / this->mapSize.y;
    const auto far = (light->distance > 0) ? light->distance : camera->far;

    auto c = dynamic_cast<PerspectiveCamera*>(camera.get());

    if (fov != c->fov || aspect != c->aspect || far != camera->far) {

        c->fov = fov;
        c->aspect = aspect;
        c->far = far;
        c->updateProjectionMatrix();
    }

    LightShadow::updateMatrices(light);
}
