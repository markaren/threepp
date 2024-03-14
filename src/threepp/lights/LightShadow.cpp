
#include "threepp/lights/LightShadow.hpp"

#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/lights/light_interfaces.hpp"

using namespace threepp;


LightShadow::LightShadow(std::unique_ptr<Camera> camera)
    : camera(std::move(camera)) {}


size_t LightShadow::getViewportCount() const {

    return this->_viewports.size();
}

const Frustum& LightShadow::getFrustum() const {

    return this->_frustum;
}

void LightShadow::updateMatrices(Light& light) {

    auto& shadowCamera = this->camera;
    auto& shadowMatrix = this->matrix;

    _lightPositionWorld.setFromMatrixPosition(*light.matrixWorld);
    shadowCamera->position.copy(_lightPositionWorld);

    auto lightWithTarget = dynamic_cast<LightWithTarget*>(&light);
    _lookTarget.setFromMatrixPosition(*lightWithTarget->target().matrixWorld);
    shadowCamera->lookAt(_lookTarget);
    shadowCamera->updateMatrixWorld();

    _projScreenMatrix.multiplyMatrices(shadowCamera->projectionMatrix, shadowCamera->matrixWorldInverse);
    this->_frustum.setFromProjectionMatrix(_projScreenMatrix);

    shadowMatrix.set(
            0.5f, 0.0f, 0.0f, 0.5f,
            0.0f, 0.5f, 0.0f, 0.5f,
            0.0f, 0.0f, 0.5f, 0.5f,
            0.0f, 0.0f, 0.0f, 1.0f);

    shadowMatrix.multiply(shadowCamera->projectionMatrix);
    shadowMatrix.multiply(shadowCamera->matrixWorldInverse);
}

Vector4& LightShadow::getViewport(size_t viewportIndex) {

    return this->_viewports[viewportIndex];
}

Vector2& LightShadow::getFrameExtents() {

    return this->_frameExtents;
}

void LightShadow::dispose() {

    if (this->map) {

        this->map->dispose();
    }

    if (this->mapPass) {

        this->mapPass->dispose();
    }
}

LightShadow::~LightShadow() {

    dispose();
}
