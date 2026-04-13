
#include "RenderStates.hpp"

#include "threepp/scenes/Scene.hpp"

using namespace threepp;


const Lights& RenderState::getLights() const {

    return lights_;
}

const std::vector<Light*>& RenderState::getLightsArray() const {

    return lightsArray_;
}

const std::vector<Light*>& RenderState::getShadowsArray() const {

    return shadowsArray_;
}

void RenderState::init() {

    lightsArray_.clear();
    shadowsArray_.clear();
}

void RenderState::pushLight(Light* light) {

    lightsArray_.emplace_back(light);
}

void RenderState::pushShadow(Light* shadowLight) {

    shadowsArray_.emplace_back(shadowLight);
}

void RenderState::setupLights() {

    lights_.setup(lightsArray_);
}

void RenderState::setupLightsView(Camera* camera) {

    lights_.setupView(lightsArray_, camera);
}


RenderState* RenderStates::get(Object3D* scene, size_t renderCallDepth) {

    while (renderCallDepth >= renderStates_[scene->uuid].size()) {

        renderStates_[scene->uuid].emplace_back(std::make_unique<RenderState>());
    }

    return renderStates_[scene->uuid].at(renderCallDepth).get();
}

void RenderStates::dispose() {

    renderStates_.clear();
}
