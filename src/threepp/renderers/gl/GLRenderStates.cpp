
#include "threepp/renderers/gl/GLRenderStates.hpp"

#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using namespace threepp::gl;


const GLLights& GLRenderState::getLights() const {

    return lights_;
}

const std::vector<Light*>& GLRenderState::getLightsArray() const {

    return lightsArray_;
}

const std::vector<Light*>& GLRenderState::getShadowsArray() const {

    return shadowsArray_;
}

void GLRenderState::init() {

    lightsArray_.clear();
    shadowsArray_.clear();
}

void GLRenderState::pushLight(Light* light) {

    lightsArray_.emplace_back(light);
}

void GLRenderState::pushShadow(Light* shadowLight) {

    shadowsArray_.emplace_back(shadowLight);
}

void GLRenderState::setupLights() {

    lights_.setup(lightsArray_);
}

void GLRenderState::setupLightsView(Camera* camera) {

    lights_.setupView(lightsArray_, camera);
}


GLRenderState* GLRenderStates::get(Object3D* scene, size_t renderCallDepth) {

    if (renderCallDepth >= renderStates_[scene->uuid].size()) {

        renderStates_[scene->uuid].emplace_back(std::make_unique<GLRenderState>());
    }

    return renderStates_[scene->uuid].at(renderCallDepth).get();
}

void GLRenderStates::dispose() {

    renderStates_.clear();
}
