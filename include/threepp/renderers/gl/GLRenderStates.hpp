// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLRenderStates.js

#ifndef THREEPP_GLRENDERSTATES_HPP
#define THREEPP_GLRENDERSTATES_HPP

#include "GLLights.hpp"

namespace threepp::gl {

    struct GLRenderState {

        GLRenderState() = default;

        const GLLights &getLights() const {

            return lights_;
        }
        const std::vector<Light *> &getLightsArray() const {

            return lightsArray_;
        }
        const std::vector<Light *> &getShadowsArray() const {

            return shadowsArray_;
        }

        void init() {

            lightsArray_.clear();
            shadowsArray_.clear();
        }

        void pushLight(Light *light) {

            lightsArray_.emplace_back(light);
        }

        void pushShadow(Light *shadowLight) {

            shadowsArray_.emplace_back(shadowLight);
        }

        void setupLights() {

            lights_.setup(lightsArray_);
        }

        void setupLightsView(Camera *camera) {

            lights_.setupView(lightsArray_, camera);
        }


    private:
        GLLights lights_;

        std::vector<Light *> lightsArray_;
        std::vector<Light *> shadowsArray_;
    };

    struct GLRenderStates {

        GLRenderStates() = default;

        std::shared_ptr<GLRenderState> get(Scene *scene, size_t renderCallDepth = 1) {

            if (renderCallDepth >= renderStates_[scene->uuid].size()) {

                renderStates_[scene->uuid].emplace_back(std::make_shared<GLRenderState>());
            }

            return renderStates_[scene->uuid].at(renderCallDepth);
        }

        void dispose() {

            renderStates_.clear();
        }

    private:
        std::unordered_map<std::string, std::vector<std::shared_ptr<GLRenderState>>> renderStates_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLRENDERSTATES_HPP
