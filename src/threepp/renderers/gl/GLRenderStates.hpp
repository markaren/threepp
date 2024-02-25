// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLRenderStates.js

#ifndef THREEPP_GLRENDERSTATES_HPP
#define THREEPP_GLRENDERSTATES_HPP

#include "GLLights.hpp"

namespace threepp::gl {

    struct GLRenderState {

        const GLLights& getLights() const;

        const std::vector<Light*>& getLightsArray() const;

        const std::vector<Light*>& getShadowsArray() const;

        void init();

        void pushLight(Light* light);

        void pushShadow(Light* shadowLight);

        void setupLights();

        void setupLightsView(Camera* camera);


    private:
        GLLights lights_;

        std::vector<Light*> lightsArray_;
        std::vector<Light*> shadowsArray_;
    };

    struct GLRenderStates {

        GLRenderState* get(Object3D* scene, size_t renderCallDepth = 1);

        void dispose();

    private:
        std::unordered_map<std::string, std::vector<std::unique_ptr<GLRenderState>>> renderStates_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLRENDERSTATES_HPP
