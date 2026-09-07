// Backend-neutral render state management (light state, shadow arrays).
// Extracted from renderers/gl/GLRenderStates.hpp — contains zero GL API calls.

#ifndef THREEPP_COMMON_RENDERSTATES_HPP
#define THREEPP_COMMON_RENDERSTATES_HPP

#include "Lights.hpp"

namespace threepp {

    struct RenderState {

        const Lights& getLights() const;

        const std::vector<Light*>& getLightsArray() const;

        const std::vector<Light*>& getShadowsArray() const;

        void init();

        void pushLight(Light* light);

        void pushShadow(Light* shadowLight);

        void setupLights();

        void setupLightsView(Camera* camera);


    private:
        Lights lights_;

        std::vector<Light*> lightsArray_;
        std::vector<Light*> shadowsArray_;
    };

    struct RenderStates {

        RenderState* get(Object3D* scene, size_t renderCallDepth = 1);

        void dispose();

    private:
        std::unordered_map<std::string, std::vector<std::unique_ptr<RenderState>>> renderStates_;
    };

}// namespace threepp

#endif//THREEPP_COMMON_RENDERSTATES_HPP
