// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLShadowMap.js

#ifndef THREEPP_GLSHADOWMAP_HPP
#define THREEPP_GLSHADOWMAP_HPP

#include "threepp/constants.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class GLRenderer;
    class Light;
    class Object3D;
    class Camera;

    namespace gl {

        class GLObjects;

        struct GLShadowMap {

            bool enabled = false;

            bool autoUpdate = true;
            bool needsUpdate = false;

            ShadowMap type;

            explicit GLShadowMap(GLObjects& objects);

            void render(GLRenderer& renderer, const std::vector<Light*>& lights, Object3D* scene, Camera* camera);

            ~GLShadowMap();

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLSHADOWMAP_HPP
