// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLShadowMap.js

#ifndef THREEPP_GLSHADOWMAP_HPP
#define THREEPP_GLSHADOWMAP_HPP

#include <memory>
#include <vector>

namespace threepp {

    class GLRenderer;
    class Light;
    class Scene;
    class Camera;

    namespace gl {

        class GLObjects;

        struct GLShadowMap {

            bool enabled = false;

            bool autoUpdate = true;
            bool needsUpdate = false;

            int type;

            explicit GLShadowMap(GLObjects& objects);

            void render(GLRenderer& renderer, const std::vector<Light*>& lights, Scene* scene, Camera* camera);

            ~GLShadowMap();

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLSHADOWMAP_HPP
