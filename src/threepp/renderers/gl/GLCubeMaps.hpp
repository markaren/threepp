
#ifndef THREEPP_GLCUBEMAPS_HPP
#define THREEPP_GLCUBEMAPS_HPP

#include "threepp/textures/CubeTexture.hpp"

#include <unordered_map>

namespace threepp {

    class GLRenderer;
    class GLRenderTarget;

    namespace gl {

        class GLCubeMaps {

        public:
            GLCubeMaps(GLRenderer& renderer);

            Texture* get(Texture* texture);

            void dispose();

        private:
            GLRenderer& renderer;
            std::unordered_map<Texture*, std::shared_ptr<GLRenderTarget>> cubemaps;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLCUBEMAPS_HPP
