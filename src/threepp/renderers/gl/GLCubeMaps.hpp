
#ifndef THREEPP_GLCUBEMAPS_HPP
#define THREEPP_GLCUBEMAPS_HPP

#include "threepp/renderers/GLCubeRenderTarget.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <unordered_map>

namespace threepp {

    namespace gl {

        class GLCubeMaps {

        public:
            explicit GLCubeMaps(GLRenderer& renderer);

            void get(Texture* texture);

            void dispose();

        private:
            GLRenderer& renderer;
            std::unordered_map<Texture*, std::unique_ptr<GLCubeRenderTarget>> cubemaps;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLCUBEMAPS_HPP
