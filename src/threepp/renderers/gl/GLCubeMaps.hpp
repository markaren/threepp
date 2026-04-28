
#ifndef THREEPP_GLCUBEMAPS_HPP
#define THREEPP_GLCUBEMAPS_HPP

#include "threepp/renderers/GLCubeRenderTarget.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <memory>
#include <unordered_map>

namespace threepp {

    namespace gl {

        class GLPMREM;

        class GLCubeMaps {

        public:
            explicit GLCubeMaps(GLRenderer& renderer);
            ~GLCubeMaps();

            // Returns the converted CubeTexture for equirectangular inputs,
            // the original texture for anything else, or nullptr if not ready.
            // Mirrors three.js WebGLCubeMaps.get().
            Texture* get(Texture* texture);

            // Returns a PMREM (CubeUV-packed 2D) texture for equirectangular
            // inputs used as IBL / envMap. Falls back to the original texture
            // for non-equirect inputs (pre-baked cube PMREMs etc.).
            Texture* getPMREM(Texture* texture);

            void dispose();

        private:
            GLRenderer& renderer;
            std::unordered_map<Texture*, std::unique_ptr<GLCubeRenderTarget>> cubemaps;
            std::unordered_map<Texture*, std::unique_ptr<RenderTarget>> pmrems;
            std::unique_ptr<GLPMREM> pmremGenerator;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLCUBEMAPS_HPP
