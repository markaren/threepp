// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLPrograms.js

#ifndef THREEPP_GLPROGRAMS_HPP
#define THREEPP_GLPROGRAMS_HPP

#include "GLClipping.hpp"
#include "GLLights.hpp"
#include "GLProgram.hpp"
#include "ProgramParameters.hpp"

#include "threepp/core/Object3D.hpp"
#include <threepp/core/Uniform.hpp>

#include <memory>
#include <string>

#include <vector>

namespace threepp {

    class GLRenderer;

    namespace gl {

        struct GLPrograms {

            std::vector<std::unique_ptr<GLProgram>> programs;

            bool logarithmicDepthBuffer;
            bool floatVertexTextures;
            int maxVertexUniforms;
            bool vertexTextures;

        private:
            GLClipping& clipping;
            GLBindingStates& bindingStates;

        public:
            GLPrograms(GLBindingStates& bindingStates, GLClipping& clipping);

            static ProgramParameters getParameters(
                    const GLRenderer& renderer,
                    const GLClipping& clipping,
                    Material* material,
                    const GLLights::LightState& lights,
                    size_t numShadows,
                    Scene* scene,
                    Object3D* object);

            static std::string getProgramCacheKey(const GLRenderer& renderer, const ProgramParameters& parameters);

            static UniformMap* getUniforms(Material& material);

            GLProgram* acquireProgram(const GLRenderer& renderer, const ProgramParameters& parameters, const std::string& cacheKey);

            void releaseProgram(GLProgram* program);
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLPROGRAMS_HPP
