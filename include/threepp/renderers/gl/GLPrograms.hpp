// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLPrograms.js

#ifndef THREEPP_GLPROGRAMS_HPP
#define THREEPP_GLPROGRAMS_HPP

#include "GLCapabilities.hpp"
#include "GLClipping.hpp"
#include "GLLights.hpp"
#include "GLProgram.hpp"
#include "ProgramParameters.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <threepp/core/Uniform.hpp>
#include <unordered_map>
#include <vector>

namespace threepp {

    class GLRenderer;

    namespace gl {

        struct GLPrograms {

            std::vector<std::shared_ptr<GLProgram>> programs;

            bool logarithmicDepthBuffer;
            bool floatVertexTextures;
            int maxVertexUniforms;
            bool vertexTextures;

        private:
            GLClipping &clipping;
            GLBindingStates &bindingStates;

        public:
            GLPrograms(GLBindingStates &bindingStates, GLClipping &clipping);

            static ProgramParameters getParameters(
                    const GLRenderer &renderer,
                    Material *material,
                    const GLLights::LightState &lights,
                    size_t numShadows,
                    Scene* scene,
                    Object3D* object);

            static std::string getProgramCacheKey(const GLRenderer &renderer, const ProgramParameters &parameters);

            static std::shared_ptr<UniformMap> getUniforms(Material* material);

            std::shared_ptr<GLProgram> acquireProgram(const GLRenderer &renderer, const ProgramParameters &parameters, const std::string &cacheKey);

            void releaseProgram(const std::shared_ptr<GLProgram> &program);
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLPROGRAMS_HPP
