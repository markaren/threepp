// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProgram.js

#ifndef THREEPP_GLPROGRAM_HPP
#define THREEPP_GLPROGRAM_HPP

#include "GLUniforms.hpp"
#include "ProgramParameters.hpp"

#include <memory>
#include <utility>

namespace threepp {

    class GLRenderer;

    namespace gl {

        struct GLBindingStates;

        struct GLProgram {

            std::string name;
            int id = programIdCount++;
            std::string cacheKey;
            int usedTimes = 1;
            int program = -1;

            GLProgram(const GLRenderer* renderer, std::string cacheKey, const ProgramParameters* parameters, GLBindingStates* bindingStates);

            GLProgram(const GLProgram&) = delete;
            GLProgram(GLProgram&&) = delete;
            GLProgram& operator=(const GLProgram&) = delete;
            GLProgram& operator=(GLProgram&&) = delete;

            GLUniforms* getUniforms();

            std::unordered_map<std::string, int> getAttributes();

            void destroy();

        protected:
            GLBindingStates* bindingStates = nullptr;
            std::unique_ptr<GLUniforms> cachedUniforms;
            std::unordered_map<std::string, int> cachedAttributes;

            GLProgram() = default;

            inline static int programIdCount{0};
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLPROGRAM_HPP
