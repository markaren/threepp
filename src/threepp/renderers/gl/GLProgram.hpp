// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProgram.js

#ifndef THREEPP_GLPROGRAM_HPP
#define THREEPP_GLPROGRAM_HPP

#include "ProgramParameters.hpp"

#include <memory>
#include <utility>

namespace threepp {

    class GLRenderer;

    namespace gl {

        struct GLBindingStates;
        struct GLUniforms;

        struct GLProgram {

            std::string name;
            int id = programIdCount++;
            std::string cacheKey;
            int usedTimes = 1;
            std::optional<unsigned int> program;
            unsigned int vertexShader;
            unsigned int fragmentShader;

            std::shared_ptr<GLUniforms> getUniforms();

            std::unordered_map<std::string, int> getAttributes();

            void destroy();

            static std::shared_ptr<GLProgram> create(const GLRenderer &renderer, std::string cacheKey, const ProgramParameters &parameters, GLBindingStates &bindingStates);

        private:
            GLBindingStates &bindingStates;
            std::shared_ptr<GLUniforms> cachedUniforms;
            std::unordered_map<std::string, int> cachedAttributes;

            GLProgram(const GLRenderer &renderer, std::string cacheKey, const ProgramParameters &parameters, GLBindingStates &bindingStates);

            inline static int programIdCount = 0;
        };

        inline bool operator==(const GLProgram &p1, const GLProgram &p2) {
            return p1.cacheKey == p2.cacheKey;
        }

        inline bool operator!=(const GLProgram &p1, const GLProgram &p2) {
            return !(p1 == p2);
        }
    }// namespace gl
}// namespace threepp

#endif//THREEPP_GLPROGRAM_HPP
