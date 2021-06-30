// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProgram.js

#ifndef THREEPP_GLPROGRAM_HPP
#define THREEPP_GLPROGRAM_HPP

//#include "GLBindingStates.hpp"

#include "GLUniforms.hpp"

#include <glad/glad.h>

#include <unordered_map>
#include <utility>
#include <optional>

namespace threepp::gl {

    struct GLProgram {

        int id = programIdCount++;

        int usedTimes;

        std::string cacheKey;

        std::optional<GLuint> program;

        GLProgram(std::string cacheKey): cacheKey(std::move(cacheKey))  {}

        GLUniforms &getUniforms();

        void destroy();

    private:

//        GLBindingStates &bindingStates;

        inline static int programIdCount = 0;
    };

    inline bool operator==(const GLProgram &p1, const GLProgram &p2) {
        return p1.cacheKey == p2.cacheKey;
    }

    inline bool operator!=(const GLProgram &p1, const GLProgram &p2) {
        return !(p1 == p2);
    }

}// namespace threepp::gl

#endif//THREEPP_GLPROGRAM_HPP
