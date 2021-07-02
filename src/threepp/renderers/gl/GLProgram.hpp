// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProgram.js

#ifndef THREEPP_GLPROGRAM_HPP
#define THREEPP_GLPROGRAM_HPP

#include "ProgramParameters.hpp"

#include <memory>

namespace threepp::gl {

    struct GLBindingStates;
    struct GLUniforms;

    struct GLProgram {

        int id = programIdCount++;

        int usedTimes;

        std::string cacheKey;

        std::optional<unsigned int> program;

        GLProgram(std::string cacheKey, const ProgramParameters &parameters, GLBindingStates& bindingStates);

        std::shared_ptr<GLUniforms> getUniforms();

        std::unordered_map<std::string, int> getAttributes();

        void destroy();

    private:

        struct Impl;
        std::unique_ptr<Impl> pimpl_;

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
