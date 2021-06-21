// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProgram.js

#ifndef THREEPP_GLPROGRAM_HPP
#define THREEPP_GLPROGRAM_HPP

#include "threepp/renderers/GLRenderer.hpp"

namespace threepp::gl {

    class GLProgram {

    public:
       // GLProgram(std::shared_ptr<GLRenderer> renderer, std::string cacheKey) {}

    private:

        inline static int programIdCount = 0;
    };



}

#endif//THREEPP_GLPROGRAM_HPP
