// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLShader.js

#ifndef THREEPP_GLSHADER_HPP
#define THREEPP_GLSHADER_HPP

#include <glad/glad.h>

#include <string>

namespace threepp::gl {

    unsigned int createShader(int type, const char *str) {

        const auto shader = glCreateShader(type);

        glShaderSource(shader, 1, &str, nullptr);
        glCompileShader(shader);

        return shader;
    }


}// namespace threepp::gl

#endif//THREEPP_GLSHADER_HPP
