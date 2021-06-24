
#ifndef THREEPP_GLPARAMETER_HPP
#define THREEPP_GLPARAMETER_HPP

#include <glad/glad.h>

namespace {

    GLint glGetParameter(GLenum id) {
        GLint result;
        glGetIntegerv(id, &result);
        return result;
    }

}

#endif//THREEPP_GLPARAMETER_HPP
