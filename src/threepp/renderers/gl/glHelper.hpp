
#ifndef THREEPP_GLHELPER_HPP
#define THREEPP_GLHELPER_HPP

#include "glad/glad.h"

namespace {

    GLint glGetParameter(GLenum id) {
        GLint result;
        glGetIntegerv(id, &result);
        return result;
    }

}// namespace

#endif//THREEPP_GLHELPER_HPP
