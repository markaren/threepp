// https://github.com/mrdoob/three.js/blob/dev/src/renderers/webgl/WebGLUtils.js

#ifndef THREEPP_GLUTILS_HPP
#define THREEPP_GLUTILS_HPP

#ifndef EMSCRIPTEN
#include <glad/glad.h>
#else
#include <GL/gl.h>
#endif

#include "threepp/constants.hpp"

namespace threepp::gl {

    inline GLint glGetParameteri(GLenum id) {
        GLint result;
        glGetIntegerv(id, &result);
        return result;
    }

    inline GLfloat glGetParameterf(GLenum id) {
        GLfloat result;
        glGetFloatv(id, &result);
        return result;
    }

    constexpr GLuint toGLFormat(Format p) {

        switch (p) {
            case Format::Alpha:
                return GL_ALPHA;
            case Format::RGB:
                return GL_RGB;
            case Format::RGBA:
                return GL_RGBA;
            case Format::BGR:
                return GL_BGR;
            case Format::BGRA:
                return GL_BGRA;
            case Format::Luminance:
                return GL_LUMINANCE;
            case Format::LuminanceAlpha:
                return GL_LUMINANCE_ALPHA;
            case Format::Depth:
                return GL_DEPTH_COMPONENT;
            case Format::DepthStencil:
                return GL_DEPTH_STENCIL;
            case Format::Red:
                return GL_RED;
            case Format::RedInteger:
                return GL_RED_INTEGER;
            case Format::RG:
                return GL_RG;
            case Format::RGInteger:
                return GL_RG_INTEGER;
            case Format::RGBInteger:
                return GL_RGB_INTEGER;
            case Format::RGBAInteger:
                return GL_RGBA_INTEGER;
            default:
                return 0;
        }
    }

    constexpr GLuint toGLType(Type p) {

        switch (p) {
            case Type::UnsignedByte:
                return GL_UNSIGNED_BYTE;
            case Type::UnsignedShort4444:
                return GL_UNSIGNED_SHORT_4_4_4_4;
            case Type::UnsignedShort5551:
                return GL_UNSIGNED_SHORT_5_5_5_1;
            case Type::UnsignedShort565:
                return GL_UNSIGNED_SHORT_5_6_5;

            case Type::Byte:
                return GL_BYTE;
            case Type::Short:
                return GL_SHORT;
            case Type::UnsignedShort:
                return GL_UNSIGNED_SHORT;
            case Type::Int:
                return GL_INT;
            case Type::UnsignedInt:
                return GL_UNSIGNED_INT;
            case Type::Float:
                return GL_FLOAT;

            case Type::UnsignedInt248:
                return GL_UNSIGNED_INT_24_8;
            default:
                return 0;
        }
    }

}// namespace threepp::gl

#endif//THREEPP_GLUTILS_HPP
