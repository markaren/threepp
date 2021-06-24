// https://github.com/mrdoob/three.js/blob/dev/src/renderers/webgl/WebGLUtils.js

#ifndef THREEPP_GLUTILS_HPP
#define THREEPP_GLUTILS_HPP

#include "glad/glad.h"

#include "threepp/constants.hpp"

namespace threepp::gl {

    inline GLuint convert(int p) {

        if (p == UnsignedByteType) return GL_UNSIGNED_BYTE;
        if (p == UnsignedShort4444Type) return GL_UNSIGNED_SHORT_4_4_4_4;
        if (p == UnsignedShort5551Type) return GL_UNSIGNED_SHORT_5_5_5_1;
        if (p == UnsignedShort565Type) return GL_UNSIGNED_SHORT_5_6_5;

        if (p == ByteType) return GL_BYTE;
        if (p == ShortType) return GL_SHORT;
        if (p == UnsignedShortType) return GL_UNSIGNED_SHORT;
        if (p == IntType) return GL_INT;
        if (p == UnsignedIntType) return GL_UNSIGNED_INT;
        if (p == FloatType) return GL_FLOAT;

        if (p == AlphaFormat) return GL_ALPHA;
        if (p == RGBFormat) return GL_RGB;
        if (p == RGBAFormat) return GL_RGBA;
        if (p == LuminanceFormat) return GL_LUMINANCE;
        if (p == LuminanceAlphaFormat) return GL_LUMINANCE_ALPHA;
        if (p == DepthFormat) return GL_DEPTH_COMPONENT;
        if (p == DepthStencilFormat) return GL_DEPTH_STENCIL;
        if (p == RedFormat) return GL_RED;

        if (p == RedIntegerFormat) return GL_RED_INTEGER;
        if (p == RGFormat) return GL_RG;
        if (p == RGIntegerFormat) return GL_RG_INTEGER;
        if (p == RGBIntegerFormat) return GL_RGB_INTEGER;
        if (p == RGBAIntegerFormat) return GL_RGBA_INTEGER;

        if (p == UnsignedInt248Type) return GL_UNSIGNED_INT_24_8;

        return 0;
    }

}// namespace threepp::gl

#endif//THREEPP_GLUTILS_HPP
