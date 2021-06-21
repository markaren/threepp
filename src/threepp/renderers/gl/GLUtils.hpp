// https://github.com/mrdoob/three.js/blob/dev/src/renderers/webgl/WebGLUtils.js

#ifndef THREEPP_GLUTILS_HPP
#define THREEPP_GLUTILS_HPP

#include "glad/glad.h"

#include "threepp/constants.hpp"

#include <optional>

namespace threepp::gl {

    unsigned int convert(int p) {

        if (p) {

            if ( p* == UnsignedByteType ) return GL_UNSIGNED_BYTE;
            if ( p* == UnsignedShort4444Type ) return GL_UNSIGNED_SHORT_4_4_4_4;
            if ( p* == UnsignedShort5551Type ) return GL_UNSIGNED_SHORT_5_5_5_1;
            if ( p* == UnsignedShort565Type ) return GL_UNSIGNED_SHORT_5_6_5;

            if ( p* == ByteType ) return GL_BYTE;
            if ( p* == ShortType ) return GL_SHORT;
            if ( p* == UnsignedShortType ) return GL_UNSIGNED_SHORT;
            if ( p* == IntType ) return GL_INT;
            if ( p* == UnsignedIntType ) return GL_UNSIGNED_INT;
            if ( p* == FloatType ) return GL_FLOAT;

            if ( p* == AlphaFormat ) return GL_ALPHA;
            if ( p* == RGBFormat ) return GL_RGB;
            if ( p* == RGBAFormat ) return GL_RGBA;
            if ( p* == LuminanceFormat ) return GL_LUMINANCE;
            if ( p* == LuminanceAlphaFormat ) return GL_LUMINANCE_ALPHA;
            if ( p* == DepthFormat ) return GL_DEPTH_COMPONENT;
            if ( p* == DepthStencilFormat ) return GL_DEPTH_STENCIL;
            if ( p* == RedFormat ) return GL_RED;

            if ( p* == RedIntegerFormat ) return GL_RED_INTEGER;
            if ( p* == RGFormat ) return GL_RG;
            if ( p* == RGIntegerFormat ) return GL_RG_INTEGER;
            if ( p* == RGBIntegerFormat ) return GL_RGB_INTEGER;
            if ( p* == RGBAIntegerFormat ) return GL_RGBA_INTEGER;

            if ( p* == UnsignedInt248Type ) return GL_UNSIGNED_INT_24_8;

            return 0;

//            if (p* == RepeatWrapping) return GL_REPEAT
//            if (p* == ClampToEdgeWrapping) return GL12.GL_CLAMP_TO_EDGE
//            if (p* == MirroredRepeatWrapping) return GL_MIRRORED_REPEAT
//
//            if (p* == NearestFilter) return GL_NEAREST
//            if (p* == NearestMipMapNearestFilter) return GL_NEAREST_MIPMAP_NEAREST
//            if (p* == NearestMipMapLinearFilter) return GL_NEAREST_MIPMAP_LINEAR
//
//            if (p* == Linear) return GL_LINEAR
//            if (p* == LinearMipMapNearest) return GL_LINEAR_MIPMAP_NEAREST
//            if (p* == LinearMipMapLinear) return GL_LINEAR_MIPMAP_LINEAR
//
//            if (p* == TextureType.UnsignedByte) return GL_UNSIGNED_BYTE
//            if (p* == TextureType.UnsignedShort4444) return GL12.GL_UNSIGNED_SHORT_4_4_4_4
//            if (p* == TextureType.UnsignedShort5551) return GL12.GL_UNSIGNED_SHORT_5_5_5_1
//            if (p* == TextureType.UnsignedShort565) return GL12.GL_UNSIGNED_SHORT_5_6_5
//
//            if (p* == TextureType.Byte) return GL_BYTE
//            if (p* == TextureType.Short) return GL_SHORT
//            if (p* == TextureType.UnsignedShort) return GL_UNSIGNED_SHORT
//            if (p* == TextureType.Int) return GL_INT
//            if (p* == TextureType.UnsignedInt) return GL_UNSIGNED_INT
//            if (p* == TextureType.Float) return GL_FLOAT
//
//            if (p* == TextureType.HalfFloat) {
//                return GL_HALF_FLOAT
//            }
//
//            if (p* == TextureFormat.Alpha) return GL_ALPHA
//            if (p* == TextureFormat.RGB) return GL_RGB
//            if (p* == TextureFormat.RGBA) return GL_RGBA
//            if (p* == TextureFormat.Luminance) return GL_LUMINANCE
//            if (p* == TextureFormat.LuminanceAlpha) return GL_LUMINANCE_ALPHA
//            if (p* == TextureFormat.Depth) return GL_DEPTH_COMPONENT
//            if (p* == TextureFormat.DepthStencil) return GL_DEPTH_STENCIL
//            if (p* == TextureFormat.Red) return GL_RED
//
//            if (p* == BlendingEquation.Add) return GL_FUNC_ADD
//            if (p* == BlendingEquation.Subtract) return GL_FUNC_SUBTRACT
//            if (p* == BlendingEquation.ReverseSubtract) return GL_FUNC_REVERSE_SUBTRACT
//
//            if (p* == BlendingFactor.Zero) return GL_ZERO
//            if (p* == BlendingFactor.One) return GL_ONE
//            if (p* == BlendingFactor.SrcColor) return GL_SRC_COLOR
//            if (p* == BlendingFactor.OneMinusSrcColor) return GL_ONE_MINUS_SRC_COLOR
//            if (p* == BlendingFactor.SrcAlpha) return GL_SRC_ALPHA
//            if (p* == BlendingFactor.OneMinusSrcAlpha) return GL_ONE_MINUS_SRC_ALPHA
//            if (p* == BlendingFactor.DstAlpha) return GL_DST_ALPHA
//            if (p* == BlendingFactor.OneMinusDstAlpha) return GL_ONE_MINUS_DST_ALPHA
//
//            if (p* == BlendingFactor.DstColor) return GL_DST_COLOR
//            if (p* == BlendingFactor.OneMinusDstColor) return GL_ONE_MINUS_DST_COLOR
//            if (p* == BlendingFactor.SrcAlphaSaturate) return GL_SRC_ALPHA_SATURATE
//
//            if (p* == BlendingEquation.Min || p* == BlendingEquation.Max) {
//                if (p* == BlendingEquation.Min) return GL_MIN
//                if (p* == BlendingEquation.Max) return GL_MAX
//            }
//
//            if (p* == TextureType.UnsignedInt248) {
//                return GL_UNSIGNED_INT_24_8
//            }
//
//            return 0
            
        }



    }

}

#endif//THREEPP_GLUTILS_HPP
