// https://github.com/mrdoob/three.js/blob/r129/src/constants.js

#ifndef THREEPP_CONSTANTS_HPP
#define THREEPP_CONSTANTS_HPP

#include <type_traits>

namespace threepp {

    // https://stackoverflow.com/questions/11421432/how-can-i-output-the-value-of-an-enum-class-in-c11
    template<typename Enumeration>
    constexpr auto as_integer(const Enumeration value)
            -> typename std::underlying_type<Enumeration>::type {
        return static_cast<typename std::underlying_type<Enumeration>::type>(value);
    }

    enum class CullFace {
        None = 0,
        Back = 1,
        Front = 2,
        FrontBack = 3
    };

    enum class ShadowMap {
        Basic,
        PFC,
        PFCSoft,
        VSM
    };

    enum class Side {
        Front,
        Back,
        Double
    };

    const int FlatShading = 1;
    const int SmoothShading = 2;

    enum class Blending {
        None = 0,
        Normal = 1,
        Additive = 2,
        Subtractive = 3,
        Multiply = 4,
        Custom = 5
    };

    enum class BlendEquation {
        Add = 100,
        Subtract = 101,
        ReverseSubtract = 102,
        Min = 103,
        Max = 104
    };

    enum class BlendFactor {
        Zero = 200,
        One = 201,
        SrcColor = 202,
        OneMinusSrcColor = 203,
        SrcAlpha = 204,
        OneMinusSrcAlpha = 205,
        DstAlpha = 206,
        OneMinusDstAlpha = 207,
        DstColor = 208,
        OneMinusDstColor = 209,
        SrcAlphaSaturate = 210
    };

    enum class DepthFunc {
        Never = 0,
        Always = 1,
        Less = 2,
        LessEqual = 3,
        Equal = 4,
        GreaterEqual = 5,
        Greater = 6,
        NotEqual = 7
    };

    enum class CombineOperation {
        Multiply = 0,
        Mix = 1,
        Add = 2
    };

    enum class ToneMapping : int {
        None = 0,
        Linear = 1,
        Reinhard = 2,
        Cineon = 3,
        ACESFilmic = 4,
        Custom = 5
    };

    enum class Mapping {
        UV = 300,
        CubeReflection = 301,
        CubeRefraction = 302,
        EquirectangularReflection = 303,
        EquirectangularRefraction = 304,
        CubeUVReflection = 306,
        CubeUVRefraction = 307
    };

    enum class TextureWrapping : int {
        Repeat = 1000,
        ClampToEdge = 1001,
        MirroredRepeat = 1002
    };

    enum class Filter {
        Nearest = 1003,
        NearestMipmapNearest = 1004,
        NearestMipMapNearest = 1004,
        NearestMipmapLinear = 1005,
        NearestMipMapLinear = 1005,
        Linear = 1006,
        LinearMipmapNearest = 1007,
        LinearMipMapNearest = 1007,
        LinearMipmapLinear = 1008,
        LinearMipMapLinear = 1008
    };

    enum class Type {
        UnsignedByte = 1009,
        Byte = 1010,
        Short = 1011,
        UnsignedShort = 1012,
        Int = 1013,
        UnsignedInt = 1014,
        Float = 1015,
        HalfFloat = 1016,
        UnsignedShort4444 = 1017,
        UnsignedShort5551 = 1018,
        UnsignedShort565 = 1019,
        UnsignedInt248 = 1020
    };

    enum class Format {
        Alpha,
        RGB,
        RGBA,
        Luminance,
        LuminanceAlpha,
        Depth,
        DepthStencil,
        Red,
        RedInteger,
        RG,
        RGInteger,
        RGBInteger,
        RGBAInteger
    };

    enum class Loop {
        Once = 2200,
        Repeat = 2201,
        PingPong = 2202
    };

    const int InterpolateDiscrete = 2300;
    const int InterpolateLinear = 2301;
    const int InterpolateSmooth = 2302;
    const int ZeroCurvatureEnding = 2400;
    const int ZeroSlopeEnding = 2401;
    const int WrapAroundEnding = 2402;
    const int NormalAnimationBlendMode = 2500;
    const int AdditiveAnimationBlendMode = 2501;
    const int TrianglesDrawMode = 0;
    const int TriangleStripDrawMode = 1;
    const int TriangleFanDrawMode = 2;

    enum class Encoding : int {
        Linear = 3000,
        sRGB = 3001,
        Gamma = 3007,
        RGBE = 3002,
        LogLuv = 3003,
        RGBM7 = 3004,
        RGBM16 = 3005,
        RGBD = 3006
    };

    enum class DepthPacking {
        Basic = 3200,
        RGBA = 3201
    };

    const int TangentSpaceNormalMap = 0;
    const int ObjectSpaceNormalMap = 1;

    enum class StencilOp {
        Zero = 0,
        Keep = 7680,
        Replace = 7681,
        Increment = 7682,
        Decrement = 7683,
        IncrementWrap = 34055,
        DecrementWrap = 34056,
        Invert = 5386
    };

    enum class StencilFunc {
        Never = 512,
        Less = 513,
        Equal = 514,
        LessEqual = 515,
        Greater = 516,
        NotEqual = 517,
        GreaterEqual = 518,
        Always = 519

    };

    enum class DrawUsage : int {
        Static = 35044,
        Dynamic = 35048,
        Stream = 35040
    };

    const int StaticReadUsage = 35045;
    const int DynamicReadUsage = 35049;
    const int StreamReadUsage = 35041;
    const int StaticCopyUsage = 35046;
    const int DynamicCopyUsage = 35050;
    const int StreamCopyUsage = 35042;

}// namespace threepp

#endif//THREEPP_CONSTANTS_HPP
