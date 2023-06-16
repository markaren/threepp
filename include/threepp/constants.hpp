// https://github.com/mrdoob/three.js/blob/r129/src/constants.js

#ifndef THREEPP_CONSTANTS_HPP
#define THREEPP_CONSTANTS_HPP

#include <type_traits>

namespace threepp {

    // https://stackoverflow.com/questions/11421432/how-can-i-output-the-value-of-an-enum-class-in-c11
    template<typename Enumeration>
    auto as_integer(const Enumeration value)
            -> typename std::underlying_type<Enumeration>::type {
        return static_cast<typename std::underlying_type<Enumeration>::type>(value);
    }

    const int CullFaceNone = 0;
    const int CullFaceBack = 1;
    const int CullFaceFront = 2;
    const int CullFaceFrontBack = 3;

    //    const int BasicShadowMap = 0;
    //    const int PCFShadowMap = 1;
    //    const int PCFSoftShadowMap = 2;
    //    const int VSMShadowMap = 3;

    enum class ShadowMap {
        Basic,
        PFC,
        PFCSoft,
        VSM
    };

    //    const int FrontSide = 0;
    //    const int BackSide = 1;
    //    const int DoubleSide = 2;

    enum class Side {
        Front,
        Back,
        Double
    };

    const int FlatShading = 1;
    const int SmoothShading = 2;
    const int NoBlending = 0;
    const int NormalBlending = 1;
    const int AdditiveBlending = 2;
    const int SubtractiveBlending = 3;
    const int MultiplyBlending = 4;
    const int CustomBlending = 5;
    const int AddEquation = 100;
    const int SubtractEquation = 101;
    const int ReverseSubtractEquation = 102;
    const int MinEquation = 103;
    const int MaxEquation = 104;
    const int ZeroFactor = 200;
    const int OneFactor = 201;
    const int SrcColorFactor = 202;
    const int OneMinusSrcColorFactor = 203;
    const int SrcAlphaFactor = 204;
    const int OneMinusSrcAlphaFactor = 205;
    const int DstAlphaFactor = 206;
    const int OneMinusDstAlphaFactor = 207;
    const int DstColorFactor = 208;
    const int OneMinusDstColorFactor = 209;
    const int SrcAlphaSaturateFactor = 210;
    const int NeverDepth = 0;
    const int AlwaysDepth = 1;
    const int LessDepth = 2;
    const int LessEqualDepth = 3;
    const int EqualDepth = 4;
    const int GreaterEqualDepth = 5;
    const int GreaterDepth = 6;
    const int NotEqualDepth = 7;
    const int MultiplyOperation = 0;
    const int MixOperation = 1;
    const int AddOperation = 2;
    //    const int NoToneMapping = 0;
    //    const int LinearToneMapping = 1;
    //    const int ReinhardToneMapping = 2;
    //    const int CineonToneMapping = 3;
    //    const int ACESFilmicToneMapping = 4;
    //    const int CustomToneMapping = 5;

    enum class ToneMapping : int {
        None = 0,
        Linear = 1,
        Reinhard = 2,
        Cineon = 3,
        ACESFilmic = 4,
        Custom = 5
    };

    const int UVMapping = 300;
    const int CubeReflectionMapping = 301;
    const int CubeRefractionMapping = 302;
    const int EquirectangularReflectionMapping = 303;
    const int EquirectangularRefractionMapping = 304;
    const int CubeUVReflectionMapping = 306;
    const int CubeUVRefractionMapping = 307;

    //    const int RepeatWrapping = 1000;
    //    const int ClampToEdgeWrapping = 1001;
    //    const int MirroredRepeatWrapping = 1002;

    enum class TextureWrapping : int {
        Repeat = 1000,
        ClampToEdge = 1001,
        MirroredRepeat = 1002
    };

    const int NearestFilter = 1003;
    const int NearestMipmapNearestFilter = 1004;
    const int NearestMipMapNearestFilter = 1004;
    const int NearestMipmapLinearFilter = 1005;
    const int NearestMipMapLinearFilter = 1005;
    const int LinearFilter = 1006;
    const int LinearMipmapNearestFilter = 1007;
    const int LinearMipMapNearestFilter = 1007;
    const int LinearMipmapLinearFilter = 1008;
    const int LinearMipMapLinearFilter = 1008;
    const int UnsignedByteType = 1009;
    const int ByteType = 1010;
    const int ShortType = 1011;
    const int UnsignedShortType = 1012;
    const int IntType = 1013;
    const int UnsignedIntType = 1014;
    const int FloatType = 1015;
    const int HalfFloatType = 1016;
    const int UnsignedShort4444Type = 1017;
    const int UnsignedShort5551Type = 1018;
    const int UnsignedShort565Type = 1019;
    const int UnsignedInt248Type = 1020;
    const int AlphaFormat = 1021;
    const int RGBFormat = 1022;
    const int RGBAFormat = 1023;
    const int LuminanceFormat = 1024;
    const int LuminanceAlphaFormat = 1025;
    const int RGBEFormat = RGBAFormat;
    const int DepthFormat = 1026;
    const int DepthStencilFormat = 1027;
    const int RedFormat = 1028;
    const int RedIntegerFormat = 1029;
    const int RGFormat = 1030;
    const int RGIntegerFormat = 1031;
    const int RGBIntegerFormat = 1032;
    const int RGBAIntegerFormat = 1033;

    const int LoopOnce = 2200;
    const int LoopRepeat = 2201;
    const int LoopPingPong = 2202;
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

    //    const int LinearEncoding = 3000;
    //    const int sRGBEncoding = 3001;
    //    const int GammaEncoding = 3007;
    //    const int RGBEEncoding = 3002;
    //    const int LogLuvEncoding = 3003;
    //    const int RGBM7Encoding = 3004;
    //    const int RGBM16Encoding = 3005;
    //    const int RGBDEncoding = 3006;

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

    const int BasicDepthPacking = 3200;
    const int RGBADepthPacking = 3201;
    const int TangentSpaceNormalMap = 0;
    const int ObjectSpaceNormalMap = 1;

    const int ZeroStencilOp = 0;
    const int KeepStencilOp = 7680;
    const int ReplaceStencilOp = 7681;
    const int IncrementStencilOp = 7682;
    const int DecrementStencilOp = 7683;
    const int IncrementWrapStencilOp = 34055;
    const int DecrementWrapStencilOp = 34056;
    const int InvertStencilOp = 5386;

    const int NeverStencilFunc = 512;
    const int LessStencilFunc = 513;
    const int EqualStencilFunc = 514;
    const int LessEqualStencilFunc = 515;
    const int GreaterStencilFunc = 516;
    const int NotEqualStencilFunc = 517;
    const int GreaterEqualStencilFunc = 518;
    const int AlwaysStencilFunc = 519;

    //    const int StaticDrawUsage = 35044;
    //    const int DynamicDrawUsage = 35048;
    //    const int StreamDrawUsage = 35040;

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
