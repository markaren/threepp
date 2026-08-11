// Shared enum <-> three.js numeric-constant tables for the "Object" JSON scene
// format (metadata.version 4.5), used by ObjectExporter and ObjectLoader.
//
// https://github.com/mrdoob/three.js/blob/r129/src/constants.js

#ifndef THREEPP_OBJECTJSONCONSTANTS_HPP
#define THREEPP_OBJECTJSONCONSTANTS_HPP

#include "threepp/constants.hpp"
#include "threepp/core/BufferAttribute.hpp"

#include <string>

namespace threepp::objectjson {

    // Most threepp enums already carry the three.js numeric value verbatim
    // (Side, Blending, BlendFactor, BlendEquation, DepthFunc, CombineOperation,
    // NormalMapType, DepthPacking, StencilOp, StencilFunc, Mapping,
    // TextureWrapping, Filter, Type, DrawUsage). Those round-trip through
    // as_integer/static_cast. The ones below do not.

    // ---------------------------------------------------------------- Format
    // three.js pixel formats (r129). threepp's Format enum is a plain 0..n
    // enumeration, so it needs an explicit table.
    inline int formatToJson(Format f) {

        switch (f) {
            case Format::Alpha: return 1021;
            case Format::RGB: return 1022;
            case Format::RGBA: return 1023;
            // three.js has no BGR/BGRA; the closest channel count is used and
            // the swizzle is lost (documented deviation).
            case Format::BGR: return 1022;
            case Format::BGRA: return 1023;
            case Format::Luminance: return 1024;
            case Format::LuminanceAlpha: return 1025;
            case Format::Depth: return 1026;
            case Format::DepthStencil: return 1027;
            case Format::Red: return 1028;
            case Format::RedInteger: return 1029;
            case Format::RG: return 1030;
            case Format::RGInteger: return 1031;
            case Format::RGBInteger: return 1032;
            case Format::RGBAInteger: return 1033;
        }
        return 1023;
    }

    inline Format formatFromJson(int v) {

        switch (v) {
            case 1021: return Format::Alpha;
            case 1022: return Format::RGB;
            case 1023: return Format::RGBA;
            case 1024: return Format::Luminance;
            case 1025: return Format::LuminanceAlpha;
            case 1026: return Format::Depth;
            case 1027: return Format::DepthStencil;
            case 1028: return Format::Red;
            case 1029: return Format::RedInteger;
            case 1030: return Format::RG;
            case 1031: return Format::RGInteger;
            case 1032: return Format::RGBInteger;
            case 1033: return Format::RGBAInteger;
            default: return Format::RGBA;
        }
    }

    // -------------------------------------------------------------- Encoding
    // three.js texture encodings. threepp's ColorSpace shares the numeric
    // values except for NoColorSpace (-1), which has no three.js counterpart
    // and is written as LinearEncoding for interop. The exact threepp value is
    // preserved alongside in the additive "threeppColorSpace" key.
    inline int colorSpaceToJsonEncoding(ColorSpace cs) {

        return cs == ColorSpace::NoColorSpace ? 3000 : as_integer(cs);
    }

    inline ColorSpace colorSpaceFromJsonEncoding(int v) {

        switch (v) {
            case 3000: return ColorSpace::Linear;
            case 3001: return ColorSpace::sRGB;
            case 3002: return ColorSpace::RGBE;
            case 3003: return ColorSpace::LogLuv;
            case 3004: return ColorSpace::RGBM7;
            case 3005: return ColorSpace::RGBM16;
            case 3006: return ColorSpace::RGBD;
            case 3007: return ColorSpace::Gamma;
            default: return ColorSpace::NoColorSpace;
        }
    }

    // --------------------------------------------------------- Interpolation
    // three.js: InterpolateDiscrete 2300, InterpolateLinear 2301,
    // InterpolateSmooth 2302.
    inline int interpolationToJson(Interpolation i) {

        switch (i) {
            case Interpolation::Discrete: return 2300;
            case Interpolation::Linear: return 2301;
            case Interpolation::Smooth: return 2302;
        }
        return 2301;
    }

    inline Interpolation interpolationFromJson(int v) {

        switch (v) {
            case 2300: return Interpolation::Discrete;
            case 2302: return Interpolation::Smooth;
            default: return Interpolation::Linear;
        }
    }

    // ------------------------------------------------------------ BlendMode
    // three.js: NormalAnimationBlendMode 2500, AdditiveAnimationBlendMode 2501.
    inline int blendModeToJson(AnimationBlendMode m) {

        return m == AnimationBlendMode::Additive ? 2501 : 2500;
    }

    inline AnimationBlendMode blendModeFromJson(int v) {

        return v == 2501 ? AnimationBlendMode::Additive : AnimationBlendMode::Normal;
    }

    // ------------------------------------------------------- typed array name
    // three.js writes `array.constructor.name`. Each of threepp's six host-side
    // AttributeType values has an exact three.js typed-array counterpart, so the
    // stored integers of a narrowed attribute (see compressAttributes()) go out
    // raw and come back bit-identical. `normalized` carries the [0,1] / [-1,1]
    // mapping in both engines, following the same GL/Vulkan UNORM/SNORM rules,
    // so no decode/re-encode is involved.
    inline const char* attributeTypeToArrayName(AttributeType type) {

        switch (type) {
            case AttributeType::Float: return "Float32Array";
            case AttributeType::UInt32: return "Uint32Array";
            case AttributeType::UInt16: return "Uint16Array";
            case AttributeType::Int16: return "Int16Array";
            case AttributeType::UInt8: return "Uint8Array";
            case AttributeType::Int8: return "Int8Array";
        }
        return "Float32Array";
    }

    // ---------------------------------------------------------- scene archive
    // A .tpz is a zipped project folder, not a second format: the document is
    // the same three.js 4.5 JSON it would be on disk, sitting next to the same
    // images/ and buffers/ it would reference there. These are the names both
    // sides agree on; everything else the loader learns from a url in the JSON.
    inline constexpr const char* archiveDocument = "scene.json";
    inline constexpr const char* archiveImageDir = "images/";
    inline constexpr const char* archiveBufferDir = "buffers/";

}// namespace threepp::objectjson

#endif//THREEPP_OBJECTJSONCONSTANTS_HPP
