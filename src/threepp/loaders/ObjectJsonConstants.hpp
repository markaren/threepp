// Shared enum <-> three.js numeric-constant tables for the "Object" JSON scene
// format (metadata.version 4.5), used by ObjectExporter and ObjectLoader.
//
// https://github.com/mrdoob/three.js/blob/r129/src/constants.js

#ifndef THREEPP_OBJECTJSONCONSTANTS_HPP
#define THREEPP_OBJECTJSONCONSTANTS_HPP

#include "threepp/constants.hpp"
#include "threepp/core/BufferAttribute.hpp"

#include <algorithm>
#include <filesystem>
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
    // A linked model travels inside the archive as assets/<n>_<filename>, where
    // n numbers the scene's sources in sorted order — so the same scene always
    // produces the same names, and however many subtrees point at one file it is
    // stored once.
    inline constexpr const char* archiveAssetDir = "assets/";

    // What ObjectLoader stamps on a subtree it re-imported OUT of an archive:
    // "<absolute archive path>|<entry name>". The '|' is the whole trick — it
    // cannot occur in a Windows path — so an exporter can always tell a mark
    // from an ordinary path and copy the bytes archive-to-archive instead of
    // hunting for a file that never existed on disk.
    inline constexpr char archiveAssetMark = '|';

    // ── Splat clouds ────────────────────────────────────────────────────────
    // A SplatCloud is written as a `threeppSplat` block: a file to load the
    // splats from plus what the importer did to them, never the splats
    // themselves (a scan is gigabytes; the .ply is the better container). The
    // two userData keys are the editor's SplatImportConfig ones, read here by
    // name so the library needs nothing from editor-core.
    inline constexpr const char* splatSourceKey = "splatSource";
    inline constexpr const char* splatOpsKey = "splatImportOps";
    // Where a cloud's bytes go inside an archive: a copy of its source file,
    // or a freshly written .ply when it never had one.
    inline constexpr const char* archiveSplatDir = "splats/";
    // The sidecar directory next to a loose document, for a cloud with no
    // source file.
    inline constexpr const char* splatSidecarDir = "splats/";

    // Splits a mark into its two halves. False when `text` is an ordinary path,
    // which is the common case and not an error.
    inline bool splitArchiveAsset(const std::string& text, std::string& archive, std::string& entry) {

        const auto bar = text.find(archiveAssetMark);
        if (bar == std::string::npos) return false;

        archive = text.substr(0, bar);
        entry = text.substr(bar + 1);

        return !archive.empty() && !entry.empty();
    }

    // The one format allowed inside an archive. Everything else references
    // siblings the archive does not carry — an .fbx its textures, an .obj its
    // .mtl, a .urdf a whole package tree — and would come back quietly poorer
    // than it went in. Refuse rather than guess; those subtrees keep being
    // written out in full, which the binary sections make cheap anyway.
    inline bool travelsInArchive(const std::string& source) {

        std::string archive, entry;
        const std::string& name = splitArchiveAsset(source, archive, entry) ? entry : source;

        auto extension = std::filesystem::path(name).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        return extension == ".glb";
    }

}// namespace threepp::objectjson

#endif//THREEPP_OBJECTJSONCONSTANTS_HPP
