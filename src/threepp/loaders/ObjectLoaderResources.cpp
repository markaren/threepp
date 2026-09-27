// https://github.com/mrdoob/three.js/blob/r129/src/loaders/ObjectLoader.js
// https://github.com/mrdoob/three.js/blob/r129/src/loaders/MaterialLoader.js
// https://github.com/mrdoob/three.js/blob/r129/src/loaders/BufferGeometryLoader.js

// The document's resource tables - images, textures, materials, geometries and
// animations - built before ObjectLoader.cpp assembles the object tree from them.
// A translation unit of its own because it instantiates nlohmann::json and the
// standard containers over a different set of types than the object tree does:
// as one file the two came to more sections than a MinGW Debug object can hold.

#include "ObjectJsonParse.hpp"

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/tracks/BooleanKeyframeTrack.hpp"
#include "threepp/animation/tracks/ColorKeyframeTrack.hpp"
#include "threepp/animation/tracks/NumberKeyframeTrack.hpp"
#include "threepp/animation/tracks/QuaternionKeyframeTrack.hpp"
#include "threepp/animation/tracks/StringKeyframeTrack.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/geometries.hpp"
#include "threepp/geometries/LatheGeometry.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/PolyhedronGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/loaders/AssetSource.hpp"
#include "threepp/loaders/SogLoader.hpp"
#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatLod.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/loaders/Xacro.hpp"
#include "threepp/materials/materials.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshMatcapMaterial.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include "ObjectJsonConstants.hpp"
#include "threepp/utils/Base64.hpp"

#include <cstring>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace threepp;
using namespace threepp::objectjson;

namespace {

    // The bytes of one `url`/`byteOffset`/`byteLength` window, or nothing.
    //
    // Every bound is checked against the section that was actually loaded, not
    // against what the JSON claims about it: a document and its sections can
    // disagree — an archive edited by hand, a truncated file, a document paired
    // with the wrong buffers — and a memcpy that trusts the JSON reads whatever
    // happens to follow in memory.
    const unsigned char* binaryWindow(const json& j, std::size_t elementSize, int itemSize,
                                      BufferCache& buffers, Warnings& warnings, std::size_t& count) {

        const auto url = value<std::string>(j, "url", "");
        const auto* section = buffers.get(url, warnings);
        if (!section) return nullptr;

        const auto offset = value<std::uint64_t>(j, "byteOffset", 0);
        const auto length = value<std::uint64_t>(j, "byteLength", 0);

        // Subtraction rather than offset + length, so a length near the 64-bit
        // ceiling cannot wrap the comparison.
        if (offset > section->size() || length > section->size() - offset) {

            warnings.add("attribute claims " + std::to_string(length) + " bytes at offset " +
                         std::to_string(offset) + " of '" + url + "', which is " +
                         std::to_string(section->size()) + " bytes - skipped");
            return nullptr;
        }

        if (elementSize == 0 || length % elementSize != 0 ||
            itemSize <= 0 || (length / elementSize) % static_cast<std::uint64_t>(itemSize) != 0) {

            warnings.add("attribute of " + std::to_string(length) + " bytes in '" + url +
                         "' is not a whole number of " + std::to_string(itemSize) +
                         "-component items - skipped");
            return nullptr;
        }

        count = static_cast<std::size_t>(length / elementSize);

        return section->data() + offset;
    }

    template<class T>
    std::optional<std::vector<T>> readBinaryArray(const json& j, int itemSize,
                                                  BufferCache& buffers, Warnings& warnings) {

        std::size_t count = 0;
        const auto* first = binaryWindow(j, sizeof(T), itemSize, buffers, warnings, count);
        if (!first) return std::nullopt;

        std::vector<T> out(count);
        // memcpy rather than a typed pointer cast: the section is a byte array
        // whose alignment is whatever the allocator gave it, and the offsets are
        // only padded to 4.
        if (count > 0) std::memcpy(out.data(), first, count * sizeof(T));

        return out;
    }

    // -------------------------------------------------------------- images

    // The two `url` forms are two different contracts, and they do not want the same row order.
    //
    //   data URI   ObjectExporter::writeImage emitted the texture's rows VERBATIM, and
    //              writeTexture declares `flipY: false` beside them - the bytes are already in
    //              final order. Flipping here would invert every embedded texture in every
    //              document ever saved.
    //   file path  ImageStorage::Reference stores the path the texture was ORIGINALLY loaded
    //              from. Re-reading it is an import, and every importer in the tree
    //              (TextureLoader, ImageLoader, RGBELoader, EXRLoader) defaults flipY = true.
    //              Decoding it with false hands back the opposite of the texture that was saved.
    //
    // Both used to pass `false`, so a Reference-mode save came back with every texture upside
    // down - and so did any document referencing an image by path, which is how an environment
    // map is most naturally written.
    //
    // An explicit `flipY` on the image entry overrides either default — which is
    // how the archive says which of the two it holds. A member copied out of the
    // texture's source file is an import and takes the path default (true); one
    // the exporter had to re-encode carries flipY: false beside it, because
    // those rows went in verbatim.
    std::vector<Image> decodeImage(const json& entry, const json& url, int channels,
                                   const DocumentSource& source) {

        ImageLoader loader;
        std::vector<Image> out;

        const bool hasOverride = entry.contains("flipY") && entry["flipY"].is_boolean();
        const bool override = hasOverride && entry["flipY"].get<bool>();

        const auto decodeOne = [&](const std::string& u) -> std::optional<Image> {
            if (u.rfind("data:", 0) == 0) {
                const auto comma = u.find(',');
                if (comma == std::string::npos) return std::nullopt;
                const auto bytes = utils::base64Decode(u.substr(comma + 1));
                return loader.load(bytes, channels, hasOverride ? override : false);
            }
            // Falls through to the path when the archive does not hold the
            // name: a document inside an archive may still reference a file
            // outside it, and that url means what it always meant.
            if (source.archive && source.archive->has(u)) {
                auto bytes = source.read(u);
                if (!bytes) return std::nullopt;
                return loader.load(*bytes, channels, hasOverride ? override : true);
            }
            return loader.load(source.resourcePath.empty() ? fs::path(u) : source.resourcePath / u, channels,
                               hasOverride ? override : true);
        };

        if (url.is_array()) {
            for (const auto& entry : url) {
                if (!entry.is_string()) continue;
                auto image = decodeOne(entry.get<std::string>());
                if (image) out.push_back(std::move(*image));
            }
        } else if (url.is_string()) {
            auto image = decodeOne(url.get<std::string>());
            if (image) out.push_back(std::move(*image));
        }

        return out;
    }

    // ----------------------------------------------------------- materials

    std::shared_ptr<Material> createMaterial(const std::string& type) {

        if (type == "MeshStandardMaterial") return MeshStandardMaterial::create();
        if (type == "MeshPhysicalMaterial") return MeshPhysicalMaterial::create();
        if (type == "MeshBasicMaterial") return MeshBasicMaterial::create();
        if (type == "MeshPhongMaterial") return MeshPhongMaterial::create();
        if (type == "MeshLambertMaterial") return MeshLambertMaterial::create();
        if (type == "MeshToonMaterial") return MeshToonMaterial::create();
        if (type == "MeshNormalMaterial") return MeshNormalMaterial::create();
        if (type == "MeshDepthMaterial") return MeshDepthMaterial::create();
        if (type == "MeshMatcapMaterial") return MeshMatcapMaterial::create();
        if (type == "LineBasicMaterial") return LineBasicMaterial::create();
        if (type == "LineDashedMaterial") return LineDashedMaterial::create();
        if (type == "PointsMaterial") return PointsMaterial::create();
        if (type == "SpriteMaterial") return SpriteMaterial::create();
        if (type == "ShadowMaterial") return ShadowMaterial::create();
        if (type == "ShaderMaterial") return ShaderMaterial::create();
        if (type == "RawShaderMaterial") return RawShaderMaterial::create();

        return nullptr;
    }

    std::shared_ptr<Texture> lookupTexture(const json& j, const char* key, const TextureMap& textures, Warnings& warnings) {

        if (!j.contains(key) || !j[key].is_string()) return nullptr;

        const auto it = textures.find(j[key].get<std::string>());
        if (it == textures.end()) {
            warnings.add("undefined texture '" + j[key].get<std::string>() + "'");
            return nullptr;
        }
        return it->second;
    }

    // Builds the setValues() payload, restricted to the keys the concrete
    // material actually understands (probed through the capability mixins), so
    // three.js documents carrying keys threepp lacks stay silent.
    std::unordered_map<std::string, MaterialValue> collectMaterialValues(
            Material& material, const json& j, const TextureMap& textures, Warnings& warnings) {

        std::unordered_map<std::string, MaterialValue> values;

        const auto setFloat = [&](const char* key) {
            if (j.contains(key) && j[key].is_number()) values[key] = j[key].get<float>();
        };
        const auto setBool = [&](const char* key) {
            if (j.contains(key) && j[key].is_boolean()) values[key] = j[key].get<bool>();
        };
        const auto setColor = [&](const char* key) {
            if (j.contains(key) && j[key].is_number()) values[key] = colorFrom(j[key]);
        };
        const auto setVector2 = [&](const char* key) {
            if (j.contains(key)) values[key] = vector2From(j[key], {1, 1});
        };
        const auto setTexture = [&](const char* key) {
            if (auto texture = lookupTexture(j, key, textures, warnings)) values[key] = texture;
        };

        if (dynamic_cast<MaterialWithColor*>(&material)) setColor("color");
        if (dynamic_cast<MaterialWithRoughness*>(&material)) {
            setFloat("roughness");
            setTexture("roughnessMap");
        }
        if (dynamic_cast<MaterialWithMetalness*>(&material)) {
            setFloat("metalness");
            setTexture("metalnessMap");
        }
        if (dynamic_cast<MaterialWithSheen*>(&material)) {
            setColor("sheenColor");
            setFloat("sheenRoughness");
        }
        if (dynamic_cast<MaterialWithEmissive*>(&material)) {
            setColor("emissive");
            setFloat("emissiveIntensity");
            setTexture("emissiveMap");
        }
        if (dynamic_cast<MaterialWithSpecular*>(&material)) {
            setColor("specular");
            setFloat("shininess");
        }
        if (dynamic_cast<MaterialWithClearcoat*>(&material)) {
            setFloat("clearcoat");
            setFloat("clearcoatRoughness");
            setTexture("clearcoatMap");
            setTexture("clearcoatRoughnessMap");
            setTexture("clearcoatNormalMap");
            if (j.contains("clearcoatNormalScale")) setVector2("clearcoatNormalScale");
        }
        if (dynamic_cast<MaterialWithTransmission*>(&material)) {
            setFloat("transmission");
            setFloat("ior");
            setFloat("dispersion");
            setTexture("transmissionMap");
        }
        if (dynamic_cast<MaterialWithThickness*>(&material)) {
            setFloat("thickness");
            setBool("thinWalled");
            setTexture("thicknessMap");
        }
        if (dynamic_cast<MaterialWithAttenuation*>(&material)) {
            setFloat("attenuationDistance");
            setColor("attenuationColor");
            setFloat("scatterDistance");
            setColor("scatterColor");
        }
        if (dynamic_cast<MaterialWithIridescence*>(&material)) {
            setFloat("iridescence");
            setFloat("iridescenceIOR");
            setFloat("iridescenceThicknessNm");
        }
        if (dynamic_cast<MaterialWithPbrSpecular*>(&material)) {
            setFloat("specularIntensity");
            setColor("specularColor");
        }
        if (dynamic_cast<MaterialWithMap*>(&material)) setTexture("map");
        if (dynamic_cast<MaterialWithMatCap*>(&material)) setTexture("matcap");
        if (dynamic_cast<MaterialWithAlphaMap*>(&material)) setTexture("alphaMap");
        if (dynamic_cast<MaterialWithLightMap*>(&material)) {
            setTexture("lightMap");
            setFloat("lightMapIntensity");
        }
        if (dynamic_cast<MaterialWithAoMap*>(&material)) {
            setTexture("aoMap");
            setFloat("aoMapIntensity");
        }
        if (dynamic_cast<MaterialWithBumpMap*>(&material)) {
            setTexture("bumpMap");
            setFloat("bumpScale");
        }
        if (dynamic_cast<MaterialWithNormalMap*>(&material)) {
            setTexture("normalMap");
            if (j.contains("normalMapType")) values["normalMapType"] = static_cast<NormalMapType>(j["normalMapType"].get<int>());
            if (j.contains("normalScale")) setVector2("normalScale");
        }
        if (dynamic_cast<MaterialWithDisplacementMap*>(&material)) {
            setTexture("displacementMap");
            setFloat("displacementScale");
            setFloat("displacementBias");
        }
        if (dynamic_cast<MaterialWithSpecularMap*>(&material)) setTexture("specularMap");
        if (dynamic_cast<MaterialWithEnvMap*>(&material)) {
            setTexture("envMap");
            setFloat("envMapIntensity");
        }
        if (dynamic_cast<MaterialWithReflectivity*>(&material)) setFloat("reflectivity");
        if (dynamic_cast<MaterialWithRefractionRatio*>(&material)) setFloat("refractionRatio");
        if (dynamic_cast<MaterialWithCombine*>(&material) && j.contains("combine")) {
            values["combine"] = static_cast<CombineOperation>(j["combine"].get<int>());
        }
        if (dynamic_cast<MaterialWithGradientMap*>(&material)) setTexture("gradientMap");
        if (dynamic_cast<MaterialWithSize*>(&material)) {
            setFloat("size");
            setBool("sizeAttenuation");
        }
        if (dynamic_cast<MaterialWithLineWidth*>(&material)) setFloat("linewidth");
        if (dynamic_cast<LineDashedMaterial*>(&material)) {
            setFloat("dashSize");
            setFloat("gapSize");
            setFloat("scale");
        }
        if (dynamic_cast<MaterialWithRotation*>(&material)) setFloat("rotation");
        if (dynamic_cast<MaterialWithWireframe*>(&material)) {
            setBool("wireframe");
            setFloat("wireframeLinewidth");
        }
        if (dynamic_cast<MaterialWithFlatShading*>(&material)) setBool("flatShading");
        if (dynamic_cast<MaterialWithVertexTangents*>(&material)) setBool("vertexTangents");
        if (dynamic_cast<MaterialWithDepthPacking*>(&material) && j.contains("depthPacking")) {
            values["depthPacking"] = static_cast<DepthPacking>(j["depthPacking"].get<int>());
        }
        if (dynamic_cast<MaterialWithClipping*>(&material)) setBool("clipping");
        if (dynamic_cast<MaterialWithLights*>(&material)) setBool("lights");
        if (dynamic_cast<ShaderMaterial*>(&material)) {
            if (j.contains("vertexShader")) values["vertexShader"] = j["vertexShader"].get<std::string>();
            if (j.contains("fragmentShader")) values["fragmentShader"] = j["fragmentShader"].get<std::string>();
        }

        // ---- base Material state (present on every subclass)
        setBool("fog");
        setBool("vertexColors");
        setFloat("opacity");
        setBool("transparent");
        setBool("depthTest");
        setBool("depthWrite");
        setBool("colorWrite");
        setBool("stencilWrite");
        setBool("polygonOffset");
        setFloat("polygonOffsetFactor");
        setFloat("polygonOffsetUnits");
        setBool("dithering");
        setFloat("alphaTest");
        setBool("alphaToCoverage");
        setBool("alphaHash");
        setBool("forceSinglePass");
        setBool("premultipliedAlpha");
        setBool("visible");
        setBool("toneMapped");

        if (j.contains("blending")) values["blending"] = static_cast<Blending>(j["blending"].get<int>());
        if (j.contains("side")) values["side"] = static_cast<Side>(j["side"].get<int>());
        if (j.contains("shadowSide")) values["shadowSide"] = static_cast<Side>(j["shadowSide"].get<int>());
        if (j.contains("blendSrc")) values["blendSrc"] = static_cast<BlendFactor>(j["blendSrc"].get<int>());
        if (j.contains("blendDst")) values["blendDst"] = static_cast<BlendFactor>(j["blendDst"].get<int>());
        if (j.contains("blendEquation")) values["blendEquation"] = static_cast<BlendEquation>(j["blendEquation"].get<int>());
        if (j.contains("blendSrcAlpha") && !j["blendSrcAlpha"].is_null()) values["blendSrcAlpha"] = static_cast<BlendFactor>(j["blendSrcAlpha"].get<int>());
        if (j.contains("blendDstAlpha") && !j["blendDstAlpha"].is_null()) values["blendDstAlpha"] = static_cast<BlendFactor>(j["blendDstAlpha"].get<int>());
        if (j.contains("blendEquationAlpha") && !j["blendEquationAlpha"].is_null()) values["blendEquationAlpha"] = static_cast<BlendEquation>(j["blendEquationAlpha"].get<int>());
        if (j.contains("depthFunc")) values["depthFunc"] = static_cast<DepthFunc>(j["depthFunc"].get<int>());
        if (j.contains("stencilWriteMask")) values["stencilWriteMask"] = j["stencilWriteMask"].get<int>();
        if (j.contains("stencilRef")) values["stencilRef"] = j["stencilRef"].get<int>();
        if (j.contains("stencilFuncMask")) values["stencilFuncMask"] = j["stencilFuncMask"].get<int>();
        if (j.contains("stencilFunc")) values["stencilFunc"] = static_cast<StencilFunc>(j["stencilFunc"].get<int>());
        if (j.contains("stencilFail")) values["stencilFail"] = static_cast<StencilOp>(j["stencilFail"].get<int>());
        if (j.contains("stencilZFail")) values["stencilZFail"] = static_cast<StencilOp>(j["stencilZFail"].get<int>());
        if (j.contains("stencilZPass")) values["stencilZPass"] = static_cast<StencilOp>(j["stencilZPass"].get<int>());

        return values;
    }

    // The threepp-only material fields (see the matching writer in
    // ObjectExporter). These have no three.js key and no setValue() entry, so
    // they are applied straight onto the object rather than through setValues().
    void applyThreeppMaterialExtensions(
            Material& material, const json& j, const TextureMap& textures, Warnings& warnings) {

        const auto getFloat = [&](const char* key, float& target) {
            if (j.contains(key) && j[key].is_number()) target = j[key].get<float>();
        };
        const auto getTexture = [&](const char* key, std::shared_ptr<Texture>& target) {
            if (auto texture = lookupTexture(j, key, textures, warnings)) target = texture;
        };

        if (j.contains("threeppTextureAnimatedHint") && j["threeppTextureAnimatedHint"].is_boolean()) {
            material.textureAnimatedHint = j["threeppTextureAnimatedHint"].get<bool>();
        }

        if (auto* m = dynamic_cast<MaterialWithDetailMap*>(&material)) {
            getTexture("threeppDetailMap", m->detailMap);
            getFloat("threeppDetailRepeat", m->detailRepeat);
            getFloat("threeppDetailStrength", m->detailStrength);
            getTexture("threeppDetailNormalMap", m->detailNormalMap);
            getFloat("threeppDetailNormalScale", m->detailNormalScale);
            getFloat("threeppDetailRoughStrength", m->detailRoughStrength);
        }

        // Three.js r129 keys, but with no setValue() entry on the concrete
        // materials — applied directly like the threepp-only fields above.
        if (auto* m = dynamic_cast<MaterialWithMorphTargets*>(&material)) {
            if (j.contains("morphTargets") && j["morphTargets"].is_boolean()) m->morphTargets = j["morphTargets"].get<bool>();
            if (j.contains("morphNormals") && j["morphNormals"].is_boolean()) m->morphNormals = j["morphNormals"].get<bool>();
        }

        if (auto* m = dynamic_cast<MaterialWithTerrainMaps*>(&material)) {
            getTexture("threeppTerrainWeightMap", m->terrainWeightMap);
            getTexture("threeppTerrainNormalMap", m->terrainNormalMap);
            for (int i = 0; i < MaterialWithTerrainMaps::kTerrainBands; i++) {
                const auto suffix = std::to_string(i);
                getTexture(("threeppTerrainBandAlbedo" + suffix).c_str(), m->terrainBandAlbedo[i]);
                getTexture(("threeppTerrainBandNormalRough" + suffix).c_str(), m->terrainBandNormalRough[i]);
            }
            const auto getFloats = [&](const char* key, auto& target) {
                if (!j.contains(key) || !j[key].is_array()) return;
                const auto& arr = j[key];
                for (size_t i = 0; i < target.size() && i < arr.size(); i++) {
                    if (arr[i].is_number()) target[i] = arr[i].get<float>();
                }
            };
            getFloats("threeppTerrainBandRepeat", m->terrainBandRepeat);
            getFloats("threeppTerrainBandRoughness", m->terrainBandRoughness);
            getFloat("threeppTerrainBandStrength", m->terrainBandStrength);
            getFloat("threeppTerrainBandNormalScale", m->terrainBandNormalScale);
            getFloat("threeppTerrainBandRoughStrength", m->terrainBandRoughStrength);
            getFloat("threeppTerrainHeightBlend", m->terrainHeightBlend);
        }
        if (auto* m = dynamic_cast<MaterialWithTranslucency*>(&material)) {
            getFloat("threeppTranslucency", m->translucency);
            if (j.contains("threeppTranslucencyColor") && j["threeppTranslucencyColor"].is_number()) {
                m->translucencyColor.copy(colorFrom(j["threeppTranslucencyColor"]));
            }
        }
    }

    // ---------------------------------------------------------- geometries

    template<class T>
    std::shared_ptr<BufferAttribute> makeAttribute(const json& array, int itemSize, bool normalized) {

        return TypedBufferAttribute<T>::create(array.get<std::vector<T>>(), itemSize, normalized);
    }

    // The `url` form: the numbers are in a binary section instead of the JSON.
    // Additive to the format and to nothing else — the entry still carries its
    // type, itemSize and normalized flag, so what comes back is the attribute
    // the array form would have produced, read with a memcpy rather than a
    // parse. A window that does not check out yields no attribute at all, which
    // the caller reports and skips.
    template<class T>
    std::shared_ptr<BufferAttribute> makeBinaryAttribute(const json& j, int itemSize, bool normalized,
                                                         BufferCache& buffers, Warnings& warnings) {

        auto array = readBinaryArray<T>(j, itemSize, buffers, warnings);
        if (!array) return nullptr;

        return TypedBufferAttribute<T>::create(std::move(*array), itemSize, normalized);
    }

    // three.js typed-array name -> threepp TypedBufferAttribute. Six of the names
    // map exactly onto an AttributeType, so a narrowed attribute round-trips with
    // its stored integers bit-identical and its `normalized` flag intact. The two
    // names with no threepp counterpart are widened, each with a warning.
    std::shared_ptr<BufferAttribute> parseAttribute(const json& j, BufferCache& buffers, Warnings& warnings) {

        const auto type = value<std::string>(j, "type", "Float32Array");
        const int itemSize = value(j, "itemSize", 1);
        const bool normalized = value(j, "normalized", false);

        // Only the six types with an exact threepp counterpart can be read
        // straight out of a section; the two that are widened (Float64Array,
        // Int32Array) have no binary form because nothing writes one.
        if (!j.contains("array") && j.contains("url")) {

            std::shared_ptr<BufferAttribute> binary;

            if (type == "Float32Array") {
                binary = makeBinaryAttribute<float>(j, itemSize, normalized, buffers, warnings);
            } else if (type == "Uint32Array") {
                binary = makeBinaryAttribute<unsigned int>(j, itemSize, normalized, buffers, warnings);
            } else if (type == "Uint16Array") {
                binary = makeBinaryAttribute<std::uint16_t>(j, itemSize, normalized, buffers, warnings);
            } else if (type == "Int16Array") {
                binary = makeBinaryAttribute<std::int16_t>(j, itemSize, normalized, buffers, warnings);
            } else if (type == "Uint8Array" || type == "Uint8ClampedArray") {
                binary = makeBinaryAttribute<std::uint8_t>(j, itemSize, normalized, buffers, warnings);
            } else if (type == "Int8Array") {
                binary = makeBinaryAttribute<std::int8_t>(j, itemSize, normalized, buffers, warnings);
            } else {
                warnings.add("attribute type '" + type + "' has no binary form - skipped");
                return nullptr;
            }

            if (binary && j.contains("usage")) {
                binary->setUsage(static_cast<DrawUsage>(j["usage"].get<int>()));
            }

            return binary;
        }

        if (!j.contains("array")) return nullptr;
        const auto& array = j["array"];

        std::shared_ptr<BufferAttribute> attribute;

        if (type == "Float32Array") {
            attribute = makeAttribute<float>(array, itemSize, normalized);
        } else if (type == "Uint32Array") {
            attribute = makeAttribute<unsigned int>(array, itemSize, normalized);
        } else if (type == "Uint16Array") {
            attribute = makeAttribute<std::uint16_t>(array, itemSize, normalized);
        } else if (type == "Int16Array") {
            attribute = makeAttribute<std::int16_t>(array, itemSize, normalized);
        } else if (type == "Uint8Array" || type == "Uint8ClampedArray") {
            // Uint8ClampedArray differs from Uint8Array only in how JS coerces
            // out-of-range writes; the stored bytes are identical, so this is exact.
            attribute = makeAttribute<std::uint8_t>(array, itemSize, normalized);
        } else if (type == "Int8Array") {
            attribute = makeAttribute<std::int8_t>(array, itemSize, normalized);

        } else if (type == "Float64Array") {
            // No 64-bit attribute type in threepp. Float32 is the least-lossy
            // target: it keeps sign and magnitude, losing only mantissa bits.
            warnings.add("attribute type 'Float64Array' has no threepp counterpart - "
                         "narrowed to Float32Array (double-precision mantissa bits are lost)");
            attribute = makeAttribute<float>(array, itemSize, normalized);

        } else if (type == "Int32Array") {
            // No signed 32-bit attribute type. Two candidate widenings, neither
            // lossless in general, so pick per data: UInt32 keeps all 32 bits but
            // misreads negatives, Float32 keeps the sign but is only exact to 2^24.
            const auto values = array.get<std::vector<std::int64_t>>();
            const bool anyNegative = std::any_of(values.begin(), values.end(),
                                                 [](std::int64_t v) { return v < 0; });

            if (!anyNegative) {
                warnings.add("attribute type 'Int32Array' has no threepp counterpart - "
                             "stored as Uint32Array (bit-exact: no negative values present)");
                std::vector<unsigned int> widened;
                widened.reserve(values.size());
                for (const auto v : values) widened.push_back(static_cast<unsigned int>(v));
                attribute = IntBufferAttribute::create(std::move(widened), itemSize, normalized);
            } else {
                warnings.add("attribute type 'Int32Array' has no threepp counterpart and carries "
                             "negative values - converted to Float32Array (exact for magnitudes "
                             "up to 2^24, beyond which precision is lost)");
                std::vector<float> widened;
                widened.reserve(values.size());
                for (const auto v : values) widened.push_back(static_cast<float>(v));
                attribute = FloatBufferAttribute::create(std::move(widened), itemSize, normalized);
            }

        } else {
            warnings.add("unknown attribute array type '" + type + "' - read as Float32Array");
            attribute = makeAttribute<float>(array, itemSize, normalized);
        }

        if (j.contains("usage")) attribute->setUsage(static_cast<DrawUsage>(j["usage"].get<int>()));

        return attribute;
    }

    // The index in its binary form. Widened to uint32 whatever the section
    // holds, because BufferGeometry's index always is one host-side.
    std::optional<std::vector<unsigned int>> parseBinaryIndex(const json& j, BufferCache& buffers, Warnings& warnings) {

        const auto type = value<std::string>(j, "type", "Uint32Array");

        if (type == "Uint16Array") {

            auto narrow = readBinaryArray<std::uint16_t>(j, 1, buffers, warnings);
            if (!narrow) return std::nullopt;

            return std::vector<unsigned int>(narrow->begin(), narrow->end());
        }

        if (type != "Uint32Array") {

            warnings.add("index type '" + type + "' has no binary form - the geometry comes back unindexed");
            return std::nullopt;
        }

        return readBinaryArray<unsigned int>(j, 1, buffers, warnings);
    }

    std::shared_ptr<BufferGeometry> parseDataGeometry(const json& data, BufferCache& buffers, Warnings& warnings) {

        auto geometry = BufferGeometry::create();

        // BufferGeometry's index is always uint32 host-side, so a three.js
        // Uint16Array index widens here (lossless - indices are non-negative).
        if (data.contains("index")) {

            const auto& index = data["index"];
            if (index.contains("array")) {
                geometry->setIndex(index["array"].get<std::vector<unsigned int>>());
            } else if (index.contains("url")) {
                if (auto indices = parseBinaryIndex(index, buffers, warnings)) {
                    geometry->setIndex(std::move(*indices));
                }
            }
        }

        if (data.contains("attributes")) {
            for (auto it = data["attributes"].begin(); it != data["attributes"].end(); ++it) {
                if (auto attribute = parseAttribute(it.value(), buffers, warnings)) {
                    geometry->setAttribute(it.key(), attribute);
                }
            }
        }

        if (data.contains("morphAttributes")) {
            for (auto it = data["morphAttributes"].begin(); it != data["morphAttributes"].end(); ++it) {
                auto* target = geometry->getOrCreateMorphAttribute(it.key());
                for (const auto& entry : it.value()) {
                    if (auto attribute = parseAttribute(entry, buffers, warnings)) target->push_back(attribute);
                }
            }
            geometry->morphTargetsRelative = value(data, "morphTargetsRelative", false);
        }

        if (data.contains("groups")) {
            for (const auto& g : data["groups"]) {
                geometry->addGroup(value(g, "start", 0), value(g, "count", 0),
                                   value(g, "materialIndex", 0u));
            }
        }

        if (data.contains("boundingSphere")) {
            const auto& bs = data["boundingSphere"];
            Vector3 center;
            if (bs.contains("center") && bs["center"].is_array() && bs["center"].size() >= 3) {
                center.set(bs["center"][0].get<float>(), bs["center"][1].get<float>(), bs["center"][2].get<float>());
            }
            geometry->boundingSphere = Sphere(center, value(bs, "radius", 0.f));
        }

        return geometry;
    }

    std::shared_ptr<BufferGeometry> parseParametricGeometry(const std::string& type, const json& j) {

        if (type == "BoxGeometry") {
            return BoxGeometry::create(BoxGeometry::Params(
                    value(j, "width", 1.f), value(j, "height", 1.f), value(j, "depth", 1.f),
                    value(j, "widthSegments", 1u), value(j, "heightSegments", 1u), value(j, "depthSegments", 1u)));
        }
        if (type == "SphereGeometry") {
            return SphereGeometry::create(SphereGeometry::Params(
                    value(j, "radius", 1.f), value(j, "widthSegments", 32u), value(j, "heightSegments", 16u),
                    value(j, "phiStart", 0.f), value(j, "phiLength", math::TWO_PI),
                    value(j, "thetaStart", 0.f), value(j, "thetaLength", math::PI)));
        }
        if (type == "PlaneGeometry") {
            return PlaneGeometry::create(PlaneGeometry::Params(
                    value(j, "width", 1.f), value(j, "height", 1.f),
                    value(j, "widthSegments", 1u), value(j, "heightSegments", 1u)));
        }
        if (type == "CylinderGeometry") {
            return CylinderGeometry::create(CylinderGeometry::Params(
                    value(j, "radiusTop", 1.f), value(j, "radiusBottom", 1.f), value(j, "height", 1.f),
                    value(j, "radialSegments", 16u), value(j, "heightSegments", 1u), value(j, "openEnded", false),
                    value(j, "thetaStart", 0.f), value(j, "thetaLength", math::TWO_PI)));
        }
        if (type == "ConeGeometry") {
            return ConeGeometry::create(ConeGeometry::Params(
                    value(j, "radius", 1.f), value(j, "height", 1.f),
                    value(j, "radialSegments", 16u), value(j, "heightSegments", 1u), value(j, "openEnded", false),
                    value(j, "thetaStart", 0.f), value(j, "thetaLength", math::TWO_PI)));
        }
        if (type == "CircleGeometry") {
            return CircleGeometry::create(CircleGeometry::Params(
                    value(j, "radius", 1.f), value(j, "segments", 16u),
                    value(j, "thetaStart", 0.f), value(j, "thetaLength", math::TWO_PI)));
        }
        if (type == "RingGeometry") {
            return RingGeometry::create(RingGeometry::Params(
                    value(j, "innerRadius", 0.5f), value(j, "outerRadius", 1.f),
                    value(j, "thetaSegments", 16u), value(j, "phiSegments", 1u),
                    value(j, "thetaStart", 0.f), value(j, "thetaLength", math::TWO_PI)));
        }
        if (type == "TorusGeometry") {
            return TorusGeometry::create(TorusGeometry::Params(
                    value(j, "radius", 1.f), value(j, "tube", 0.4f),
                    value(j, "radialSegments", 20u), value(j, "tubularSegments", 64u),
                    value(j, "arc", math::TWO_PI)));
        }
        if (type == "TorusKnotGeometry") {
            return TorusKnotGeometry::create(TorusKnotGeometry::Params(
                    value(j, "radius", 1.f), value(j, "tube", 0.4f),
                    value(j, "tubularSegments", 64u), value(j, "radialSegments", 16u),
                    value(j, "p", 2u), value(j, "q", 3u)));
        }
        if (type == "CapsuleGeometry") {
            return CapsuleGeometry::create(CapsuleGeometry::Params(
                    value(j, "radius", 0.5f), value(j, "length", 1.f),
                    value(j, "capSegments", 8u), value(j, "radialSegments", 16u)));
        }
        if (type == "IcosahedronGeometry") {
            return IcosahedronGeometry::create(value(j, "radius", 1.f), value(j, "detail", 0u));
        }
        if (type == "OctahedronGeometry") {
            return OctahedronGeometry::create(value(j, "radius", 1.f), value(j, "detail", 0u));
        }
        if (type == "LatheGeometry") {
            std::vector<Vector2> points;
            if (j.contains("points")) {
                for (const auto& p : j["points"]) {
                    if (p.is_array() && p.size() >= 2) {
                        points.emplace_back(p[0].get<float>(), p[1].get<float>());
                    } else if (p.is_object()) {
                        points.emplace_back(value(p, "x", 0.f), value(p, "y", 0.f));
                    }
                }
            }
            return LatheGeometry::create(LatheGeometry::Params(
                    points, value(j, "segments", 12u),
                    value(j, "phiStart", 0.f), value(j, "phiLength", math::TWO_PI)));
        }

        return nullptr;
    }

    // ----------------------------------------------------------- animations

    std::shared_ptr<KeyframeTrack> parseTrack(const json& j, Warnings& warnings) {

        const auto name = value<std::string>(j, "name", "");
        const auto type = value<std::string>(j, "type", "number");
        const auto times = j.contains("times") ? j["times"].get<std::vector<float>>() : std::vector<float>{};
        const auto values = j.contains("values") ? j["values"].get<std::vector<float>>() : std::vector<float>{};

        std::optional<Interpolation> interpolation;
        if (j.contains("interpolation")) interpolation = interpolationFromJson(j["interpolation"].get<int>());

        std::string lower;
        for (char c : type) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

        if (lower == "vector" || lower == "vector2" || lower == "vector3" || lower == "vector4") {
            return std::make_shared<VectorKeyframeTrack>(name, times, values, interpolation);
        }
        if (lower == "quaternion") {
            return std::make_shared<QuaternionKeyframeTrack>(name, times, values, interpolation);
        }
        if (lower == "color") {
            return std::make_shared<ColorKeyframeTrack>(name, times, values, interpolation);
        }
        if (lower == "bool" || lower == "boolean") {
            return std::make_shared<BooleanKeyframeTrack>(name, times, values);
        }
        if (lower == "string") {
            return std::make_shared<StringKeyframeTrack>(name, times, values);
        }
        if (lower == "scalar" || lower == "double" || lower == "float" || lower == "number" || lower == "integer") {
            return std::make_shared<NumberKeyframeTrack>(name, times, values, interpolation);
        }

        warnings.add("unsupported keyframe track type '" + type + "'");
        return nullptr;
    }

}// namespace

namespace threepp::objectjson {

    ImageMap parseImages(const json& j, const DocumentSource& source, Warnings& warnings) {

        ImageMap images;
        if (!j.is_array()) return images;

        for (const auto& entry : j) {
            if (!entry.contains("uuid") || !entry.contains("url")) continue;

            const auto channels = value(entry, "threeppChannels", 4);
            auto decoded = decodeImage(entry, entry["url"], channels, source);

            if (decoded.empty()) {
                warnings.add("could not decode image '" + entry["uuid"].get<std::string>() + "'");
                continue;
            }

            images[entry["uuid"].get<std::string>()] = std::move(decoded);
        }

        return images;
    }

    TextureMap parseTextures(const json& j, const ImageMap& images) {

        TextureMap textures;
        if (!j.is_array()) return textures;

        for (const auto& entry : j) {
            if (!entry.contains("uuid")) continue;

            std::vector<Image> imageData;
            if (entry.contains("image")) {
                const auto it = images.find(entry["image"].get<std::string>());
                if (it != images.end()) imageData = it->second;
            }

            std::shared_ptr<Texture> texture = imageData.size() == 6
                                                       ? std::static_pointer_cast<Texture>(CubeTexture::create(imageData))
                                                       : Texture::create(imageData);

            texture->setUuid(entry["uuid"].get<std::string>());
            texture->name = value<std::string>(entry, "name", "");

            if (entry.contains("mapping")) texture->mapping = static_cast<Mapping>(entry["mapping"].get<int>());

            if (entry.contains("repeat")) texture->repeat = vector2From(entry["repeat"], {1, 1});
            if (entry.contains("offset")) texture->offset = vector2From(entry["offset"]);
            if (entry.contains("center")) texture->center = vector2From(entry["center"]);
            texture->rotation = value(entry, "rotation", 0.f);

            if (entry.contains("wrap") && entry["wrap"].is_array() && entry["wrap"].size() >= 2) {
                texture->wrapS = static_cast<TextureWrapping>(entry["wrap"][0].get<int>());
                texture->wrapT = static_cast<TextureWrapping>(entry["wrap"][1].get<int>());
            }

            if (entry.contains("format")) texture->format = formatFromJson(entry["format"].get<int>());
            if (entry.contains("type")) texture->type = static_cast<Type>(entry["type"].get<int>());

            if (entry.contains("threeppColorSpace")) {
                texture->colorSpace = static_cast<ColorSpace>(entry["threeppColorSpace"].get<int>());
            } else if (entry.contains("encoding")) {
                texture->colorSpace = colorSpaceFromJsonEncoding(entry["encoding"].get<int>());
            }

            if (entry.contains("minFilter")) texture->minFilter = static_cast<Filter>(entry["minFilter"].get<int>());
            if (entry.contains("magFilter")) texture->magFilter = static_cast<Filter>(entry["magFilter"].get<int>());
            texture->anisotropy = value(entry, "anisotropy", 1);

            texture->premultiplyAlpha = value(entry, "premultiplyAlpha", false);
            texture->unpackAlignment = value(entry, "unpackAlignment", 4);
            texture->generateMipmaps = value(entry, "generateMipmaps", true);

            texture->needsUpdate();

            textures[texture->uuid()] = texture;
        }

        return textures;
    }

    MaterialMap parseMaterials(const json& j, const TextureMap& textures, Warnings& warnings) {

        MaterialMap materials;
        if (!j.is_array()) return materials;

        for (const auto& entry : j) {
            if (!entry.contains("uuid")) continue;

            const auto type = value<std::string>(entry, "type", "MeshStandardMaterial");

            auto material = createMaterial(type);
            if (!material) {
                // Mirrors three.js MaterialLoader, which falls back to the
                // default material rather than failing the whole document.
                warnings.add("unsupported material type '" + type + "' - falling back to MeshStandardMaterial");
                material = MeshStandardMaterial::create();
            }

            material->setUuid(entry["uuid"].get<std::string>());
            material->name = value<std::string>(entry, "name", "");

            material->setValues(collectMaterialValues(*material, entry, textures, warnings));
            applyThreeppMaterialExtensions(*material, entry, textures, warnings);

            if (auto* withDefines = dynamic_cast<MaterialWithDefines*>(material.get())) {
                if (entry.contains("defines") && entry["defines"].is_object()) {
                    for (auto it = entry["defines"].begin(); it != entry["defines"].end(); ++it) {
                        if (it.value().is_string()) withDefines->defines[it.key()] = it.value().get<std::string>();
                    }
                }
            }

            materials[material->uuid()] = material;
        }

        return materials;
    }

    GeometryMap parseGeometries(const json& j, BufferCache& buffers, Warnings& warnings) {

        GeometryMap geometries;
        if (!j.is_array()) return geometries;

        for (const auto& entry : j) {
            if (!entry.contains("uuid")) continue;

            const auto type = value<std::string>(entry, "type", "BufferGeometry");

            std::shared_ptr<BufferGeometry> geometry;

            if (type == "BufferGeometry" || type == "InstancedBufferGeometry") {
                if (entry.contains("data")) {
                    geometry = parseDataGeometry(entry["data"], buffers, warnings);
                } else {
                    geometry = BufferGeometry::create();
                }
            } else {
                geometry = parseParametricGeometry(type, entry);
                if (!geometry && entry.contains("data")) {
                    geometry = parseDataGeometry(entry["data"], buffers, warnings);
                }
                if (!geometry) {
                    warnings.add("unsupported geometry type '" + type + "'");
                    continue;
                }
            }

            geometry->uuid = entry["uuid"].get<std::string>();
            geometry->name = value<std::string>(entry, "name", "");

            geometries[geometry->uuid] = geometry;
        }

        return geometries;
    }

    AnimationMap parseAnimations(const json& j, Warnings& warnings) {

        AnimationMap animations;
        if (!j.is_array()) return animations;

        for (const auto& entry : j) {

            std::vector<std::shared_ptr<KeyframeTrack>> tracks;
            if (entry.contains("tracks")) {
                for (const auto& t : entry["tracks"]) {
                    if (auto track = parseTrack(t, warnings)) tracks.push_back(track);
                }
            }

            auto clip = std::make_shared<AnimationClip>(
                    value<std::string>(entry, "name", ""),
                    value(entry, "duration", 1.f),
                    tracks,
                    blendModeFromJson(value(entry, "blendMode", 2500)));

            if (entry.contains("uuid")) clip->setUuid(entry["uuid"].get<std::string>());

            animations[clip->uuid()] = clip;
        }

        return animations;
    }

}// namespace threepp::objectjson
