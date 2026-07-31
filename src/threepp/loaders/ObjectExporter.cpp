// https://github.com/mrdoob/three.js/blob/r129/src/core/Object3D.js (toJSON)
// https://github.com/mrdoob/three.js/blob/r129/src/materials/Material.js (toJSON)
// https://github.com/mrdoob/three.js/blob/r129/src/core/BufferGeometry.js (toJSON)
// https://github.com/mrdoob/three.js/blob/r129/src/textures/Texture.js (toJSON)

#include "threepp/loaders/ObjectExporter.hpp"

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/geometries.hpp"
#include "threepp/geometries/LatheGeometry.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/PolyhedronGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/materials/materials.hpp"
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
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include "ObjectJsonConstants.hpp"
#include "threepp/loaders/AssetSource.hpp"
#include "threepp/utils/Base64.hpp"

#include <nlohmann/json.hpp>

#include "stb_image_write.h"

#include <cstdint>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_set>

// ordered_json keeps object keys in insertion order, so the same scene always
// dumps to a byte-identical document (autosave/VCS diffs stay minimal).
using json = nlohmann::ordered_json;
using namespace threepp;
using namespace threepp::objectjson;

namespace {

    // Collector for the shared, uuid-keyed top-level arrays. Insertion order is
    // scene-traversal order, which makes exports reproducible run to run.
    struct Meta {

        ImageStorage imageStorage{ImageStorage::Embed};
        ModelStorage modelStorage{ModelStorage::Embed};
        std::filesystem::path resourcePath;

        std::vector<json> geometries;
        std::vector<json> materials;
        std::vector<json> textures;
        std::vector<json> images;
        std::vector<json> skeletons;
        std::vector<json> animations;

        std::vector<std::string> warnings;

        std::unordered_set<std::string> seen;

        // Path as it should appear in the document: relative to resourcePath
        // when that is possible, so a scene and its assets can be moved or
        // checked in together. Absolute otherwise — a wrong relative path is
        // worse than an honest absolute one.
        [[nodiscard]] std::string reference(const std::filesystem::path& path) const {

            if (resourcePath.empty()) return path.generic_string();

            std::error_code ec;
            const auto relative = std::filesystem::relative(path, resourcePath, ec);
            if (ec || relative.empty()) return path.generic_string();

            return relative.generic_string();
        }

        bool claim(const std::string& uuid) {

            return seen.insert(uuid).second;
        }

        void warn(std::string message) {

            std::cerr << "[ObjectExporter] " << message << std::endl;
            warnings.push_back(std::move(message));
        }
    };

    // Deterministic iteration over threepp's unordered maps.
    template<class Map>
    std::vector<std::string> sortedKeys(const Map& map) {

        std::vector<std::string> keys;
        keys.reserve(map.size());
        for (const auto& [key, _] : map) keys.push_back(key);
        std::sort(keys.begin(), keys.end());

        return keys;
    }

    // -0.f and 0.f compare equal but serialize differently ("-0.0" vs "0.0").
    // Matrix4::compose() produces negative zeros for ordinary transforms, so an
    // exported document that is loaded and re-exported would otherwise differ
    // from the original on those bytes alone - which defeats the point of
    // deterministic output for autosave diffs.
    constexpr float noNegativeZero(float v) {

        return v == 0.f ? 0.f : v;
    }

    json toArray(const Matrix4& m) {

        json out = json::array();
        for (const auto e : m.elements) out.push_back(noNegativeZero(e));
        return out;
    }

    json toArray(const Vector2& v) {

        return json::array({noNegativeZero(v.x), noNegativeZero(v.y)});
    }

    unsigned int hex(const Color& c) {

        return c.getHex();
    }

    json writeUserData(const std::unordered_map<std::string, std::any>& userData, Meta& meta) {

        json out = json::object();

        for (const auto& key : sortedKeys(userData)) {

            const auto& value = userData.at(key);

            if (value.type() == typeid(bool)) {
                out[key] = std::any_cast<bool>(value);
            } else if (value.type() == typeid(int)) {
                out[key] = std::any_cast<int>(value);
            } else if (value.type() == typeid(unsigned int)) {
                out[key] = std::any_cast<unsigned int>(value);
            } else if (value.type() == typeid(std::int64_t)) {
                out[key] = std::any_cast<std::int64_t>(value);
            } else if (value.type() == typeid(float)) {
                out[key] = std::any_cast<float>(value);
            } else if (value.type() == typeid(double)) {
                out[key] = std::any_cast<double>(value);
            } else if (value.type() == typeid(std::string)) {
                out[key] = std::any_cast<std::string>(value);
            } else if (value.type() == typeid(const char*)) {
                out[key] = std::string(std::any_cast<const char*>(value));
            } else {
                meta.warn("skipping userData entry '" + key + "': unsupported type " + value.type().name());
            }
        }

        return out;
    }

    // ------------------------------------------------------------- images

    void pngWriter(void* context, void* data, int size) {

        auto* out = static_cast<std::vector<unsigned char>*>(context);
        const auto* bytes = static_cast<unsigned char*>(data);
        out->insert(out->end(), bytes, bytes + size);
    }

    std::optional<std::string> encodePngDataUrl(const Image& image) {

        const int channels = image.channels();
        if (channels < 1 || channels > 4) return std::nullopt;
        if (image.width() == 0 || image.height() == 0) return std::nullopt;

        std::vector<unsigned char> pixels;
        if (image.isFloat()) {
            const auto& src = image.data<float>();
            pixels.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i) {
                pixels[i] = static_cast<unsigned char>(std::lround(std::clamp(src[i], 0.f, 1.f) * 255.f));
            }
        } else {
            pixels = image.data<unsigned char>();
        }

        std::vector<unsigned char> png;
        const int stride = static_cast<int>(image.width()) * channels;
        if (!stbi_write_png_to_func(pngWriter, &png,
                                    static_cast<int>(image.width()),
                                    static_cast<int>(image.height()),
                                    channels, pixels.data(), stride)) {
            return std::nullopt;
        }

        return "data:image/png;base64," + utils::base64Encode(png);
    }

    // ------------------------------------------------------------ textures

    std::string writeTexture(Texture& texture, Meta& meta);

    // A texture can be written as a path only when its pixels came from one
    // file. A cube map is six faces behind a single `sourceFile`, so there is
    // nothing to point the other five at; it embeds like it always did.
    bool canReference(const Texture& texture) {

        return !texture.sourceFile.empty() && texture.images().size() != 6;
    }

    // Emits the `images` entry backing `texture` and returns its uuid, or an
    // empty string when there is no CPU-side pixel data to embed.
    std::string writeImage(const Texture& texture, Meta& meta) {

        const auto& images = texture.images();
        if (images.empty()) return {};

        // Deterministic and stable across a single export: the image entry is
        // keyed off the owning texture, which is what three.js effectively does
        // too (it stamps a uuid onto the HTMLImage the first time it sees it).
        const std::string imageUuid = texture.uuid() + "-image";

        if (!meta.claim(imageUuid)) return imageUuid;

        json entry;
        entry["uuid"] = imageUuid;

        const bool isCube = images.size() == 6;

        // A path reference stands in for the whole entry — the pixels are
        // already on disk, in a better format than a re-encoded PNG.
        if (meta.imageStorage == ImageStorage::Reference && canReference(texture)) {
            entry["url"] = meta.reference(texture.sourceFile);
            entry["threeppChannels"] = images.front().channels();
            meta.images.push_back(entry);
            return imageUuid;
        }

        if (isCube) {
            json urls = json::array();
            for (const auto& image : images) {
                auto url = encodePngDataUrl(image);
                if (!url) return {};
                urls.push_back(*url);
            }
            entry["url"] = urls;
            entry["threeppChannels"] = images.front().channels();
        } else {
            auto url = encodePngDataUrl(images.front());
            if (!url) return {};
            entry["url"] = *url;
            entry["threeppChannels"] = images.front().channels();
        }

        meta.images.push_back(entry);

        return imageUuid;
    }

    std::string writeTexture(Texture& texture, Meta& meta) {

        if (!meta.claim(texture.uuid())) return texture.uuid();

        json entry;
        entry["uuid"] = texture.uuid();
        entry["name"] = texture.name;

        entry["mapping"] = as_integer(texture.mapping);

        entry["repeat"] = toArray(texture.repeat);
        entry["offset"] = toArray(texture.offset);
        entry["center"] = toArray(texture.center);
        entry["rotation"] = texture.rotation;

        entry["wrap"] = json::array({as_integer(texture.wrapS), as_integer(texture.wrapT)});

        entry["format"] = formatToJson(texture.format);
        entry["type"] = as_integer(texture.type);
        entry["encoding"] = colorSpaceToJsonEncoding(texture.colorSpace);
        // threepp's NoColorSpace has no three.js counterpart; keep the exact
        // value alongside so a threepp -> threepp round-trip is lossless.
        entry["threeppColorSpace"] = as_integer(texture.colorSpace);

        entry["minFilter"] = as_integer(texture.minFilter);
        entry["magFilter"] = as_integer(texture.magFilter);
        entry["anisotropy"] = texture.anisotropy;

        // threepp has no flipY on Texture; loaders decide row order up-front.
        entry["flipY"] = false;

        entry["premultiplyAlpha"] = texture.premultiplyAlpha;
        entry["unpackAlignment"] = texture.unpackAlignment;

        entry["generateMipmaps"] = texture.generateMipmaps;

        if (meta.imageStorage != ImageStorage::Omit) {

            // Reference mode is best-effort per texture: one that never came
            // from a file (procedural, or unpacked from inside a .glb) has
            // nothing to point at, so it embeds instead. Say so, because the
            // document is then not as small - or as portable - as asked for.
            if (meta.imageStorage == ImageStorage::Reference && !canReference(texture)) {
                meta.warn("texture '" + (texture.name.empty() ? texture.uuid() : texture.name) +
                          "' cannot be referenced - embedded instead");
            }

            const auto imageUuid = writeImage(texture, meta);
            if (imageUuid.empty()) {
                meta.warn("texture '" + texture.uuid() +
                          "' has no CPU-side image data - written without an image reference");
            } else {
                entry["image"] = imageUuid;
            }
        }

        meta.textures.push_back(entry);

        return texture.uuid();
    }

    void writeTextureSlot(json& data, const char* key, const std::shared_ptr<Texture>& texture, Meta& meta) {

        if (!texture) return;

        data[key] = writeTexture(*texture, meta);
    }

    // ----------------------------------------------------------- materials

    std::string writeMaterial(Material& material, Meta& meta) {

        if (!meta.claim(material.uuid())) return material.uuid();

        json data;
        data["uuid"] = material.uuid();
        data["type"] = material.type();
        if (!material.name.empty()) data["name"] = material.name;

        if (auto* m = dynamic_cast<MaterialWithColor*>(&material)) {
            data["color"] = hex(m->color);
        }
        if (auto* m = dynamic_cast<MaterialWithRoughness*>(&material)) {
            data["roughness"] = m->roughness;
            writeTextureSlot(data, "roughnessMap", m->roughnessMap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithMetalness*>(&material)) {
            data["metalness"] = m->metalness;
            writeTextureSlot(data, "metalnessMap", m->metalnessMap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithSheen*>(&material)) {
            // r129 wrote a single `sheen` colour. threepp only carries the split
            // sheenColor/sheenRoughness pair of later revisions, so that key is
            // neither written here nor read back by ObjectLoader.
            data["sheenColor"] = hex(m->sheenColor);
            data["sheenRoughness"] = m->sheenRoughness;
        }
        if (auto* m = dynamic_cast<MaterialWithEmissive*>(&material)) {
            data["emissive"] = hex(m->emissive);
            data["emissiveIntensity"] = m->emissiveIntensity;
            writeTextureSlot(data, "emissiveMap", m->emissiveMap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithSpecular*>(&material)) {
            data["specular"] = hex(m->specular);
            data["shininess"] = m->shininess;
        }
        if (auto* m = dynamic_cast<MaterialWithClearcoat*>(&material)) {
            data["clearcoat"] = m->clearcoat;
            data["clearcoatRoughness"] = m->clearcoatRoughness;
            writeTextureSlot(data, "clearcoatMap", m->clearcoatMap, meta);
            writeTextureSlot(data, "clearcoatRoughnessMap", m->clearcoatRoughnessMap, meta);
            if (m->clearcoatNormalMap) {
                writeTextureSlot(data, "clearcoatNormalMap", m->clearcoatNormalMap, meta);
                data["clearcoatNormalScale"] = toArray(m->clearcoatNormalScale);
            }
        }
        if (auto* m = dynamic_cast<MaterialWithTransmission*>(&material)) {
            data["transmission"] = m->transmission;
            data["ior"] = m->ior;
            data["dispersion"] = m->dispersion;
            writeTextureSlot(data, "transmissionMap", m->transmissionMap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithThickness*>(&material)) {
            data["thickness"] = m->thickness;
            data["thinWalled"] = m->thinWalled;
            writeTextureSlot(data, "thicknessMap", m->thicknessMap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithAttenuation*>(&material)) {
            data["attenuationDistance"] = m->attenuationDistance;
            data["attenuationColor"] = hex(m->attenuationColor);
        }
        if (auto* m = dynamic_cast<MaterialWithIridescence*>(&material)) {
            data["iridescence"] = m->iridescence;
            data["iridescenceIOR"] = m->iridescenceIOR;
            data["iridescenceThicknessNm"] = m->iridescenceThicknessNm;
        }
        if (auto* m = dynamic_cast<MaterialWithPbrSpecular*>(&material)) {
            data["specularIntensity"] = m->specularIntensity;
            data["specularColor"] = hex(m->specularColor);
        }
        if (auto* m = dynamic_cast<MaterialWithMap*>(&material)) {
            writeTextureSlot(data, "map", m->map, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithMatCap*>(&material)) {
            writeTextureSlot(data, "matcap", m->matcap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithAlphaMap*>(&material)) {
            writeTextureSlot(data, "alphaMap", m->alphaMap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithLightMap*>(&material)) {
            writeTextureSlot(data, "lightMap", m->lightMap, meta);
            data["lightMapIntensity"] = m->lightMapIntensity;
        }
        if (auto* m = dynamic_cast<MaterialWithAoMap*>(&material)) {
            writeTextureSlot(data, "aoMap", m->aoMap, meta);
            data["aoMapIntensity"] = m->aoMapIntensity;
        }
        if (auto* m = dynamic_cast<MaterialWithBumpMap*>(&material)) {
            writeTextureSlot(data, "bumpMap", m->bumpMap, meta);
            data["bumpScale"] = m->bumpScale;
        }
        if (auto* m = dynamic_cast<MaterialWithNormalMap*>(&material)) {
            writeTextureSlot(data, "normalMap", m->normalMap, meta);
            data["normalMapType"] = as_integer(m->normalMapType);
            data["normalScale"] = toArray(m->normalScale);
        }
        if (auto* m = dynamic_cast<MaterialWithDisplacementMap*>(&material)) {
            writeTextureSlot(data, "displacementMap", m->displacementMap, meta);
            data["displacementScale"] = m->displacementScale;
            data["displacementBias"] = m->displacementBias;
        }
        // threepp extensions with no three.js counterpart. Namespaced so a
        // three.js reader ignores them and a threepp round-trip stays lossless.
        if (auto* m = dynamic_cast<MaterialWithDetailMap*>(&material)) {
            writeTextureSlot(data, "threeppDetailMap", m->detailMap, meta);
            data["threeppDetailRepeat"] = m->detailRepeat;
            data["threeppDetailStrength"] = m->detailStrength;
            writeTextureSlot(data, "threeppDetailNormalMap", m->detailNormalMap, meta);
            data["threeppDetailNormalScale"] = m->detailNormalScale;
            data["threeppDetailRoughStrength"] = m->detailRoughStrength;
        }
        if (auto* m = dynamic_cast<MaterialWithTranslucency*>(&material)) {
            data["threeppTranslucency"] = m->translucency;
            data["threeppTranslucencyColor"] = hex(m->translucencyColor);
        }
        if (auto* m = dynamic_cast<MaterialWithSpecularMap*>(&material)) {
            writeTextureSlot(data, "specularMap", m->specularMap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithEnvMap*>(&material)) {
            writeTextureSlot(data, "envMap", m->envMap, meta);
            data["envMapIntensity"] = m->envMapIntensity;
        }
        if (auto* m = dynamic_cast<MaterialWithReflectivity*>(&material)) {
            data["reflectivity"] = m->reflectivity;
        }
        if (auto* m = dynamic_cast<MaterialWithRefractionRatio*>(&material)) {
            data["refractionRatio"] = m->refractionRatio;
        }
        if (auto* m = dynamic_cast<MaterialWithCombine*>(&material)) {
            data["combine"] = as_integer(m->combine);
        }
        if (auto* m = dynamic_cast<MaterialWithGradientMap*>(&material)) {
            writeTextureSlot(data, "gradientMap", m->gradientMap, meta);
        }
        if (auto* m = dynamic_cast<MaterialWithSize*>(&material)) {
            data["size"] = m->size;
            data["sizeAttenuation"] = m->sizeAttenuation;
        }
        if (auto* m = dynamic_cast<MaterialWithLineWidth*>(&material)) {
            data["linewidth"] = m->linewidth;
        }
        if (auto* m = dynamic_cast<LineDashedMaterial*>(&material)) {
            data["dashSize"] = m->dashSize;
            data["gapSize"] = m->gapSize;
            data["scale"] = m->scale;
        }
        if (auto* m = dynamic_cast<MaterialWithRotation*>(&material)) {
            data["rotation"] = m->rotation;
        }
        if (auto* m = dynamic_cast<MaterialWithWireframe*>(&material)) {
            data["wireframe"] = m->wireframe;
            data["wireframeLinewidth"] = m->wireframeLinewidth;
        }
        if (auto* m = dynamic_cast<MaterialWithFlatShading*>(&material)) {
            data["flatShading"] = m->flatShading;
        }
        if (auto* m = dynamic_cast<MaterialWithVertexTangents*>(&material)) {
            data["vertexTangents"] = m->vertexTangents;
        }
        if (auto* m = dynamic_cast<MaterialWithDepthPacking*>(&material)) {
            data["depthPacking"] = as_integer(m->depthPacking);
        }
        if (auto* m = dynamic_cast<MaterialWithClipping*>(&material)) {
            data["clipping"] = m->clipping;
        }
        if (auto* m = dynamic_cast<MaterialWithLights*>(&material)) {
            data["lights"] = m->lights;
        }
        if (auto* m = dynamic_cast<MaterialWithDefines*>(&material)) {
            if (!m->defines.empty()) {
                json defines = json::object();
                for (const auto& k : sortedKeys(m->defines)) defines[k] = m->defines.at(k);
                data["defines"] = defines;
            }
        }
        if (auto* m = dynamic_cast<ShaderMaterial*>(&material)) {
            data["vertexShader"] = m->vertexShader;
            data["fragmentShader"] = m->fragmentShader;
            if (!m->uniforms.empty()) {
                // Uniform values are a deep, backend-flavoured variant; three.js
                // serializes them but threepp defers (see ObjectLoader docs).
                meta.warn("ShaderMaterial '" + material.uuid() + "': uniforms are not serialized");
            }
        }

        // base Material state
        data["blending"] = as_integer(material.blending);
        data["side"] = as_integer(material.side);
        data["vertexColors"] = material.vertexColors;
        data["opacity"] = material.opacity;
        data["transparent"] = material.transparent;
        data["fog"] = material.fog;

        data["blendSrc"] = as_integer(material.blendSrc);
        data["blendDst"] = as_integer(material.blendDst);
        data["blendEquation"] = as_integer(material.blendEquation);
        if (material.blendSrcAlpha) data["blendSrcAlpha"] = as_integer(*material.blendSrcAlpha);
        if (material.blendDstAlpha) data["blendDstAlpha"] = as_integer(*material.blendDstAlpha);
        if (material.blendEquationAlpha) data["blendEquationAlpha"] = as_integer(*material.blendEquationAlpha);

        data["depthFunc"] = as_integer(material.depthFunc);
        data["depthTest"] = material.depthTest;
        data["depthWrite"] = material.depthWrite;
        data["colorWrite"] = material.colorWrite;

        data["stencilWrite"] = material.stencilWrite;
        data["stencilWriteMask"] = material.stencilWriteMask;
        data["stencilFunc"] = as_integer(material.stencilFunc);
        data["stencilRef"] = material.stencilRef;
        data["stencilFuncMask"] = material.stencilFuncMask;
        data["stencilFail"] = as_integer(material.stencilFail);
        data["stencilZFail"] = as_integer(material.stencilZFail);
        data["stencilZPass"] = as_integer(material.stencilZPass);

        if (material.shadowSide) data["shadowSide"] = as_integer(*material.shadowSide);

        data["polygonOffset"] = material.polygonOffset;
        data["polygonOffsetFactor"] = material.polygonOffsetFactor;
        data["polygonOffsetUnits"] = material.polygonOffsetUnits;

        data["dithering"] = material.dithering;
        data["alphaTest"] = material.alphaTest;
        data["alphaToCoverage"] = material.alphaToCoverage;
        data["premultipliedAlpha"] = material.premultipliedAlpha;

        data["visible"] = material.visible;
        data["toneMapped"] = material.toneMapped;

        // threepp extension: a Vulkan temporal-pass hint, no three.js counterpart.
        // Only written when set, so the common case costs no bytes.
        if (material.textureAnimatedHint) data["threeppTextureAnimatedHint"] = true;

        meta.materials.push_back(data);

        return material.uuid();
    }

    // ---------------------------------------------------------- geometries

    // Writes the backing store verbatim. array() returns the STORED scalars, not
    // the denormalized ones, which is exactly what three.js expects next to a
    // `normalized` flag - so nothing is decoded on the way out.
    template<class T>
    bool storeAttributeArray(const BufferAttribute& attribute, json& data) {

        const auto* typed = dynamic_cast<const TypedBufferAttribute<T>*>(&attribute);
        if (!typed) return false;

        data["array"] = typed->array();
        return true;
    }

    json writeAttribute(const BufferAttribute& attribute) {

        json data;
        data["itemSize"] = attribute.itemSize();
        // Load-bearing since compressAttributes() landed: for the narrow integer
        // types this flag is what declares the [0,1]/[-1,1] mapping. Dropping it
        // turns a normal or a UV attribute into garbage on reload.
        data["normalized"] = attribute.normalized();
        data["type"] = attributeTypeToArrayName(attribute.type());

        bool stored = false;
        switch (attribute.type()) {
            case AttributeType::Float: stored = storeAttributeArray<float>(attribute, data); break;
            case AttributeType::UInt32: stored = storeAttributeArray<unsigned int>(attribute, data); break;
            case AttributeType::UInt16: stored = storeAttributeArray<std::uint16_t>(attribute, data); break;
            case AttributeType::Int16: stored = storeAttributeArray<std::int16_t>(attribute, data); break;
            case AttributeType::UInt8: stored = storeAttributeArray<std::uint8_t>(attribute, data); break;
            case AttributeType::Int8: stored = storeAttributeArray<std::int8_t>(attribute, data); break;
        }

        if (!stored) return json();

        if (attribute.getUsage() != DrawUsage::Static) {
            data["usage"] = as_integer(attribute.getUsage());
        }

        return data;
    }

    // Returns true when `geometry` was written in three.js' compact parametric
    // form, false when the caller should fall back to the universal data form.
    bool writeParametricGeometry(const BufferGeometry& geometry, json& data) {

        if (const auto* g = dynamic_cast<const BoxGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["width"] = p.width;
            data["height"] = p.height;
            data["depth"] = p.depth;
            data["widthSegments"] = p.widthSegments;
            data["heightSegments"] = p.heightSegments;
            data["depthSegments"] = p.depthSegments;
            return true;
        }
        if (const auto* g = dynamic_cast<const SphereGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["radius"] = p.radius;
            data["widthSegments"] = p.widthSegments;
            data["heightSegments"] = p.heightSegments;
            data["phiStart"] = p.phiStart;
            data["phiLength"] = p.phiLength;
            data["thetaStart"] = p.thetaStart;
            data["thetaLength"] = p.thetaLength;
            return true;
        }
        if (const auto* g = dynamic_cast<const PlaneGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["width"] = p.width;
            data["height"] = p.height;
            data["widthSegments"] = p.widthSegments;
            data["heightSegments"] = p.heightSegments;
            return true;
        }
        if (const auto* g = dynamic_cast<const ConeGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["radius"] = p.radius;
            data["height"] = p.height;
            data["radialSegments"] = p.radialSegments;
            data["heightSegments"] = p.heightSegments;
            data["openEnded"] = p.openEnded;
            data["thetaStart"] = p.thetaStart;
            data["thetaLength"] = p.thetaLength;
            return true;
        }
        if (const auto* g = dynamic_cast<const CylinderGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["radiusTop"] = p.radiusTop;
            data["radiusBottom"] = p.radiusBottom;
            data["height"] = p.height;
            data["radialSegments"] = p.radialSegments;
            data["heightSegments"] = p.heightSegments;
            data["openEnded"] = p.openEnded;
            data["thetaStart"] = p.thetaStart;
            data["thetaLength"] = p.thetaLength;
            return true;
        }
        if (const auto* g = dynamic_cast<const CircleGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["radius"] = p.radius;
            data["segments"] = p.segments;
            data["thetaStart"] = p.thetaStart;
            data["thetaLength"] = p.thetaLength;
            return true;
        }
        if (const auto* g = dynamic_cast<const RingGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["innerRadius"] = p.innerRadius;
            data["outerRadius"] = p.outerRadius;
            data["thetaSegments"] = p.thetaSegments;
            data["phiSegments"] = p.phiSegments;
            data["thetaStart"] = p.thetaStart;
            data["thetaLength"] = p.thetaLength;
            return true;
        }
        if (const auto* g = dynamic_cast<const TorusKnotGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["radius"] = p.radius;
            data["tube"] = p.tube;
            data["tubularSegments"] = p.tubularSegments;
            data["radialSegments"] = p.radialSegments;
            data["p"] = p.p;
            data["q"] = p.q;
            return true;
        }
        if (const auto* g = dynamic_cast<const TorusGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["radius"] = p.radius;
            data["tube"] = p.tube;
            data["radialSegments"] = p.radialSegments;
            data["tubularSegments"] = p.tubularSegments;
            data["arc"] = p.arc;
            return true;
        }
        if (const auto* g = dynamic_cast<const CapsuleGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            data["radius"] = p.radius;
            data["length"] = p.length;
            data["capSegments"] = p.capSegments;
            data["radialSegments"] = p.radialSegments;
            return true;
        }
        if (const auto* g = dynamic_cast<const PolyhedronGeometry*>(&geometry)) {
            // Icosahedron/Octahedron: three.js serializes radius + detail.
            data["radius"] = g->radius;
            data["detail"] = g->detail;
            return true;
        }
        if (const auto* g = dynamic_cast<const LatheGeometry*>(&geometry)) {
            const auto& p = g->parameters;
            json points = json::array();
            for (const auto& pt : p.points) points.push_back(json{{"x", pt.x}, {"y", pt.y}});
            data["points"] = points;
            data["segments"] = p.segments;
            data["phiStart"] = p.phiStart;
            data["phiLength"] = p.phiLength;
            return true;
        }

        return false;
    }

    std::string writeGeometry(BufferGeometry& geometry, Meta& meta) {

        if (!meta.claim(geometry.uuid)) return geometry.uuid;

        json data;
        data["uuid"] = geometry.uuid;
        data["type"] = geometry.type();
        if (!geometry.name.empty()) data["name"] = geometry.name;

        if (writeParametricGeometry(geometry, data)) {

            meta.geometries.push_back(data);
            return geometry.uuid;
        }

        // Universal data form. `type` stays whatever the class reports; a
        // three.js reader falls back to BufferGeometry for unknown types.
        data["type"] = "BufferGeometry";
        if (geometry.type() != "BufferGeometry") data["threeppType"] = geometry.type();

        json inner;
        inner["attributes"] = json::object();

        // The host-side index is always uint32 (BufferGeometry::index_ is an
        // IntBufferAttribute); the uint16 index buffers added in dce8acbb are a
        // Vulkan device-side packing that never reaches this layer. Derived from
        // type() regardless, so this stays correct if that ever changes.
        if (const auto* index = geometry.getIndex()) {
            json idx;
            idx["type"] = attributeTypeToArrayName(index->type());
            idx["array"] = index->array();
            inner["index"] = idx;
        }

        for (const auto& name : sortedKeys(geometry.getAttributes())) {
            auto entry = writeAttribute(*geometry.getAttributes().at(name));
            if (entry.is_null()) {
                meta.warn("geometry '" + geometry.uuid + "': skipping attribute '" + name +
                          "' (unsupported array type)");
                continue;
            }
            inner["attributes"][name] = entry;
        }

        if (!geometry.getMorphAttributes().empty()) {
            json morph = json::object();
            for (const auto& name : sortedKeys(geometry.getMorphAttributes())) {
                json arr = json::array();
                for (const auto& attribute : geometry.getMorphAttributes().at(name)) {
                    auto entry = writeAttribute(*attribute);
                    if (!entry.is_null()) arr.push_back(entry);
                }
                morph[name] = arr;
            }
            inner["morphAttributes"] = morph;
            inner["morphTargetsRelative"] = geometry.morphTargetsRelative;
        }

        if (!geometry.groups.empty()) {
            json groups = json::array();
            for (const auto& g : geometry.groups) {
                groups.push_back(json{{"start", g.start}, {"count", g.count}, {"materialIndex", g.materialIndex}});
            }
            inner["groups"] = groups;
        }

        if (geometry.boundingSphere) {
            inner["boundingSphere"] = json{
                    {"center", json::array({geometry.boundingSphere->center.x,
                                            geometry.boundingSphere->center.y,
                                            geometry.boundingSphere->center.z})},
                    {"radius", geometry.boundingSphere->radius}};
        }

        data["data"] = inner;

        meta.geometries.push_back(data);

        return geometry.uuid;
    }

    // ----------------------------------------------------------- animations

    json writeTrack(const KeyframeTrack& track) {

        json data;
        data["name"] = track.getName();
        data["times"] = track.getTimes();
        data["values"] = track.getValues();
        data["interpolation"] = interpolationToJson(track.getInterpolation());
        data["type"] = track.ValueTypeName();

        return data;
    }

    std::string writeAnimation(const AnimationClip& clip, Meta& meta) {

        if (!meta.claim(clip.uuid())) return clip.uuid();

        json data;
        data["uuid"] = clip.uuid();
        data["name"] = clip.name();
        data["duration"] = clip.getDuration();
        data["blendMode"] = blendModeToJson(clip.blendMode);

        json tracks = json::array();
        for (const auto& track : clip.getTracks()) {
            if (track) tracks.push_back(writeTrack(*track));
        }
        data["tracks"] = tracks;

        meta.animations.push_back(data);

        return clip.uuid();
    }

    // ------------------------------------------------------------ skeletons

    std::string writeSkeleton(Skeleton& skeleton, Meta& meta) {

        if (!meta.claim(skeleton.uuid())) return skeleton.uuid();

        json data;
        data["uuid"] = skeleton.uuid();

        json bones = json::array();
        for (const auto& bone : skeleton.bones) {
            if (bone) bones.push_back(bone->uuid);
        }
        data["bones"] = bones;

        json inverses = json::array();
        for (const auto& m : skeleton.boneInverses) {
            inverses.push_back(toArray(m));
        }
        data["boneInverses"] = inverses;

        meta.skeletons.push_back(data);

        return skeleton.uuid();
    }

    // -------------------------------------------------- linked asset subtrees

    // Pre-order walk of everything below `root`, matching the order ObjectLoader
    // will see when it re-imports the asset and walks the result the same way.
    // The root itself is excluded: its own fields are written as the ordinary
    // object entry and transplanted onto the re-import.
    void forEachDescendant(Object3D& root, const std::function<void(Object3D&)>& visit) {

        for (auto* child : root.children) {
            if (!child) continue;
            visit(*child);
            forEachDescendant(*child, visit);
        }
    }

    // The per-node edits that survive a re-import. Geometry, materials and
    // textures are deliberately not here — those come back from the asset file,
    // which is the whole point of referencing it. Everything a scene editor
    // routinely changes about an imported node without touching its mesh is.
    json writeAssetOverrides(Object3D& root, Meta& meta) {

        json nodes = json::array();

        int index = 0;
        forEachDescendant(root, [&](Object3D& node) {
            const int i = index++;

            json entry;
            // Written unconditionally: it is what tells a reader that the asset
            // still has the shape the document was saved against.
            entry["i"] = i;
            entry["name"] = node.name;

            if (node.matrixAutoUpdate) node.updateMatrix();
            entry["matrix"] = toArray(*node.matrix);

            if (!node.visible) entry["visible"] = false;
            if (node.castShadow) entry["castShadow"] = true;
            if (node.receiveShadow) entry["receiveShadow"] = true;
            if (!node.frustumCulled) entry["frustumCulled"] = false;
            if (node.renderOrder != 0) entry["renderOrder"] = node.renderOrder;
            if (node.layers.mask() != 1) entry["layers"] = node.layers.mask();
            if (!node.userData.empty()) entry["userData"] = writeUserData(node.userData, meta);

            nodes.push_back(entry);
        });

        return nodes;
    }

    // -------------------------------------------------------------- objects

    json writeObject(Object3D& object, Meta& meta, bool includeMatrix = true);

    json writeShadow(LightShadow& shadow, Meta& meta) {

        json data;
        data["bias"] = shadow.bias;
        data["normalBias"] = shadow.normalBias;
        data["radius"] = shadow.radius;
        data["mapSize"] = toArray(shadow.mapSize);

        if (shadow.camera) {
            data["camera"] = writeObject(*shadow.camera, meta, false);
        }

        return data;
    }

    json writeObject(Object3D& object, Meta& meta, bool includeMatrix) {

        json data;

        // A linked subtree writes a path and a table of edits where its
        // geometry, materials and children would otherwise go.
        const auto asset = meta.modelStorage == ModelStorage::Reference
                                   ? assetSource(object)
                                   : std::filesystem::path{};
        const bool linked = !asset.empty();

        data["uuid"] = object.uuid;
        data["type"] = object.type();
        if (!object.name.empty()) data["name"] = object.name;

        if (object.castShadow) data["castShadow"] = true;
        if (object.receiveShadow) data["receiveShadow"] = true;
        if (!object.visible) data["visible"] = false;
        if (!object.frustumCulled) data["frustumCulled"] = false;
        if (object.renderOrder != 0) data["renderOrder"] = object.renderOrder;

        if (!object.userData.empty()) data["userData"] = writeUserData(object.userData, meta);

        data["layers"] = object.layers.mask();

        if (includeMatrix) {
            if (object.matrixAutoUpdate) object.updateMatrix();
            data["matrix"] = toArray(*object.matrix);
            if (!object.matrixAutoUpdate) data["matrixAutoUpdate"] = false;
        }

        // ---- linked asset
        // Everything past this point describes what the object IS, and for a
        // linked subtree that is the asset file's business - geometry,
        // materials, skeletons, clips and children all come back from the
        // re-import. Only the edits made on top of it are ours to record.
        if (linked) {

            // The mark itself stays out of userData here: threeppAsset.path is
            // the authoritative copy, and it is relative. Leaving the absolute
            // sourceFile string in userData would pin this machine's directory
            // layout into an otherwise portable document — every other machine
            // would re-save it with a spurious diff. ObjectLoader re-stamps the
            // mark from the resolved path on load.
            if (data.contains("userData")) {
                data["userData"].erase(assetSourceKey);
                if (data["userData"].empty()) data.erase("userData");
            }

            json ref;
            ref["path"] = meta.reference(asset);
            ref["nodes"] = writeAssetOverrides(object, meta);
            data["threeppAsset"] = ref;

            return data;
        }

        // ---- cameras
        if (auto* camera = dynamic_cast<PerspectiveCamera*>(&object)) {
            data["fov"] = camera->fov;
            data["zoom"] = camera->zoom;
            data["near"] = camera->nearPlane;
            data["far"] = camera->farPlane;
            data["focus"] = camera->focus;
            data["aspect"] = camera->aspect;
            data["filmGauge"] = camera->filmGauge;
            data["filmOffset"] = camera->filmOffset;
        } else if (auto* ortho = dynamic_cast<OrthographicCamera*>(&object)) {
            data["zoom"] = ortho->zoom;
            data["left"] = ortho->left;
            data["right"] = ortho->right;
            data["top"] = ortho->top;
            data["bottom"] = ortho->bottom;
            data["near"] = ortho->nearPlane;
            data["far"] = ortho->farPlane;
        }

        // ---- lights
        if (auto* light = dynamic_cast<Light*>(&object)) {
            data["color"] = hex(light->color);
            data["intensity"] = light->intensity;

            if (auto* hemi = dynamic_cast<HemisphereLight*>(light)) {
                data["groundColor"] = hex(hemi->groundColor);
            }
            if (auto* point = dynamic_cast<PointLight*>(light)) {
                data["distance"] = point->distance;
                data["decay"] = point->decay;
            }
            if (auto* spot = dynamic_cast<SpotLight*>(light)) {
                data["distance"] = spot->distance;
                data["angle"] = spot->angle;
                data["penumbra"] = spot->penumbra;
                data["decay"] = spot->decay;
            }
            if (auto* rect = dynamic_cast<RectAreaLight*>(light)) {
                data["width"] = rect->width;
                data["height"] = rect->height;
            }
            if (auto* withShadow = dynamic_cast<LightWithShadow*>(light)) {
                if (withShadow->shadow) data["shadow"] = writeShadow(*withShadow->shadow, meta);
            }
        }

        // ---- scene
        if (auto* scene = dynamic_cast<Scene*>(&object)) {
            if (!scene->background.empty()) {
                if (scene->background.isColor()) {
                    data["background"] = hex(scene->background.color());
                } else if (auto tex = scene->background.texture()) {
                    data["background"] = writeTexture(*tex, meta);
                }
            }
            if (scene->environment) {
                data["environment"] = writeTexture(*scene->environment, meta);
            }
            if (scene->fog) {
                if (std::holds_alternative<Fog>(*scene->fog)) {
                    const auto& fog = std::get<Fog>(*scene->fog);
                    data["fog"] = json{{"type", "Fog"}, {"color", hex(fog.color)}, {"near", fog.nearPlane}, {"far", fog.farPlane}};
                } else {
                    const auto& fog = std::get<FogExp2>(*scene->fog);
                    data["fog"] = json{{"type", "FogExp2"}, {"color", hex(fog.color)}, {"density", fog.density}};
                }
            }
        }

        // ---- instancing
        if (auto* instanced = dynamic_cast<InstancedMesh*>(&object)) {
            data["count"] = instanced->count();
            if (auto* m = instanced->instanceMatrix()) data["instanceMatrix"] = writeAttribute(*m);
            if (auto* c = instanced->instanceColor()) data["instanceColor"] = writeAttribute(*c);
        }

        // ---- skinning
        if (auto* skinned = dynamic_cast<SkinnedMesh*>(&object)) {
            data["bindMode"] = skinned->bindMode == SkinnedMesh::BindMode::Detached ? "detached" : "attached";
            data["bindMatrix"] = toArray(skinned->bindMatrix);
            if (skinned->skeleton) {
                data["skeleton"] = writeSkeleton(*skinned->skeleton, meta);
            }
        }

        // ---- LOD
        if (auto* lod = dynamic_cast<LOD*>(&object)) {
            data["autoUpdate"] = lod->autoUpdate;
            json levels = json::array();
            for (const auto& level : lod->getLevels()) {
                if (!level.object) continue;
                levels.push_back(json{{"object", level.object->uuid}, {"distance", level.distance}});
            }
            data["levels"] = levels;
        }

        // ---- sprite pivot (threepp extension; ignored by three.js)
        if (auto* sprite = dynamic_cast<Sprite*>(&object)) {
            data["center"] = toArray(sprite->center);
        }

        // ---- geometry / material
        if (object.is<Mesh>() || object.is<Line>() || object.is<Points>()) {
            if (auto geometry = object.geometry()) {
                data["geometry"] = writeGeometry(*geometry, meta);
            }
        }

        if (auto* withMaterials = dynamic_cast<ObjectWithMaterials*>(&object)) {
            const auto& materials = withMaterials->materials();
            if (materials.size() == 1 && materials.front()) {
                data["material"] = writeMaterial(*materials.front(), meta);
            } else if (materials.size() > 1) {
                json arr = json::array();
                for (const auto& m : materials) {
                    if (m) arr.push_back(writeMaterial(*m, meta));
                }
                data["material"] = arr;
            }
        } else if (auto material = object.material()) {
            data["material"] = writeMaterial(*material, meta);
        }

        // ---- animations (uuid references into the top-level array)
        if (!object.animations.empty()) {
            json arr = json::array();
            for (const auto& clip : object.animations) {
                if (clip) arr.push_back(writeAnimation(*clip, meta));
            }
            data["animations"] = arr;
        }

        // ---- children
        if (!object.children.empty()) {
            json arr = json::array();
            for (auto* child : object.children) {
                if (child) arr.push_back(writeObject(*child, meta));
            }
            data["children"] = arr;
        }

        return data;
    }

}// namespace

const std::vector<std::string>& ObjectExporter::warnings() const {

    return warnings_;
}

std::string ObjectExporter::toJson(Object3D& object, const ObjectExporterOptions& options) {

    warnings_.clear();

    Meta meta;
    meta.imageStorage = options.images;
    meta.modelStorage = options.models;
    meta.resourcePath = options.resourcePath;

    json output;
    output["metadata"] = json{
            {"version", 4.5},
            {"type", "Object"},
            {"generator", "threepp.ObjectExporter"}};

    auto objectJson = writeObject(object, meta);

    if (!meta.geometries.empty()) output["geometries"] = meta.geometries;
    if (!meta.materials.empty()) output["materials"] = meta.materials;
    if (!meta.textures.empty()) output["textures"] = meta.textures;
    if (!meta.images.empty()) output["images"] = meta.images;
    if (!meta.skeletons.empty()) output["skeletons"] = meta.skeletons;
    if (!meta.animations.empty()) output["animations"] = meta.animations;

    output["object"] = objectJson;

    warnings_ = std::move(meta.warnings);

    return options.prettyPrint ? output.dump(2) : output.dump();
}

void ObjectExporter::save(Object3D& object, const std::filesystem::path& path, const ObjectExporterOptions& options) {

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("[ObjectExporter] unable to open file for writing: " + path.string());
    }

    // References are relative to the document, and here we finally know where
    // the document is. An explicitly configured resourcePath still wins.
    auto resolved = options;
    if (resolved.resourcePath.empty()) resolved.resourcePath = path.parent_path();

    out << toJson(object, resolved);
}
