// https://github.com/mrdoob/three.js/blob/r129/src/loaders/ObjectLoader.js
// https://github.com/mrdoob/three.js/blob/r129/src/loaders/MaterialLoader.js
// https://github.com/mrdoob/three.js/blob/r129/src/loaders/BufferGeometryLoader.js

#include "threepp/loaders/ObjectLoader.hpp"

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
#include "threepp/loaders/ImageLoader.hpp"
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
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include "ObjectJsonConstants.hpp"
#include "threepp/utils/Base64.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace threepp;
using namespace threepp::objectjson;

namespace {

    // Everything the document asks for but threepp cannot represent is
    // collected here as well as logged, so an editor can surface it.
    struct Warnings {

        std::vector<std::string> messages;

        void add(std::string message) {

            std::cerr << "[ObjectLoader] " << message << std::endl;
            messages.push_back(std::move(message));
        }
    };

    using GeometryMap = std::unordered_map<std::string, std::shared_ptr<BufferGeometry>>;
    using MaterialMap = std::unordered_map<std::string, std::shared_ptr<Material>>;
    using TextureMap = std::unordered_map<std::string, std::shared_ptr<Texture>>;
    using ImageMap = std::unordered_map<std::string, std::vector<Image>>;
    using AnimationMap = std::unordered_map<std::string, std::shared_ptr<AnimationClip>>;
    using SkeletonMap = std::unordered_map<std::string, std::shared_ptr<Skeleton>>;

    template<class T>
    T value(const json& j, const char* key, T fallback) {

        if (!j.contains(key) || j.at(key).is_null()) return fallback;
        return j.at(key).get<T>();
    }

    Matrix4 matrixFrom(const json& j) {

        Matrix4 m;
        if (j.is_array() && j.size() == 16) {
            for (size_t i = 0; i < 16; ++i) m.elements[i] = j[i].get<float>();
        }
        return m;
    }

    Vector2 vector2From(const json& j, const Vector2& fallback = {}) {

        if (j.is_array() && j.size() >= 2) return {j[0].get<float>(), j[1].get<float>()};
        return fallback;
    }

    Color colorFrom(const json& j) {

        Color c;
        c.setHex(j.get<unsigned int>());
        return c;
    }

    void applyUserData(Object3D& object, const json& j, Warnings& warnings) {

        if (!j.is_object()) return;

        for (auto it = j.begin(); it != j.end(); ++it) {
            const auto& v = it.value();
            if (v.is_boolean()) {
                object.userData[it.key()] = v.get<bool>();
            } else if (v.is_number_integer() || v.is_number_unsigned()) {
                // JSON has no signed/unsigned distinction, so whole numbers come
                // back as int unless they do not fit.
                const auto n = v.get<std::int64_t>();
                if (n >= std::numeric_limits<int>::min() && n <= std::numeric_limits<int>::max()) {
                    object.userData[it.key()] = static_cast<int>(n);
                } else {
                    object.userData[it.key()] = n;
                }
            } else if (v.is_number_float()) {
                object.userData[it.key()] = v.get<double>();
            } else if (v.is_string()) {
                object.userData[it.key()] = v.get<std::string>();
            } else {
                warnings.add("skipping userData entry '" + it.key() + "': unsupported JSON type");
            }
        }
    }

    void applyLayers(Object3D& object, unsigned int mask) {

        object.layers.disableAll();
        for (unsigned int channel = 0; channel < 32; ++channel) {
            if (mask & (1u << channel)) object.layers.enable(channel);
        }
    }

    // -------------------------------------------------------------- images

    std::vector<Image> decodeImage(const json& url, int channels, const fs::path& resourcePath) {

        ImageLoader loader;
        std::vector<Image> out;

        const auto decodeOne = [&](const std::string& u) -> std::optional<Image> {
            if (u.rfind("data:", 0) == 0) {
                const auto comma = u.find(',');
                if (comma == std::string::npos) return std::nullopt;
                const auto bytes = utils::base64Decode(u.substr(comma + 1));
                return loader.load(bytes, channels, false);
            }
            return loader.load(resourcePath.empty() ? fs::path(u) : resourcePath / u, channels, false);
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

    ImageMap parseImages(const json& j, const fs::path& resourcePath, Warnings& warnings) {

        ImageMap images;
        if (!j.is_array()) return images;

        for (const auto& entry : j) {
            if (!entry.contains("uuid") || !entry.contains("url")) continue;

            const auto channels = value(entry, "threeppChannels", 4);
            auto decoded = decodeImage(entry["url"], channels, resourcePath);

            if (decoded.empty()) {
                warnings.add("could not decode image '" + entry["uuid"].get<std::string>() + "'");
                continue;
            }

            images[entry["uuid"].get<std::string>()] = std::move(decoded);
        }

        return images;
    }

    // ------------------------------------------------------------ textures

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
            setColor("sheen");
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

    // ---------------------------------------------------------- geometries

    std::shared_ptr<BufferAttribute> parseAttribute(const json& j) {

        const auto type = value<std::string>(j, "type", "Float32Array");
        const int itemSize = value(j, "itemSize", 1);
        const bool normalized = value(j, "normalized", false);

        if (!j.contains("array")) return nullptr;

        std::shared_ptr<BufferAttribute> attribute;

        if (isFloatArrayName(type)) {
            attribute = FloatBufferAttribute::create(j["array"].get<std::vector<float>>(), itemSize, normalized);
        } else {
            // Integer typed arrays: threepp keeps CPU-side attribute data as
            // float or unsigned int only, so anything else lands on float
            // unless it is an index (handled by the caller).
            attribute = FloatBufferAttribute::create(j["array"].get<std::vector<float>>(), itemSize, normalized);
        }

        if (j.contains("usage")) attribute->setUsage(static_cast<DrawUsage>(j["usage"].get<int>()));

        return attribute;
    }

    std::shared_ptr<BufferGeometry> parseDataGeometry(const json& data) {

        auto geometry = BufferGeometry::create();

        if (data.contains("index")) {
            geometry->setIndex(data["index"]["array"].get<std::vector<unsigned int>>());
        }

        if (data.contains("attributes")) {
            for (auto it = data["attributes"].begin(); it != data["attributes"].end(); ++it) {
                if (auto attribute = parseAttribute(it.value())) {
                    geometry->setAttribute(it.key(), attribute);
                }
            }
        }

        if (data.contains("morphAttributes")) {
            for (auto it = data["morphAttributes"].begin(); it != data["morphAttributes"].end(); ++it) {
                auto* target = geometry->getOrCreateMorphAttribute(it.key());
                for (const auto& entry : it.value()) {
                    if (auto attribute = parseAttribute(entry)) target->push_back(attribute);
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

    GeometryMap parseGeometries(const json& j, Warnings& warnings) {

        GeometryMap geometries;
        if (!j.is_array()) return geometries;

        for (const auto& entry : j) {
            if (!entry.contains("uuid")) continue;

            const auto type = value<std::string>(entry, "type", "BufferGeometry");

            std::shared_ptr<BufferGeometry> geometry;

            if (type == "BufferGeometry" || type == "InstancedBufferGeometry") {
                if (entry.contains("data")) {
                    geometry = parseDataGeometry(entry["data"]);
                } else {
                    geometry = BufferGeometry::create();
                }
            } else {
                geometry = parseParametricGeometry(type, entry);
                if (!geometry && entry.contains("data")) {
                    geometry = parseDataGeometry(entry["data"]);
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

    // -------------------------------------------------------------- objects

    struct ParseContext {

        const GeometryMap& geometries;
        const MaterialMap& materials;
        const TextureMap& textures;
        const AnimationMap& animations;
        Warnings& warnings;
    };

    std::shared_ptr<BufferGeometry> lookupGeometry(const json& j, const ParseContext& ctx) {

        if (!j.contains("geometry") || !j["geometry"].is_string()) return nullptr;

        const auto it = ctx.geometries.find(j["geometry"].get<std::string>());
        if (it == ctx.geometries.end()) {
            ctx.warnings.add("undefined geometry '" + j["geometry"].get<std::string>() + "'");
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<Material>> lookupMaterials(const json& j, const ParseContext& ctx) {

        std::vector<std::shared_ptr<Material>> out;
        if (!j.contains("material")) return out;

        const auto resolve = [&](const std::string& uuid) -> std::shared_ptr<Material> {
            const auto it = ctx.materials.find(uuid);
            if (it == ctx.materials.end()) {
                ctx.warnings.add("undefined material '" + uuid + "'");
                return nullptr;
            }
            return it->second;
        };

        if (j["material"].is_array()) {
            for (const auto& entry : j["material"]) {
                if (auto m = resolve(entry.get<std::string>())) out.push_back(m);
            }
        } else if (j["material"].is_string()) {
            if (auto m = resolve(j["material"].get<std::string>())) out.push_back(m);
        }

        return out;
    }

    void applyInstanceAttribute(FloatBufferAttribute* target, const json& j) {

        if (!target || !j.contains("array")) return;

        const auto source = j["array"].get<std::vector<float>>();
        auto& dst = target->array();
        const auto n = std::min(source.size(), dst.size());
        std::copy_n(source.begin(), n, dst.begin());
        target->needsUpdate();
    }

    void applyShadow(LightWithShadow& light, const json& j) {

        if (!light.shadow) return;

        auto& shadow = *light.shadow;
        shadow.bias = value(j, "bias", shadow.bias);
        shadow.normalBias = value(j, "normalBias", shadow.normalBias);
        shadow.radius = value(j, "radius", shadow.radius);
        if (j.contains("mapSize")) shadow.mapSize = vector2From(j["mapSize"], shadow.mapSize);

        if (!j.contains("camera") || !shadow.camera) return;

        const auto& cam = j["camera"];
        // Keep the existing camera instance (its concrete type is dictated by
        // the light) but adopt the serialized identity and parameters.
        if (cam.contains("uuid")) shadow.camera->uuid = cam["uuid"].get<std::string>();
        shadow.camera->name = value<std::string>(cam, "name", "");
        shadow.camera->zoom = value(cam, "zoom", shadow.camera->zoom);
        shadow.camera->nearPlane = value(cam, "near", shadow.camera->nearPlane);
        shadow.camera->farPlane = value(cam, "far", shadow.camera->farPlane);

        if (auto* persp = dynamic_cast<PerspectiveCamera*>(shadow.camera.get())) {
            persp->fov = value(cam, "fov", persp->fov);
            persp->aspect = value(cam, "aspect", persp->aspect);
            persp->focus = value(cam, "focus", persp->focus);
        } else if (auto* ortho = dynamic_cast<OrthographicCamera*>(shadow.camera.get())) {
            ortho->left = value(cam, "left", ortho->left);
            ortho->right = value(cam, "right", ortho->right);
            ortho->top = value(cam, "top", ortho->top);
            ortho->bottom = value(cam, "bottom", ortho->bottom);
        }

        shadow.camera->updateProjectionMatrix();
    }

    std::shared_ptr<Object3D> createObject(const json& j, const ParseContext& ctx) {

        const auto type = value<std::string>(j, "type", "Object3D");

        auto geometry = lookupGeometry(j, ctx);
        auto materials = lookupMaterials(j, ctx);
        auto material = materials.empty() ? nullptr : materials.front();

        if (type == "Scene") {

            auto scene = Scene::create();

            if (j.contains("background")) {
                if (j["background"].is_number()) {
                    scene->background = Background(colorFrom(j["background"]));
                } else if (j["background"].is_string()) {
                    const auto it = ctx.textures.find(j["background"].get<std::string>());
                    if (it != ctx.textures.end()) scene->background = Background(it->second);
                }
            }

            if (j.contains("environment") && j["environment"].is_string()) {
                const auto it = ctx.textures.find(j["environment"].get<std::string>());
                if (it != ctx.textures.end()) scene->environment = it->second;
            }

            if (j.contains("fog") && j["fog"].is_object()) {
                const auto& fog = j["fog"];
                const auto fogType = value<std::string>(fog, "type", "Fog");
                if (fogType == "FogExp2") {
                    scene->fog = FogExp2(colorFrom(fog["color"]), value(fog, "density", 0.00025f));
                } else {
                    scene->fog = Fog(colorFrom(fog["color"]), value(fog, "near", 1.f), value(fog, "far", 1000.f));
                }
            }

            return scene;
        }

        if (type == "PerspectiveCamera") {
            auto camera = PerspectiveCamera::create(
                    value(j, "fov", 50.f), value(j, "aspect", 1.f),
                    value(j, "near", 0.1f), value(j, "far", 2000.f));
            camera->zoom = value(j, "zoom", 1.f);
            camera->focus = value(j, "focus", 10.f);
            camera->filmGauge = value(j, "filmGauge", 35.f);
            camera->filmOffset = value(j, "filmOffset", 0.f);
            camera->updateProjectionMatrix();
            return camera;
        }

        if (type == "OrthographicCamera") {
            auto camera = OrthographicCamera::create(
                    value(j, "left", -1.f), value(j, "right", 1.f),
                    value(j, "top", 1.f), value(j, "bottom", -1.f),
                    value(j, "near", 0.1f), value(j, "far", 2000.f));
            camera->zoom = value(j, "zoom", 1.f);
            camera->updateProjectionMatrix();
            return camera;
        }

        if (type == "AmbientLight") {
            return AmbientLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f));
        }
        if (type == "DirectionalLight") {
            auto light = DirectionalLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f));
            if (j.contains("shadow")) applyShadow(*light, j["shadow"]);
            return light;
        }
        if (type == "PointLight") {
            auto light = PointLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f),
                                            value(j, "distance", 0.f), value(j, "decay", 1.f));
            if (j.contains("shadow")) applyShadow(*light, j["shadow"]);
            return light;
        }
        if (type == "SpotLight") {
            auto light = SpotLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f),
                                           value(j, "distance", 0.f), value(j, "angle", math::PI / 3),
                                           value(j, "penumbra", 0.f), value(j, "decay", 1.f));
            if (j.contains("shadow")) applyShadow(*light, j["shadow"]);
            return light;
        }
        if (type == "HemisphereLight") {
            return HemisphereLight::create(colorFrom(j.value("color", 0xffffffu)),
                                           colorFrom(j.value("groundColor", 0xffffffu)),
                                           value(j, "intensity", 1.f));
        }
        if (type == "RectAreaLight") {
            return RectAreaLight::create(colorFrom(j.value("color", 0xffffffu)), value(j, "intensity", 1.f),
                                         value(j, "width", 1.f), value(j, "height", 1.f));
        }

        if (type == "SkinnedMesh") {
            auto mesh = SkinnedMesh::create(geometry, material);
            if (materials.size() > 1) mesh->setMaterials(materials);
            mesh->bindMode = value<std::string>(j, "bindMode", "attached") == "detached"
                                     ? SkinnedMesh::BindMode::Detached
                                     : SkinnedMesh::BindMode::Attached;
            if (j.contains("bindMatrix")) {
                mesh->bindMatrix = matrixFrom(j["bindMatrix"]);
                mesh->bindMatrixInverse.copy(mesh->bindMatrix).invert();
            }
            return mesh;
        }

        if (type == "InstancedMesh") {
            auto mesh = InstancedMesh::create(geometry, material, value(j, "count", size_t(0)));
            if (materials.size() > 1) mesh->setMaterials(materials);
            if (j.contains("instanceMatrix")) applyInstanceAttribute(mesh->instanceMatrix(), j["instanceMatrix"]);
            if (j.contains("instanceColor")) {
                // Allocate the per-instance colour buffer through the public API
                // before filling it in.
                Color c;
                mesh->setColorAt(0, c);
                applyInstanceAttribute(mesh->instanceColor(), j["instanceColor"]);
            }
            return mesh;
        }

        if (type == "Mesh") {
            auto mesh = Mesh::create(geometry, material);
            if (materials.size() > 1) mesh->setMaterials(materials);
            return mesh;
        }

        if (type == "LOD") return LOD::create();
        if (type == "LineSegments") return LineSegments::create(geometry, material);
        if (type == "LineLoop") return LineLoop::create(geometry, material);
        if (type == "Line") return Line::create(geometry, material);
        if (type == "Points") {
            auto points = Points::create(geometry ? geometry : BufferGeometry::create(),
                                         material ? material : PointsMaterial::create());
            if (materials.size() > 1) points->setMaterials(materials);
            return points;
        }
        if (type == "Sprite") {
            auto spriteMaterial = std::dynamic_pointer_cast<SpriteMaterial>(material);
            auto sprite = Sprite::create(spriteMaterial ? spriteMaterial : SpriteMaterial::create());
            if (j.contains("center")) sprite->center = vector2From(j["center"], {0.5f, 0.5f});
            return sprite;
        }
        if (type == "Group") return Group::create();
        if (type == "Bone") return Bone::create();
        if (type == "Object3D") return Object3D::create();

        ctx.warnings.add("unsupported object type '" + type + "' - skipped");
        return nullptr;
    }

    std::shared_ptr<Object3D> parseObject(const json& j, const ParseContext& ctx) {

        auto object = createObject(j, ctx);
        if (!object) return nullptr;

        if (j.contains("uuid")) object->uuid = j["uuid"].get<std::string>();
        object->name = value<std::string>(j, "name", "");

        if (j.contains("matrix")) {
            const auto matrix = matrixFrom(j["matrix"]);
            object->matrix->copy(matrix);
            object->matrix->decompose(object->position, object->quaternion, object->scale);
            object->rotation.setFromQuaternion(object->quaternion);
            object->matrixAutoUpdate = value(j, "matrixAutoUpdate", true);
        }

        object->castShadow = value(j, "castShadow", false);
        object->receiveShadow = value(j, "receiveShadow", false);
        object->visible = value(j, "visible", true);
        object->frustumCulled = value(j, "frustumCulled", true);
        // three.js allows a negative renderOrder; threepp's field is unsigned.
        object->renderOrder = static_cast<unsigned int>(std::max(0, value(j, "renderOrder", 0)));

        if (j.contains("layers") && j["layers"].is_number()) {
            applyLayers(*object, static_cast<unsigned int>(j["layers"].get<std::int64_t>()));
        }

        if (j.contains("userData")) applyUserData(*object, j["userData"], ctx.warnings);

        if (j.contains("animations")) {
            for (const auto& entry : j["animations"]) {
                if (!entry.is_string()) continue;
                const auto it = ctx.animations.find(entry.get<std::string>());
                if (it != ctx.animations.end()) object->animations.push_back(it->second);
            }
        }

        std::vector<std::shared_ptr<Object3D>> children;
        if (j.contains("children")) {
            for (const auto& entry : j["children"]) {
                if (auto child = parseObject(entry, ctx)) children.push_back(child);
            }
        }

        auto* lod = object->as<LOD>();

        if (lod && j.contains("levels")) {

            lod->autoUpdate = value(j, "autoUpdate", true);

            std::vector<bool> consumed(children.size(), false);

            for (const auto& level : j["levels"]) {
                const auto uuid = value<std::string>(level, "object", "");
                for (size_t i = 0; i < children.size(); ++i) {
                    if (consumed[i] || children[i]->uuid != uuid) continue;
                    lod->addLevel(children[i], value(level, "distance", 0.f));
                    consumed[i] = true;
                    break;
                }
            }

            for (size_t i = 0; i < children.size(); ++i) {
                if (!consumed[i]) object->add(children[i]);
            }

        } else {
            for (const auto& child : children) object->add(child);
        }

        return object;
    }

    SkeletonMap parseSkeletons(const json& j, Object3D& root, Warnings& warnings) {

        SkeletonMap skeletons;
        if (!j.is_array()) return skeletons;

        std::unordered_map<std::string, Bone*> bones;
        root.traverse([&](Object3D& o) {
            if (auto* bone = o.as<Bone>()) bones[bone->uuid] = bone;
        });

        for (const auto& entry : j) {

            std::vector<std::shared_ptr<Bone>> boneList;
            std::vector<Matrix4> boneInverses;

            if (entry.contains("bones")) {
                for (const auto& uuid : entry["bones"]) {
                    const auto it = bones.find(uuid.get<std::string>());
                    if (it == bones.end()) {
                        warnings.add("undefined bone '" + uuid.get<std::string>() + "'");
                        continue;
                    }
                    // Non-owning: the bone is already owned by the object tree.
                    boneList.emplace_back(it->second, [](Bone*) {});
                }
            }

            if (entry.contains("boneInverses")) {
                for (const auto& m : entry["boneInverses"]) boneInverses.push_back(matrixFrom(m));
            }

            auto skeleton = Skeleton::create(boneList, boneInverses);
            if (entry.contains("uuid")) skeleton->setUuid(entry["uuid"].get<std::string>());

            skeletons[skeleton->uuid()] = skeleton;
        }

        return skeletons;
    }

    void bindSkeletons(Object3D& root, const SkeletonMap& skeletons, const json& objectJson, Warnings& warnings) {

        if (skeletons.empty()) return;

        // The skeleton uuid lives on the SkinnedMesh JSON entry, so walk the
        // document in parallel with the tree.
        std::unordered_map<std::string, std::string> meshToSkeleton;

        const std::function<void(const json&)> collect = [&](const json& j) {
            if (j.contains("skeleton") && j.contains("uuid")) {
                meshToSkeleton[j["uuid"].get<std::string>()] = j["skeleton"].get<std::string>();
            }
            if (j.contains("children")) {
                for (const auto& child : j["children"]) collect(child);
            }
        };
        collect(objectJson);

        root.traverse([&](Object3D& o) {
            auto* mesh = o.as<SkinnedMesh>();
            if (!mesh) return;

            const auto it = meshToSkeleton.find(mesh->uuid);
            if (it == meshToSkeleton.end()) return;

            const auto skeleton = skeletons.find(it->second);
            if (skeleton == skeletons.end()) {
                warnings.add("undefined skeleton '" + it->second + "'");
                return;
            }

            mesh->bind(skeleton->second, mesh->bindMatrix);
        });
    }

}// namespace

void ObjectLoader::setResourcePath(const std::filesystem::path& path) {

    resourcePath_ = path;
}

const std::vector<std::string>& ObjectLoader::warnings() const {

    return warnings_;
}

std::shared_ptr<Object3D> ObjectLoader::parse(const std::string& jsonText) {

    warnings_.clear();

    Warnings warnings;

    const auto j = json::parse(jsonText, nullptr, false);

    if (j.is_discarded() || !j.is_object()) {
        warnings.add("malformed JSON");
        warnings_ = std::move(warnings.messages);
        return nullptr;
    }

    if (!j.contains("object")) {
        warnings.add("document has no 'object' entry");
        warnings_ = std::move(warnings.messages);
        return nullptr;
    }

    const auto animations = parseAnimations(j.contains("animations") ? j["animations"] : json(), warnings);
    const auto geometries = parseGeometries(j.contains("geometries") ? j["geometries"] : json(), warnings);
    const auto images = parseImages(j.contains("images") ? j["images"] : json(), resourcePath_, warnings);
    const auto textures = parseTextures(j.contains("textures") ? j["textures"] : json(), images);
    const auto materials = parseMaterials(j.contains("materials") ? j["materials"] : json(), textures, warnings);

    const ParseContext ctx{geometries, materials, textures, animations, warnings};

    auto object = parseObject(j["object"], ctx);

    if (object) {
        const auto skeletons = parseSkeletons(j.contains("skeletons") ? j["skeletons"] : json(), *object, warnings);
        bindSkeletons(*object, skeletons, j["object"], warnings);
    }

    warnings_ = std::move(warnings.messages);

    return object;
}

std::shared_ptr<Object3D> ObjectLoader::load(const std::filesystem::path& path) {

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[ObjectLoader] unable to open " << path.string() << std::endl;
        return nullptr;
    }

    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};

    // Default the base directory to this document's, without clobbering an
    // explicit setResourcePath() or leaking into the next load().
    const auto configured = resourcePath_;
    if (resourcePath_.empty()) resourcePath_ = path.parent_path();

    auto object = parse(text);

    resourcePath_ = configured;

    return object;
}
