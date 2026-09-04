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

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
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

    // ------------------------------------------------------------ resolution
    //
    // Where a `url` in the document resolves from. A document is either a file
    // in a directory next to its images/ and buffers/, or a .tpz archive whose
    // members carry those same names — the JSON is identical either way, which
    // is the whole design: the archive is packaging, not a second format.
    //
    // Only the archive side answers here. A directory-backed image stays on
    // ImageLoader's path-taking overload, which sniffs its own file (see
    // decodeImage): handing it bytes instead would change behaviour for every
    // document that ever referenced an image by path.
    struct DocumentSource {

        std::filesystem::path resourcePath;
        const ZipReader* archive{nullptr};

        [[nodiscard]] bool has(const std::string& url) const {

            if (archive) return archive->has(url);

            std::error_code ec;
            return !resourcePath.empty() && std::filesystem::is_regular_file(resourcePath / url, ec);
        }

        [[nodiscard]] std::optional<std::vector<unsigned char>> read(const std::string& url) const {

            if (archive) {

                if (!archive->has(url)) return std::nullopt;
                return archive->read(url);
            }

            const auto path = resourcePath.empty() ? fs::path(url) : resourcePath / url;

            std::ifstream in(path, std::ios::binary | std::ios::ate);
            if (!in) return std::nullopt;

            const std::streamoff end = in.tellg();
            if (end < 0) return std::nullopt;

            std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
            in.seekg(0, std::ios::beg);
            if (end > 0) {

                in.read(reinterpret_cast<char*>(bytes.data()), end);
                if (in.gcount() != end) return std::nullopt;
            }

            return bytes;
        }
    };

    // The binary sections geometry attributes point into, read once each. A
    // geometry's attributes and its index all name the same section, so without
    // this a mesh with five attributes would read (and, in an archive, copy out
    // of the loaded archive) the same megabytes five times over.
    //
    // A section that cannot be read is remembered as absent, so the warning is
    // one per section rather than one per attribute.
    class BufferCache {

    public:
        explicit BufferCache(const DocumentSource& source): source_(source) {}

        const std::vector<unsigned char>* get(const std::string& url, Warnings& warnings) {

            const auto it = sections_.find(url);
            if (it != sections_.end()) return it->second ? &*it->second : nullptr;

            auto bytes = source_.read(url);
            if (!bytes) warnings.add("cannot read binary section '" + url + "'");

            const auto inserted = sections_.emplace(url, std::move(bytes)).first;
            return inserted->second ? &*inserted->second : nullptr;
        }

    private:
        const DocumentSource& source_;
        std::unordered_map<std::string, std::optional<std::vector<unsigned char>>> sections_;
    };

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
        // Base directory for the paths of referenced assets, same as for
        // referenced images.
        const std::filesystem::path& resourcePath;
        // The archive this document came out of, and where it is. A linked
        // asset may be a member of it, and re-importing one has to be able to
        // say which archive it came from — see resolveLinkedAsset.
        const ZipReader* archive{nullptr};
        const std::filesystem::path& archivePath;
        // ObjectLoader::setSplatCloudResolver, consulted before a splat cloud
        // is loaded from its file. Null or empty means always load.
        const ObjectLoader::SplatCloudResolver* splatResolver{nullptr};
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

    std::shared_ptr<Object3D> createSplatCloud(const json& j, const ParseContext& ctx);

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
            if (j.contains("instanceColor") && mesh->count() > 0) {
                // Allocate the per-instance colour buffer through the public API
                // before filling it in. A count-0 mesh has no buffer to allocate
                // (and setColorAt would rightly throw), so skip it.
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
        if (type == "SplatCloud") return createSplatCloud(j, ctx);
        // A robot writes its type as the plain Object3D it also is, so a reader
        // without the articulation extension still gets the hierarchy — frozen,
        // exactly as before this existed. With it, the object comes back live.
        if (type == "Object3D") {
            if (j.contains("threeppRobot")) return Robot::create();
            return Object3D::create();
        }

        ctx.warnings.add("unsupported object type '" + type + "' - skipped");
        return nullptr;
    }

    // ------------------------------------------------------- articulation

    // Rebuild a Robot's joint table over the hierarchy that has just been
    // parsed. See ObjectExporter::writeArticulation for what is in the block and
    // why it is there; the short version is that this is what stops a document
    // round trip from having to re-import the URDF, which is what used to throw
    // away everything authored into a robot's subtree.
    //
    // The nodes are the document's, referenced by uuid. Robot's tables hold
    // shared_ptrs and its origPose_/articulatedJoints_ hold RAW pointers into
    // the same nodes, so the table has to be a co-owner: a URDF-built robot owns
    // its links and joints outright, and that ownership is what keeps those raw
    // pointers valid when a link is deleted out of the hierarchy. Sharing the
    // node's existing control block (shared_from_this) keeps that property
    // without duplicating the node.
    void restoreArticulation(Robot& robot, const json& j, Warnings& warnings) {

        std::unordered_map<std::string, Object3D*> byUuid;
        robot.traverse([&](Object3D& node) { byUuid.emplace(node.uuid, &node); });

        const auto find = [&](const json& entry, const char* key) -> Object3D* {
            if (!entry.contains(key) || !entry[key].is_string()) return nullptr;
            const auto it = byUuid.find(entry[key].get<std::string>());
            return it == byUuid.end() ? nullptr : it->second;
        };

        const auto refer = [](Object3D* node) {
            // Everything the loader builds is heap-shared; a node that is not
            // (one on the stack) can still be referred to, just not co-owned.
            if (node->weak_from_this().expired()) {
                return std::shared_ptr<Object3D>(node, [](Object3D*) {});
            }
            return node->shared_from_this();
        };

        std::size_t missing = 0;

        if (j.contains("links") && j["links"].is_array()) {
            for (const auto& uuid : j["links"]) {
                if (!uuid.is_string()) continue;
                const auto it = byUuid.find(uuid.get<std::string>());
                if (it == byUuid.end()) {
                    ++missing;
                    continue;
                }
                robot.addLink(refer(it->second));
            }
        }

        // Values are applied only after the tables are complete: setJointValue
        // indexes by DOF, and the DOF count is not known until the last joint
        // is in.
        std::vector<float> values;

        if (j.contains("joints") && j["joints"].is_array()) {
            for (const auto& entry : j["joints"]) {

                auto* node = find(entry, "node");
                if (!node) {
                    ++missing;
                    continue;
                }

                Robot::JointInfo info;
                info.name = value<std::string>(entry, "name", "");
                const auto type = value<std::string>(entry, "type", "fixed");
                info.type = type == "revolute"    ? Robot::JointType::Revolute
                            : type == "prismatic" ? Robot::JointType::Prismatic
                                                  : Robot::JointType::Fixed;
                info.parent = value<std::string>(entry, "parent", "");
                info.child = value<std::string>(entry, "child", "");

                if (entry.contains("axis") && entry["axis"].size() >= 3) {
                    const auto& a = entry["axis"];
                    info.axis.set(a[0].get<float>(), a[1].get<float>(), a[2].get<float>());
                }
                if (entry.contains("limit") && entry["limit"].size() >= 2) {
                    const auto& l = entry["limit"];
                    info.range = Robot::JointRange{l[0].get<float>(), l[1].get<float>()};
                }

                // The node is sitting in its DRIVEN pose, which is the document's
                // to keep; the rest pose is the one the table needs.
                Vector3 restPosition{node->position};
                Quaternion restRotation{node->quaternion};
                if (entry.contains("rest") && entry["rest"].size() >= 7) {
                    const auto& r = entry["rest"];
                    restPosition.set(r[0].get<float>(), r[1].get<float>(), r[2].get<float>());
                    restRotation.set(r[3].get<float>(), r[4].get<float>(),
                                     r[5].get<float>(), r[6].get<float>());
                }

                robot.addJoint(refer(node), info, restPosition, restRotation);

                if (info.type != Robot::JointType::Fixed) {
                    values.push_back(value(entry, "value", 0.f));
                }
            }
        }

        robot.finalizeInPlace();

        if (const auto ee = value<std::string>(j, "endEffector", ""); !ee.empty()) {
            robot.setEndEffector(ee);
        }

        // Re-driving from the stored values does not move anything — the nodes
        // are already in this pose — but it is what puts the same numbers in the
        // joint table, so an inspector reads the angles the robot is standing in
        // rather than a row of zeros.
        for (std::size_t i = 0; i < values.size() && i < robot.numDOF(); ++i) {
            robot.setJointValue(i, values[i]);
        }

        if (missing > 0) {
            warnings.add("robot '" + robot.name + "': the articulation table references " +
                         std::to_string(missing) +
                         " node(s) missing from the hierarchy - those joints came back frozen");
        }
    }

    // ------------------------------------------------- linked asset subtrees

    // The xacro arguments the editor recorded on the placeholder when the robot was
    // imported. Re-importing without them would rebuild a DIFFERENT robot — a UR5e
    // saved as a UR5e would come back as whatever the file defaults to. Nothing about
    // RobotConfig is known at this level, and nothing needs to be: the entries are
    // plain strings, read by the one reader the loaders layer owns.
    std::map<std::string, std::string> readXacroArgs(const Object3D& object) {

        std::map<std::string, std::string> args;
        for (const auto& [name, value] : xacro::readArgsUserData(object)) args[name] = value;
        return args;
    }

    // `why` collects the loader's own account of a failure, which for a xacro is
    // the difference between a usable report and "it did not load".
    std::shared_ptr<Object3D> importAsset(const std::filesystem::path& path,
                                          const std::map<std::string, std::string>& xacroArgs,
                                          std::string& why) {

        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (extension == ".urdf" || extension == ".xacro") {
            URDFLoader loader;
            if (!xacroArgs.empty()) loader.setArgs(xacroArgs);
            auto robot = loader.load(path);
            if (!robot) why = loader.lastError();
            return robot;
        }

        ModelLoader loader;
        return loader.load(path);
    }

    // Pre-order, root excluded — the exact walk ObjectExporter numbered the
    // override table against.
    std::vector<Object3D*> flattenDescendants(Object3D& root) {

        std::vector<Object3D*> flat;

        std::function<void(Object3D&)> collect = [&](Object3D& node) {
            for (auto* child : node.children) {
                if (!child) continue;
                flat.push_back(child);
                collect(*child);
            }
        };
        collect(root);

        return flat;
    }

    // Replays the edits the document recorded on top of a freshly imported
    // subtree. Each entry carries the name the node had when the document was
    // written; a mismatch means the asset file has changed underneath and the
    // index no longer identifies the same node, so the edit is dropped rather
    // than applied to whatever now sits at that position.
    void applyAssetOverrides(Object3D& root, const json& nodes, Warnings& warnings) {

        if (!nodes.is_array()) return;

        const auto flat = flattenDescendants(root);

        std::size_t stale = 0;

        for (const auto& entry : nodes) {

            if (!entry.contains("i") || !entry["i"].is_number_integer()) continue;

            const auto index = entry["i"].get<std::int64_t>();
            if (index < 0 || static_cast<std::size_t>(index) >= flat.size()) {
                ++stale;
                continue;
            }

            auto& node = *flat[static_cast<std::size_t>(index)];
            if (node.name != value<std::string>(entry, "name", "")) {
                ++stale;
                continue;
            }

            if (entry.contains("matrix")) {
                node.matrix->copy(matrixFrom(entry["matrix"]));
                node.matrix->decompose(node.position, node.quaternion, node.scale);
                node.rotation.setFromQuaternion(node.quaternion);
            }

            node.visible = value(entry, "visible", true);
            node.castShadow = value(entry, "castShadow", false);
            node.receiveShadow = value(entry, "receiveShadow", false);
            node.frustumCulled = value(entry, "frustumCulled", true);
            node.renderOrder = value(entry, "renderOrder", 0);

            if (entry.contains("layers") && entry["layers"].is_number()) {
                applyLayers(node, static_cast<unsigned int>(entry["layers"].get<std::int64_t>()));
            }
            if (entry.contains("userData")) applyUserData(node, entry["userData"], warnings);
        }

        // One message, not one per node: a changed asset invalidates the whole
        // table at once and a few thousand identical lines help nobody.
        if (stale > 0) {
            warnings.add("linked asset '" + root.name + "' has changed since the document was saved - " +
                         std::to_string(stale) + " of " + std::to_string(nodes.size()) +
                         " saved edits could not be matched to a node");
        }
    }

    // One archive member, written to a temp file for the length of an import.
    //
    // Every importer in the tree takes a path, and giving them all a second
    // entry point that takes bytes would be a far larger change than writing
    // the bytes down — which is also why only self-contained formats may travel
    // in an archive: a file whose siblings are missing would import to
    // something quietly wrong. The extension is kept because that is what
    // importAsset dispatches on, and the uuid is what keeps two loads (or two
    // processes) out of each other's way.
    class TempAsset {

    public:
        explicit TempAsset(const std::string& entry, const std::vector<unsigned char>& bytes) {

            const std::filesystem::path name{entry};
            path_ = std::filesystem::temp_directory_path() /
                    ("threepp-asset-" + math::generateUUID() + name.extension().string());

            std::ofstream out(path_, std::ios::binary | std::ios::trunc);
            if (!out) return;
            if (!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()),
                                          static_cast<std::streamsize>(bytes.size()));
            ok_ = out.good();
        }

        TempAsset(const TempAsset&) = delete;
        TempAsset& operator=(const TempAsset&) = delete;

        ~TempAsset() {

            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }

        [[nodiscard]] bool ok() const { return ok_; }

        [[nodiscard]] const std::filesystem::path& path() const { return path_; }

    private:
        std::filesystem::path path_;
        bool ok_{false};
    };

    // Returns the re-imported subtree, or nullptr when the asset could not be
    // loaded — in which case the caller keeps the empty placeholder, so the
    // rest of the scene still opens.
    std::shared_ptr<Object3D> resolveLinkedAsset(const json& ref, const Object3D& placeholder,
                                                 const ParseContext& ctx) {

        const auto stored = value<std::string>(ref, "path", "");
        if (stored.empty()) {
            ctx.warnings.add("linked asset '" + placeholder.name + "' has no path");
            return nullptr;
        }

        // An asset that lives inside an archive: a member of the one this
        // document came out of, or one named by a "<archive>|<entry>" mark that
        // an earlier load left behind — a document saved loose can still point
        // into an archive, and that path means what it says wherever it is read.
        std::string markArchive, markEntry;
        std::optional<ZipReader> other;
        const ZipReader* holder = nullptr;

        if (splitArchiveAsset(stored, markArchive, markEntry)) {

            try {

                other.emplace(markArchive);
                holder = &*other;

            } catch (const std::exception& e) {
                ctx.warnings.add("could not open archive '" + markArchive + "' for linked asset '" +
                                 placeholder.name + "': " + e.what());
                return nullptr;
            }

        } else if (ctx.archive && ctx.archive->has(stored)) {

            markArchive = ctx.archivePath.generic_string();
            markEntry = stored;
            holder = ctx.archive;
        }

        std::optional<TempAsset> extracted;
        if (holder) {

            if (!holder->has(markEntry)) {
                ctx.warnings.add("archive '" + markArchive + "' has no entry '" + markEntry + "'");
                return nullptr;
            }

            extracted.emplace(markEntry, holder->read(markEntry));
            if (!extracted->ok()) {
                ctx.warnings.add("could not extract '" + markEntry + "' from '" + markArchive + "'");
                return nullptr;
            }
        }

        std::filesystem::path path{stored};
        if (extracted) {

            path = extracted->path();

        } else {

            if (path.is_relative() && !ctx.resourcePath.empty()) path = ctx.resourcePath / path;

            // Normalised so that re-saving this document writes the same relative
            // path it was loaded from, rather than one with a '..' baked in.
            std::error_code ec;
            if (auto canonical = std::filesystem::weakly_canonical(path, ec); !ec && !canonical.empty()) {
                path = canonical;
            }
        }

        std::shared_ptr<Object3D> imported;
        std::string error;
        try {
            imported = importAsset(path, readXacroArgs(placeholder), error);
        } catch (const std::exception& e) {
            error = e.what();
        }

        if (!imported) {
            ctx.warnings.add("could not re-import linked asset '" + path.string() + "'" +
                             (error.empty() ? "" : ": " + error));
            return nullptr;
        }

        // Identity and placement belong to the document; everything below the
        // root belongs to the asset file.
        imported->uuid = placeholder.uuid;
        imported->name = placeholder.name;
        imported->matrix->copy(*placeholder.matrix);
        imported->matrix->decompose(imported->position, imported->quaternion, imported->scale);
        imported->rotation.setFromQuaternion(imported->quaternion);
        imported->matrixAutoUpdate = placeholder.matrixAutoUpdate;
        imported->castShadow = placeholder.castShadow;
        imported->receiveShadow = placeholder.receiveShadow;
        imported->visible = placeholder.visible;
        imported->frustumCulled = placeholder.frustumCulled;
        imported->renderOrder = placeholder.renderOrder;
        applyLayers(*imported, placeholder.layers.mask());
        imported->userData = placeholder.userData;

        // Where the asset actually is now, which is not necessarily where the
        // machine that saved the document had it — and for one that came out of
        // an archive, the archive and the entry rather than the temp file this
        // import read, which is gone by the time anything asks. That mark is
        // what lets a re-save copy the bytes straight across.
        setAssetSource(*imported, extracted ? markArchive + archiveAssetMark + markEntry : path.string());

        if (ref.contains("nodes")) applyAssetOverrides(*imported, ref["nodes"], ctx.warnings);

        return imported;
    }

    // ------------------------------------------------------- splat clouds

    // Where a `threeppSplat` path resolves to on this machine: an archive
    // member extracted to a temp file (kept alive by `extracted`), or a path
    // relative to the document made absolute. `mark` is what userData's source
    // entry should say afterwards — the archive-and-entry mark for a member,
    // the absolute path otherwise — so a re-save copies the same bytes.
    bool resolveSplatPath(const std::string& stored, const ParseContext& ctx,
                          std::optional<TempAsset>& extracted, std::filesystem::path& path,
                          std::string& mark, std::string& why) {

        std::string markArchive, markEntry;
        std::optional<ZipReader> other;
        const ZipReader* holder = nullptr;

        if (splitArchiveAsset(stored, markArchive, markEntry)) {

            try {
                other.emplace(markArchive);
                holder = &*other;
            } catch (const std::exception& e) {
                why = "could not open archive '" + markArchive + "': " + e.what();
                return false;
            }

        } else if (ctx.archive && ctx.archive->has(stored)) {

            markArchive = ctx.archivePath.generic_string();
            markEntry = stored;
            holder = ctx.archive;
        }

        if (holder) {

            if (!holder->has(markEntry)) {
                why = "archive '" + markArchive + "' has no entry '" + markEntry + "'";
                return false;
            }
            extracted.emplace(markEntry, holder->read(markEntry));
            if (!extracted->ok()) {
                why = "could not extract '" + markEntry + "' from '" + markArchive + "'";
                return false;
            }
            path = extracted->path();
            mark = markArchive + archiveAssetMark + markEntry;
            return true;
        }

        path = std::filesystem::u8path(stored);
        if (path.is_relative() && !ctx.resourcePath.empty()) path = ctx.resourcePath / path;
        std::error_code ec;
        if (auto canonical = std::filesystem::weakly_canonical(path, ec); !ec && !canonical.empty()) {
            path = canonical;
        }
        const auto u8 = path.u8string();
        mark = std::string(u8.begin(), u8.end());
        return true;
    }

    // A splat cloud, from the `threeppSplat` block ObjectExporter wrote: the
    // live object the resolver hands back (the play snapshot's case), else
    // the file loaded by its content — a SOG asset, a 3DGS .ply or a colour
    // point cloud — with the importer's ops replayed. Anything that fails
    // leaves a plain Object3D placeholder, placed and named, with a warning.
    std::shared_ptr<Object3D> createSplatCloud(const json& j, const ParseContext& ctx) {

        const std::string name = value<std::string>(j, "name", value<std::string>(j, "uuid", "?"));
        const auto placeholder = [&](const std::string& why) -> std::shared_ptr<Object3D> {
            ctx.warnings.add("splat cloud '" + name + "' could not be restored: " + why +
                             " - an empty node stands in for it");
            return Object3D::create();
        };

        if (!j.contains("threeppSplat") || !j["threeppSplat"].is_object()) {
            return placeholder("no threeppSplat block");
        }
        const json& ref = j["threeppSplat"];
        const float pointMix = value(ref, "pointMix", 0.f);
        const float pointSize = value(ref, "pointSize", 2.f);

        if (ctx.splatResolver && *ctx.splatResolver) {
            if (auto live = (*ctx.splatResolver)(value<std::string>(j, "uuid", ""))) {
                live->setPointMix(pointMix);
                live->setPointSize(pointSize);
                return live;
            }
        }

        const auto stored = value<std::string>(ref, "path", "");
        if (stored.empty()) {
            return placeholder("the document carries no file for it (a snapshot without its live object?)");
        }

        std::optional<TempAsset> extracted;
        std::filesystem::path path;
        std::string mark, why;
        if (!resolveSplatPath(stored, ctx, extracted, path, mark, why)) return placeholder(why);

        const bool cull = value(ref, "cull", false);
        const int lod = value(ref, "lod", -1);

        SplatData data;
        splats::LodTable table;
        try {

            if (SogLoader::isSog(path)) {
                if (lod >= 0) {
                    data = SogLoader::load(path, {lod});
                } else {
                    auto loaded = splats::loadSogWithLod(path);
                    data = std::move(loaded.data);
                    table = std::move(loaded.table);
                }
            } else if (SplatLoader::isSplatPly(path)) {
                data = SplatLoader::loadPly(path);
            } else if (SplatLoader::isPointCloudPly(path)) {
                data = SplatLoader::loadPointCloudPly(path);
            } else {
                std::error_code ec;
                return placeholder(std::filesystem::exists(path, ec)
                                           ? "'" + path.string() + "' is not a splat scan or a point cloud"
                                           : "'" + path.string() + "' does not exist");
            }

        } catch (const std::exception& e) {
            return placeholder(e.what());
        }

        // The importer's cull, replayed on the same data: deterministic, so
        // the same splats go. Never under a LOD table, whose offsets it would
        // invalidate (the editor's import makes the same exception).
        if (cull && table.empty()) data.removeOutliers();

        auto cloud = SplatCloud::create(std::move(data));
        if (!table.empty()) cloud->setLodTable(std::move(table));
        cloud->setPointMix(pointMix);
        cloud->setPointSize(pointSize);
        // Where the file is NOW, for the inspector and the next save. parseObject
        // overwrites userData from the document right after this; restampSplatSource
        // puts it back.
        cloud->userData[splatSourceKey] = mark;
        return cloud;
    }

    // After the document's userData landed on the cloud: the source entry it
    // carries is the saving machine's absolute path, and the one createSplatCloud
    // resolved is this machine's. Same rule as assetSource on a linked model.
    void restampSplatSource(Object3D& object, const json& j, const ParseContext& ctx) {

        auto* cloud = dynamic_cast<SplatCloud*>(&object);
        if (!cloud || !j.contains("threeppSplat")) return;
        const auto stored = value<std::string>(j["threeppSplat"], "path", "");
        if (stored.empty()) return;

        std::optional<TempAsset> extracted;// not extracted here: has() decides the mark
        std::string markArchive, markEntry, mark;
        if (splitArchiveAsset(stored, markArchive, markEntry)) {
            mark = stored;
        } else if (ctx.archive && ctx.archive->has(stored)) {
            mark = ctx.archivePath.generic_string() + archiveAssetMark + stored;
        } else {
            std::filesystem::path path = std::filesystem::u8path(stored);
            if (path.is_relative() && !ctx.resourcePath.empty()) path = ctx.resourcePath / path;
            std::error_code ec;
            if (auto canonical = std::filesystem::weakly_canonical(path, ec); !ec && !canonical.empty()) {
                path = canonical;
            }
            const auto u8 = path.u8string();
            mark = std::string(u8.begin(), u8.end());
        }
        object.userData[splatSourceKey] = mark;
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
        object->renderOrder = value(j, "renderOrder", 0);

        if (j.contains("layers") && j["layers"].is_number()) {
            applyLayers(*object, static_cast<unsigned int>(j["layers"].get<std::int64_t>()));
        }

        if (j.contains("userData")) applyUserData(*object, j["userData"], ctx.warnings);
        restampSplatSource(*object, j, ctx);

        if (j.contains("animations")) {
            for (const auto& entry : j["animations"]) {
                if (!entry.is_string()) continue;
                const auto it = ctx.animations.find(entry.get<std::string>());
                if (it != ctx.animations.end()) object->animations.push_back(it->second);
            }
        }

        // A linked subtree has no children in the document — it has a file to
        // read them from. On failure the placeholder stands: correctly placed
        // and named, just empty, and the warning says why.
        if (j.contains("threeppAsset")) {
            if (auto linked = resolveLinkedAsset(j["threeppAsset"], *object, ctx)) return linked;
            return object;
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

        // After the children: the joint table names the nodes it drives, and
        // they have to be in the tree before they can be found.
        if (j.contains("threeppRobot")) {
            if (auto* robot = object->as<Robot>()) {
                restoreArticulation(*robot, j["threeppRobot"], ctx.warnings);
            }
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

void ObjectLoader::setSplatCloudResolver(SplatCloudResolver resolver) {

    splatResolver_ = std::move(resolver);
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

    const DocumentSource source{resourcePath_, archive_.get()};
    BufferCache buffers{source};

    const auto animations = parseAnimations(j.contains("animations") ? j["animations"] : json(), warnings);
    const auto geometries = parseGeometries(j.contains("geometries") ? j["geometries"] : json(), buffers, warnings);
    const auto images = parseImages(j.contains("images") ? j["images"] : json(), source, warnings);
    const auto textures = parseTextures(j.contains("textures") ? j["textures"] : json(), images);
    const auto materials = parseMaterials(j.contains("materials") ? j["materials"] : json(), textures, warnings);

    const ParseContext ctx{geometries, materials, textures, animations, warnings,
                           resourcePath_, archive_.get(), archivePath_, &splatResolver_};

    auto object = parseObject(j["object"], ctx);

    if (object) {
        const auto skeletons = parseSkeletons(j.contains("skeletons") ? j["skeletons"] : json(), *object, warnings);
        bindSkeletons(*object, skeletons, j["object"], warnings);
    }

    warnings_ = std::move(warnings.messages);

    return object;
}

std::shared_ptr<Object3D> ObjectLoader::load(const std::filesystem::path& path) {

    std::string text;

    // Sniffed, not decided by the extension: a .tpz is a zip whatever it is
    // called, and a JSON document is not one however it is named.
    if (ZipReader::looksLikeZip(path)) {

        try {

            archive_ = std::make_shared<ZipReader>(path);

        } catch (const std::exception& e) {

            std::cerr << "[ObjectLoader] " << e.what() << std::endl;
            return nullptr;
        }

        if (!archive_->has(objectjson::archiveDocument)) {

            std::cerr << "[ObjectLoader] " << path.string() << " is an archive with no "
                      << objectjson::archiveDocument << std::endl;
            archive_.reset();
            return nullptr;
        }

        const auto bytes = archive_->read(objectjson::archiveDocument);
        text.assign(bytes.begin(), bytes.end());

        // Absolute, because it goes into the assetSource mark of anything
        // re-imported out of it and that mark outlives this call.
        std::error_code ec;
        auto absolute = std::filesystem::weakly_canonical(path, ec);
        archivePath_ = ec || absolute.empty() ? path : absolute;

    } else {

        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "[ObjectLoader] unable to open " << path.string() << std::endl;
            return nullptr;
        }

        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    // Default the base directory to this document's, without clobbering an
    // explicit setResourcePath() or leaking into the next load(). An archive
    // resolves its own urls, but the document may still point outside itself —
    // at a linked asset, at an image it was saved before the archive existed —
    // and those are relative to the archive's directory.
    const auto configured = resourcePath_;
    if (resourcePath_.empty()) resourcePath_ = path.parent_path();

    auto object = parse(text);

    resourcePath_ = configured;
    archive_.reset();
    archivePath_.clear();

    return object;
}
