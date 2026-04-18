// glTF 2.0 loader for threepp

#include "threepp/loaders/GLTFLoader.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <threepp/animation/AnimationClip.hpp>
#include <threepp/animation/tracks/NumberKeyframeTrack.hpp>
#include <threepp/animation/tracks/QuaternionKeyframeTrack.hpp>
#include <threepp/animation/tracks/VectorKeyframeTrack.hpp>
#include <threepp/lights/DirectionalLight.hpp>
#include <threepp/lights/PointLight.hpp>
#include <threepp/lights/SpotLight.hpp>
#include <threepp/loaders/ImageLoader.hpp>
#include <threepp/materials/MeshBasicMaterial.hpp>
#include <threepp/materials/MeshPhysicalMaterial.hpp>
#include <threepp/objects/Bone.hpp>
#include <threepp/objects/ObjectWithMaterials.hpp>
#include <threepp/objects/Skeleton.hpp>
#include <threepp/objects/SkinnedMesh.hpp>

#include <unordered_set>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace threepp {

    // ===========================================================================
    //  Base64 decoder
    // ===========================================================================
    namespace {

        constexpr std::array<int, 256> buildDecodeTable() {
            std::array<int, 256> t{};
            t.fill(-1);
            for (int i = 0; i < 26; ++i) {
                t['A' + i] = i;
                t['a' + i] = i + 26;
            }
            for (int i = 0; i < 10; ++i) t['0' + i] = i + 52;
            t['+'] = 62;
            t['/'] = 63;
            t['='] = 0;
            return t;
        }

        constexpr auto kDecodeTable = buildDecodeTable();

        std::vector<uint8_t> base64Decode(const std::string& encoded) {
            std::vector<uint8_t> out;
            out.reserve(encoded.size() * 3 / 4);
            int val = 0, valb = -8;
            for (unsigned char c : encoded) {
                if (c == '\n' || c == '\r' || c == ' ') continue;
                int d = kDecodeTable[c];
                if (d == -1) break;
                val = (val << 6) + d;
                valb += 6;
                if (valb >= 0) {
                    out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
                    valb -= 8;
                }
            }
            return out;
        }

        // ===========================================================================
        //  glTF constants
        // ===========================================================================
        constexpr uint32_t GLB_MAGIC = 0x46546C67;     // "glTF"
        constexpr uint32_t GLB_CHUNK_JSON = 0x4E4F534A;// "JSON"
        constexpr uint32_t GLB_CHUNK_BIN = 0x004E4942; // "BIN\0"

        // glTF accessor component types
        constexpr int COMP_BYTE = 5120;
        constexpr int COMP_UNSIGNED_BYTE = 5121;
        constexpr int COMP_SHORT = 5122;
        constexpr int COMP_UNSIGNED_SHORT = 5123;
        constexpr int COMP_UNSIGNED_INT = 5125;
        constexpr int COMP_FLOAT = 5126;

        // glTF texture wrap / filter modes (mapped to OpenGL values)
        constexpr int WRAP_REPEAT = 10497;
        constexpr int WRAP_CLAMP_TO_EDGE = 33071;
        constexpr int WRAP_MIRRORED_REPEAT = 33648;

        int componentSize(int componentType) {
            switch (componentType) {
                case COMP_BYTE:
                case COMP_UNSIGNED_BYTE:
                    return 1;
                case COMP_SHORT:
                case COMP_UNSIGNED_SHORT:
                    return 2;
                case COMP_UNSIGNED_INT:
                case COMP_FLOAT:
                    return 4;
            }
            return 0;
        }

        int typeCount(const std::string& type) {
            if (type == "SCALAR") return 1;
            if (type == "VEC2") return 2;
            if (type == "VEC3") return 3;
            if (type == "VEC4") return 4;
            if (type == "MAT2") return 4;
            if (type == "MAT3") return 9;
            if (type == "MAT4") return 16;
            return 0;
        }

        // ===========================================================================
        //  Parser state
        // ===========================================================================
        struct GLTFParser {
            json gltf;
            std::vector<std::vector<uint8_t>> buffers;
            fs::path basePath;

            // Cache to avoid duplicate GPU uploads
            std::unordered_map<int, std::shared_ptr<Texture>> textureCache;
            std::unordered_map<int, std::shared_ptr<Material>> materialCache;

            // Skeleton support
            std::unordered_set<int> jointNodeSet;
            std::unordered_map<int, std::shared_ptr<Object3D>> nodeObjects;
            std::unordered_map<int, std::shared_ptr<Skeleton>> skinCache;
            std::unordered_set<int> builtNodes;

            // KHR_materials_variants support
            std::vector<std::string> variantNames;
            struct PrimVariantMapping {
                int materialIdx;
                std::vector<int> variantIndices;
            };
            // meshIdx -> primIdx -> list of variant mappings
            std::unordered_map<int, std::unordered_map<int, std::vector<PrimVariantMapping>>> primVariantData;

            // -----------------------------------------------------------------------
            //  Buffer/accessor helpers
            // -----------------------------------------------------------------------

            const std::vector<uint8_t>& resolveBuffer(int idx) {
                if (idx < static_cast<int>(buffers.size()) && !buffers[idx].empty())
                    return buffers[idx];

                const auto& bufDef = gltf["buffers"][idx];
                std::string uri = bufDef.value("uri", "");
                if (uri.empty()) throw std::runtime_error("Buffer " + std::to_string(idx) + " has no URI");

                if (uri.rfind("data:", 0) == 0) {
                    // data URI — find the comma
                    auto comma = uri.find(',');
                    buffers[idx] = base64Decode(uri.substr(comma + 1));
                } else {
                    fs::path p = basePath / uri;
                    std::ifstream f(p, std::ios::binary);
                    if (!f) throw std::runtime_error("Cannot open buffer file: " + p.string());
                    buffers[idx].assign(std::istreambuf_iterator<char>(f), {});
                }
                return buffers[idx];
            }

            struct AccessorData {
                const uint8_t* ptr;
                size_t byteStride;
                size_t count;
                int componentType;
                int numComponents;
            };

            AccessorData getAccessor(int accessorIdx) {
                const auto& acc = gltf["accessors"][accessorIdx];
                int bvIdx = acc["bufferView"].get<int>();
                const auto& bv = gltf["bufferViews"][bvIdx];
                int bufIdx = bv["buffer"].get<int>();
                size_t bvOffset = bv.value("byteOffset", 0);
                size_t bvStride = bv.value("byteStride", 0);
                size_t accOff = acc.value("byteOffset", 0);
                size_t count = acc["count"].get<size_t>();
                int ct = acc["componentType"].get<int>();
                std::string type = acc["type"].get<std::string>();
                int nc = typeCount(type);

                const auto& buf = resolveBuffer(bufIdx);
                const uint8_t* base = buf.data() + bvOffset + accOff;
                size_t stride = bvStride > 0 ? bvStride : static_cast<size_t>(componentSize(ct) * nc);

                return {base, stride, count, ct, nc};
            }

            // Read accessor into flat float vector
            std::vector<float> readFloats(int accessorIdx) {
                auto [ptr, stride, count, ct, nc] = getAccessor(accessorIdx);
                std::vector<float> out;
                out.reserve(count * nc);
                for (size_t i = 0; i < count; ++i) {
                    const uint8_t* row = ptr + i * stride;
                    for (int j = 0; j < nc; ++j) {
                        const uint8_t* src = row + j * componentSize(ct);
                        float val = 0.f;
                        switch (ct) {
                            case COMP_FLOAT: {
                                float tmp;
                                std::memcpy(&tmp, src, 4);
                                val = tmp;
                                break;
                            }
                            case COMP_UNSIGNED_BYTE:
                                val = *src / 255.f;
                                break;
                            case COMP_UNSIGNED_SHORT: {
                                uint16_t tmp;
                                std::memcpy(&tmp, src, 2);
                                val = tmp / 65535.f;
                                break;
                            }
                            case COMP_SHORT: {
                                int16_t tmp;
                                std::memcpy(&tmp, src, 2);
                                val = std::max(-1.f, tmp / 32767.f);
                                break;
                            }
                            default:
                                break;
                        }
                        out.push_back(val);
                    }
                }
                return out;
            }

            std::vector<uint32_t> readIndices(int accessorIdx) {
                auto [ptr, stride, count, ct, nc] = getAccessor(accessorIdx);
                std::vector<uint32_t> out;
                out.reserve(count);
                for (size_t i = 0; i < count; ++i) {
                    const uint8_t* src = ptr + i * stride;
                    uint32_t val = 0;
                    switch (ct) {
                        case COMP_UNSIGNED_BYTE:
                            val = *src;
                            break;
                        case COMP_UNSIGNED_SHORT:
                            std::memcpy(&val, src, 2);
                            val &= 0xFFFF;
                            break;
                        case COMP_UNSIGNED_INT:
                            std::memcpy(&val, src, 4);
                            break;
                        default:
                            break;
                    }
                    out.push_back(val);
                }
                return out;
            }

            // Read JOINTS_0 accessor as float without normalisation
            std::vector<float> readJointIndicesAsFloat(int accessorIdx) {
                auto [ptr, stride, count, ct, nc] = getAccessor(accessorIdx);
                std::vector<float> out;
                out.reserve(count * nc);
                for (size_t i = 0; i < count; ++i) {
                    const uint8_t* row = ptr + i * stride;
                    for (int j = 0; j < nc; ++j) {
                        const uint8_t* src = row + j * componentSize(ct);
                        float val = 0.f;
                        if (ct == COMP_UNSIGNED_BYTE)
                            val = static_cast<float>(*src);
                        else if (ct == COMP_UNSIGNED_SHORT) {
                            uint16_t tmp;
                            std::memcpy(&tmp, src, 2);
                            val = static_cast<float>(tmp);
                        }
                        out.push_back(val);
                    }
                }
                return out;
            }

            // -----------------------------------------------------------------------
            //  Skeleton helpers
            // -----------------------------------------------------------------------

            void gatherJoints() {
                if (!gltf.contains("skins")) return;
                for (const auto& skin : gltf["skins"]) {
                    if (!skin.contains("joints")) continue;
                    for (int ji : skin["joints"].get<std::vector<int>>())
                        jointNodeSet.insert(ji);
                }
            }

            void applyNodeTransform(const std::shared_ptr<Object3D>& obj, const json& nodeDef) {
                if (nodeDef.contains("matrix")) {
                    auto m = nodeDef["matrix"].get<std::vector<float>>();
                    Matrix4 mat4;
                    mat4.set(m[0], m[4], m[8],  m[12],
                             m[1], m[5], m[9],  m[13],
                             m[2], m[6], m[10], m[14],
                             m[3], m[7], m[11], m[15]);
                    obj->applyMatrix4(mat4);
                } else {
                    if (nodeDef.contains("translation")) {
                        auto t = nodeDef["translation"].get<std::vector<float>>();
                        obj->position.set(t[0], t[1], t[2]);
                    }
                    if (nodeDef.contains("rotation")) {
                        auto r = nodeDef["rotation"].get<std::vector<float>>();
                        obj->quaternion.set(r[0], r[1], r[2], r[3]);
                    }
                    if (nodeDef.contains("scale")) {
                        auto s = nodeDef["scale"].get<std::vector<float>>();
                        obj->scale.set(s[0], s[1], s[2]);
                    }
                }
            }

            // Pre-create all nodes so Bone objects exist before skin binding
            void preCreateNodes() {
                if (!gltf.contains("nodes")) return;
                int n = static_cast<int>(gltf["nodes"].size());
                for (int i = 0; i < n; ++i) {
                    const auto& nodeDef = gltf["nodes"][i];
                    std::shared_ptr<Object3D> obj;
                    if (jointNodeSet.count(i))
                        obj = Bone::create();
                    else
                        obj = Group::create();
                    obj->name = nodeDef.value("name", "");
                    applyNodeTransform(obj, nodeDef);
                    nodeObjects[i] = obj;
                }
            }

            std::shared_ptr<Skeleton> loadSkin(int skinIdx) {
                auto it = skinCache.find(skinIdx);
                if (it != skinCache.end()) return it->second;

                const auto& skinDef = gltf["skins"][skinIdx];
                auto jointIndices = skinDef["joints"].get<std::vector<int>>();

                std::vector<std::shared_ptr<Bone>> bones;
                for (int ji : jointIndices) {
                    auto nit = nodeObjects.find(ji);
                    if (nit != nodeObjects.end()) {
                        if (auto bone = std::dynamic_pointer_cast<Bone>(nit->second))
                            bones.push_back(bone);
                    }
                }

                std::vector<Matrix4> ibms;
                if (skinDef.contains("inverseBindMatrices")) {
                    auto floats = readFloats(skinDef["inverseBindMatrices"].get<int>());
                    for (size_t i = 0; i < bones.size(); ++i) {
                        const float* f = floats.data() + i * 16;
                        Matrix4 m;
                        // glTF is column-major; Matrix4::set takes row-major
                        m.set(f[0], f[4], f[8],  f[12],
                              f[1], f[5], f[9],  f[13],
                              f[2], f[6], f[10], f[14],
                              f[3], f[7], f[11], f[15]);
                        ibms.push_back(m);
                    }
                }

                auto skel = Skeleton::create(bones, ibms);
                skinCache[skinIdx] = skel;
                return skel;
            }

            // -----------------------------------------------------------------------
            //  Image / Texture loading
            // -----------------------------------------------------------------------

            std::optional<Image> loadImageData(int imageIdx) {
                const auto& imgDef = gltf["images"][imageIdx];
                std::vector<uint8_t> raw;

                if (imgDef.contains("bufferView")) {
                    int bvIdx = imgDef["bufferView"].get<int>();
                    const auto& bv = gltf["bufferViews"][bvIdx];
                    int bufIdx = bv["buffer"].get<int>();
                    size_t off = bv.value("byteOffset", 0);
                    size_t len = bv["byteLength"].get<size_t>();
                    const auto& buf = resolveBuffer(bufIdx);
                    raw.assign(buf.data() + off, buf.data() + off + len);
                } else if (imgDef.contains("uri")) {
                    std::string uri = imgDef["uri"].get<std::string>();
                    if (uri.rfind("data:", 0) == 0) {
                        auto comma = uri.find(',');
                        raw = base64Decode(uri.substr(comma + 1));
                    } else {
                        fs::path p = basePath / uri;
                        std::ifstream f(p, std::ios::binary);
                        if (!f) throw std::runtime_error("Cannot open image: " + p.string());
                        raw.assign(std::istreambuf_iterator<char>(f), {});
                    }
                } else {
                    throw std::runtime_error("Image " + std::to_string(imageIdx) + " has no source");
                }

                ImageLoader loader;
                return loader.load(raw, 4, false);
            }

            std::shared_ptr<Texture> loadTexture(int texIdx) {
                auto it = textureCache.find(texIdx);
                if (it != textureCache.end()) return it->second;

                const auto& texDef = gltf["textures"][texIdx];
                int imageIdx = texDef.value("source", -1);
                if (imageIdx < 0) return nullptr;

                auto image = loadImageData(imageIdx);

                auto tex = Texture::create(*image);
                tex->needsUpdate();

                // Apply sampler settings if present
                if (texDef.contains("sampler") && gltf.contains("samplers")) {
                    const auto& samp = gltf["samplers"][texDef["sampler"].get<int>()];
                    int wrapS = samp.value("wrapS", WRAP_REPEAT);
                    int wrapT = samp.value("wrapT", WRAP_REPEAT);
                    auto toWrap = [](int w) -> TextureWrapping {
                        if (w == WRAP_CLAMP_TO_EDGE) return TextureWrapping::ClampToEdge;
                        if (w == WRAP_MIRRORED_REPEAT) return TextureWrapping::MirroredRepeat;
                        return TextureWrapping::Repeat;
                    };
                    tex->wrapS = toWrap(wrapS);
                    tex->wrapT = toWrap(wrapT);
                }

                textureCache[texIdx] = tex;
                return tex;
            }

            // Apply KHR_texture_transform from a textureInfo JSON node.
            // Returns a (possibly cloned) texture with transform applied.
            std::shared_ptr<Texture> applyTextureTransform(
                    const json& texInfo, std::shared_ptr<Texture> tex) {
                if (!tex) return tex;
                int texCoordVal = texInfo.value("texCoord", 0);
                bool hasTransform = false;
                float offX = 0, offY = 0, scX = 1, scY = 1, rot = 0;
                if (texInfo.contains("extensions") &&
                    texInfo["extensions"].contains("KHR_texture_transform")) {
                    hasTransform = true;
                    const auto& tt = texInfo["extensions"]["KHR_texture_transform"];
                    if (tt.contains("offset")) {
                        offX = tt["offset"][0].get<float>();
                        offY = tt["offset"][1].get<float>();
                    }
                    if (tt.contains("scale")) {
                        scX = tt["scale"][0].get<float>();
                        scY = tt["scale"][1].get<float>();
                    }
                    rot = tt.value("rotation", 0.0f);
                    texCoordVal = tt.value("texCoord", texCoordVal);
                }
                if (!hasTransform && texCoordVal == 0) return tex;
                // Clone to avoid sharing transforms between channels
                auto clone = tex->clone();
                clone->offset = {offX, offY};
                clone->repeat = {scX, scY};
                clone->rotation = rot;
                clone->center = {0, 0};
                clone->texCoord = texCoordVal;
                clone->updateMatrix();
                return clone;
            }

            // -----------------------------------------------------------------------
            //  Material
            // -----------------------------------------------------------------------

            std::shared_ptr<Material> loadMaterial(int matIdx) {
                auto it = materialCache.find(matIdx);
                if (it != materialCache.end()) return it->second;

                const auto& matDef = gltf["materials"][matIdx];

                // KHR_materials_unlit → MeshBasicMaterial
                if (matDef.contains("extensions") &&
                    matDef["extensions"].contains("KHR_materials_unlit")) {
                    auto basicMat = MeshBasicMaterial::create();
                    basicMat->name = matDef.value("name", "");
                    if (matDef.contains("pbrMetallicRoughness")) {
                        const auto& pbr = matDef["pbrMetallicRoughness"];
                        if (pbr.contains("baseColorFactor")) {
                            auto f = pbr["baseColorFactor"].get<std::vector<float>>();
                            basicMat->color = Color(f[0], f[1], f[2]);
                            if (f.size() > 3) basicMat->opacity = f[3];
                        }
                        if (pbr.contains("baseColorTexture")) {
                            int ti = pbr["baseColorTexture"]["index"].get<int>();
                            basicMat->map = applyTextureTransform(pbr["baseColorTexture"], loadTexture(ti));
                        }
                    }
                    std::string alphaMode = matDef.value("alphaMode", "OPAQUE");
                    if (alphaMode == "BLEND") {
                        basicMat->transparent = true;
                    } else if (alphaMode == "MASK") {
                        basicMat->alphaTest = matDef.value("alphaCutoff", 0.5f);
                    }
                    if (matDef.value("doubleSided", false)) {
                        basicMat->side = Side::Double;
                    }
                    materialCache[matIdx] = basicMat;
                    return basicMat;
                }

                // Check if we need MeshPhysicalMaterial (for transmission, clearcoat, etc.)
                bool needsPhysical = false;
                if (matDef.contains("extensions")) {
                    const auto& ext = matDef["extensions"];
                    if (ext.contains("KHR_materials_transmission") ||
                        ext.contains("KHR_materials_clearcoat") ||
                        ext.contains("KHR_materials_ior") ||
                        ext.contains("KHR_materials_dispersion") ||
                        ext.contains("KHR_materials_specular") ||
                        ext.contains("KHR_materials_sheen") ||
                        ext.contains("KHR_materials_volume")) {
                        needsPhysical = true;
                    }
                }

                std::shared_ptr<MeshStandardMaterial> mat;
                std::shared_ptr<MeshPhysicalMaterial> physMat;
                if (needsPhysical) {
                    physMat = MeshPhysicalMaterial::create();
                    mat = physMat;
                } else {
                    mat = MeshStandardMaterial::create();
                }
                mat->name = matDef.value("name", "");

                // PBR Metallic-Roughness
                if (matDef.contains("pbrMetallicRoughness")) {
                    const auto& pbr = matDef["pbrMetallicRoughness"];

                    // Base color factor
                    if (pbr.contains("baseColorFactor")) {
                        auto f = pbr["baseColorFactor"].get<std::vector<float>>();
                        mat->color = Color(f[0], f[1], f[2]);
                        if (f.size() > 3) mat->opacity = f[3];
                    }

                    // Base color texture
                    if (pbr.contains("baseColorTexture")) {
                        int ti = pbr["baseColorTexture"]["index"].get<int>();
                        mat->map = applyTextureTransform(pbr["baseColorTexture"], loadTexture(ti));
                    }

                    // Metalness / roughness
                    mat->metalness = pbr.value("metallicFactor", 1.0f);
                    mat->roughness = pbr.value("roughnessFactor", 1.0f);

                    // Metallic-roughness texture (G=roughness, B=metalness per spec)
                    if (pbr.contains("metallicRoughnessTexture")) {
                        int ti = pbr["metallicRoughnessTexture"]["index"].get<int>();
                        auto tex = applyTextureTransform(pbr["metallicRoughnessTexture"], loadTexture(ti));
                        mat->metalnessMap = tex;
                        mat->roughnessMap = tex;
                    }
                }

                // Normal map
                if (matDef.contains("normalTexture")) {
                    int ti = matDef["normalTexture"]["index"].get<int>();
                    mat->normalMap = applyTextureTransform(matDef["normalTexture"], loadTexture(ti));
                    float scale = matDef["normalTexture"].value("scale", 1.0f);
                    mat->normalScale = Vector2{scale, scale};
                }

                // Occlusion map
                if (matDef.contains("occlusionTexture")) {
                    int ti = matDef["occlusionTexture"]["index"].get<int>();
                    mat->aoMap = applyTextureTransform(matDef["occlusionTexture"], loadTexture(ti));
                    mat->aoMapIntensity = matDef["occlusionTexture"].value("strength", 1.0f);
                }

                // Emissive
                if (matDef.contains("emissiveFactor")) {
                    auto e = matDef["emissiveFactor"].get<std::vector<float>>();
                    mat->emissive = Color(e[0], e[1], e[2]);
                }
                if (matDef.contains("emissiveTexture")) {
                    int ti = matDef["emissiveTexture"]["index"].get<int>();
                    mat->emissiveMap = applyTextureTransform(matDef["emissiveTexture"], loadTexture(ti));
                }

                // Alpha mode
                std::string alphaMode = matDef.value("alphaMode", "OPAQUE");
                if (alphaMode == "BLEND") {
                    mat->transparent = true;
                } else if (alphaMode == "MASK") {
                    mat->alphaTest = matDef.value("alphaCutoff", 0.5f);
                }

                // Double-sided
                if (matDef.value("doubleSided", false)) {
                    mat->side = Side::Double;
                }

                // Extensions (MeshPhysicalMaterial properties)
                if (physMat && matDef.contains("extensions")) {
                    const auto& ext = matDef["extensions"];

                    // KHR_materials_transmission
                    if (ext.contains("KHR_materials_transmission")) {
                        const auto& tr = ext["KHR_materials_transmission"];
                        physMat->transmission = tr.value("transmissionFactor", 0.0f);
                        if (tr.contains("transmissionTexture")) {
                            int ti = tr["transmissionTexture"]["index"].get<int>();
                            physMat->transmissionMap = loadTexture(ti);
                        }
                    }

                    // KHR_materials_ior
                    if (ext.contains("KHR_materials_ior")) {
                        const auto& iorExt = ext["KHR_materials_ior"];
                        physMat->setIor(iorExt.value("ior", 1.5f));
                    }

                    // KHR_materials_dispersion
                    if (ext.contains("KHR_materials_dispersion")) {
                        const auto& dispExt = ext["KHR_materials_dispersion"];
                        physMat->dispersion = dispExt.value("dispersion", 0.0f);
                    }

                    // KHR_materials_emissive_strength
                    if (ext.contains("KHR_materials_emissive_strength")) {
                        float strength = ext["KHR_materials_emissive_strength"].value("emissiveStrength", 1.0f);
                        mat->emissiveIntensity = strength;
                    }

                    // KHR_materials_sheen
                    if (ext.contains("KHR_materials_sheen")) {
                        const auto& sh = ext["KHR_materials_sheen"];
                        if (sh.contains("sheenColorFactor")) {
                            auto c = sh["sheenColorFactor"];
                            physMat->sheenColor = Color(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
                        }
                        physMat->sheenRoughness = sh.value("sheenRoughnessFactor", 0.0f);
                    }

                    // KHR_materials_specular
                    if (ext.contains("KHR_materials_specular")) {
                        const auto& sp = ext["KHR_materials_specular"];
                        physMat->specularIntensity = sp.value("specularFactor", 1.0f);
                        if (sp.contains("specularColorFactor")) {
                            auto c = sp["specularColorFactor"];
                            physMat->specularColor = Color(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
                        }
                    }

                    // KHR_materials_volume
                    if (ext.contains("KHR_materials_volume")) {
                        const auto& vol = ext["KHR_materials_volume"];
                        physMat->attenuationDistance = vol.value("attenuationDistance", 0.0f);
                        if (vol.contains("attenuationColor")) {
                            auto c = vol["attenuationColor"];
                            physMat->attenuationColor = Color(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
                        }
                        physMat->thickness = vol.value("thicknessFactor", 0.0f);
                    }

                    // KHR_materials_clearcoat
                    if (ext.contains("KHR_materials_clearcoat")) {
                        const auto& cc = ext["KHR_materials_clearcoat"];
                        physMat->clearcoat = cc.value("clearcoatFactor", 0.0f);
                        if (cc.contains("clearcoatTexture")) {
                            int ti = cc["clearcoatTexture"]["index"].get<int>();
                            physMat->clearcoatMap = loadTexture(ti);
                        }
                        physMat->clearcoatRoughness = cc.value("clearcoatRoughnessFactor", 0.0f);
                        if (cc.contains("clearcoatRoughnessTexture")) {
                            int ti = cc["clearcoatRoughnessTexture"]["index"].get<int>();
                            physMat->clearcoatRoughnessMap = loadTexture(ti);
                        }
                        if (cc.contains("clearcoatNormalTexture")) {
                            int ti = cc["clearcoatNormalTexture"]["index"].get<int>();
                            physMat->clearcoatNormalMap = loadTexture(ti);
                            float scale = cc["clearcoatNormalTexture"].value("scale", 1.0f);
                            physMat->clearcoatNormalScale = Vector2{scale, scale};
                        }
                    }
                }

                materialCache[matIdx] = mat;
                return mat;
            }

            // -----------------------------------------------------------------------
            //  Mesh
            // -----------------------------------------------------------------------

            std::shared_ptr<Object3D> loadMesh(int meshIdx, bool hasSkin = false) {
                const auto& meshDef = gltf["meshes"][meshIdx];
                const auto& primitives = meshDef["primitives"];

                // Multiple primitives → Group
                auto group = Group::create();
                group->name = meshDef.value("name", "");

                int primIdx = 0;
                for (const auto& prim : primitives) {
                    auto geometry = BufferGeometry::create();

                    // --- Attributes ---
                    const auto& attrs = prim["attributes"];

                    auto addFloatAttr = [&](const char* gltfKey, const char* threeKey, int itemSize) {
                        if (!attrs.contains(gltfKey)) return;
                        int accIdx = attrs[gltfKey].get<int>();
                        auto data = readFloats(accIdx);
                        geometry->setAttribute(
                                threeKey,
                                FloatBufferAttribute::create(std::move(data), itemSize));
                    };

                    addFloatAttr("POSITION", "position", 3);
                    addFloatAttr("NORMAL", "normal", 3);
                    addFloatAttr("TEXCOORD_0", "uv", 2);
                    addFloatAttr("TEXCOORD_1", "uv2", 2);
                    // COLOR_0: use actual accessor component count (VEC3 or VEC4)
                    if (attrs.contains("COLOR_0")) {
                        int accIdx = attrs["COLOR_0"].get<int>();
                        auto [ptr, stride, count, ct, nc] = getAccessor(accIdx);
                        auto data = readFloats(accIdx);
                        geometry->setAttribute("color",
                                FloatBufferAttribute::create(std::move(data), nc));
                    }
                    addFloatAttr("TANGENT", "tangent", 4);

                    if (hasSkin) {
                        if (attrs.contains("JOINTS_0")) {
                            auto data = readJointIndicesAsFloat(attrs["JOINTS_0"].get<int>());
                            geometry->setAttribute("skinIndex",
                                    FloatBufferAttribute::create(std::move(data), 4));
                        }
                        addFloatAttr("WEIGHTS_0", "skinWeight", 4);
                    }

                    // --- Indices ---
                    if (prim.contains("indices")) {
                        auto indices = readIndices(prim["indices"].get<int>());
                        geometry->setIndex(indices);
                    }

                    // Compute vertex normals if absent
                    if (!attrs.contains("NORMAL")) {
                        geometry->computeVertexNormals();
                    }

                    // --- Material ---
                    std::shared_ptr<Material> mat;
                    if (prim.contains("material") && gltf.contains("materials")) {
                        mat = loadMaterial(prim["material"].get<int>());
                    } else {
                        // Default: white MeshStandardMaterial
                        mat = MeshStandardMaterial::create();
                    }

                    std::shared_ptr<Mesh> mesh;
                    if (hasSkin && attrs.contains("JOINTS_0"))
                        mesh = SkinnedMesh::create(geometry, mat);
                    else
                        mesh = Mesh::create(geometry, mat);

                    // Tag for KHR_materials_variants post-load resolution
                    mesh->userData["__gltfMeshIdx"] = meshIdx;
                    mesh->userData["__gltfPrimIdx"] = primIdx;

                    // Collect per-primitive variant mappings
                    if (!variantNames.empty() &&
                        prim.contains("extensions") &&
                        prim["extensions"].contains("KHR_materials_variants")) {
                        const auto& vext = prim["extensions"]["KHR_materials_variants"];
                        if (vext.contains("mappings")) {
                            for (const auto& mapping : vext["mappings"]) {
                                PrimVariantMapping pvm;
                                pvm.materialIdx = mapping["material"].get<int>();
                                pvm.variantIndices = mapping["variants"].get<std::vector<int>>();
                                primVariantData[meshIdx][primIdx].push_back(std::move(pvm));
                            }
                        }
                    }

                    group->add(mesh);
                    ++primIdx;
                }

                // If only one primitive and not skinned, unwrap the group
                if (group->children.size() == 1 && !hasSkin) {
                    auto child = group->children[0];
                    child->name = group->name;
                    auto cloned = child->clone();
                    cloned->userData = child->userData;// Object3D::copy() doesn't copy userData
                    return cloned;
                }
                return group;
            }

            // -----------------------------------------------------------------------
            //  Node / Scene hierarchy
            // -----------------------------------------------------------------------

            // Build node hierarchy using pre-created node objects
            void buildNode(int nodeIdx) {
                if (!builtNodes.insert(nodeIdx).second) return; // already built
                const auto& nodeDef = gltf["nodes"][nodeIdx];
                auto& obj = nodeObjects[nodeIdx];

                if (nodeDef.contains("mesh") && gltf.contains("meshes")) {
                    int meshIdx = nodeDef["mesh"].get<int>();
                    int skinIdx = nodeDef.value("skin", -1);
                    bool hasSkin = skinIdx >= 0 && gltf.contains("skins");

                    auto meshObj = loadMesh(meshIdx, hasSkin);

                    if (hasSkin) {
                        auto skel = loadSkin(skinIdx);
                        // meshObj is always a Group when hasSkin (no unwrap)
                        for (auto child : meshObj->children) {
                            if (auto sm = dynamic_cast<SkinnedMesh*>(child))
                                sm->bind(skel, Matrix4());
                        }
                    }
                    obj->add(meshObj);
                }

                // KHR_lights_punctual: extract lights from glTF nodes
                if (nodeDef.contains("extensions") &&
                    nodeDef["extensions"].contains("KHR_lights_punctual")) {
                    int lightIdx = nodeDef["extensions"]["KHR_lights_punctual"]["light"].get<int>();
                    if (gltf.contains("extensions") &&
                        gltf["extensions"].contains("KHR_lights_punctual") &&
                        gltf["extensions"]["KHR_lights_punctual"].contains("lights")) {
                        const auto& lightDef = gltf["extensions"]["KHR_lights_punctual"]["lights"][lightIdx];
                        std::string ltype = lightDef.value("type", "point");
                        float intensity = lightDef.value("intensity", 1.0f);
                        Color color(1.f, 1.f, 1.f);
                        if (lightDef.contains("color")) {
                            auto c = lightDef["color"].get<std::vector<float>>();
                            if (c.size() >= 3) color.setRGB(c[0], c[1], c[2]);
                        }
                        float range = lightDef.value("range", 0.0f);

                        std::shared_ptr<Light> light;
                        if (ltype == "directional") {
                            light = DirectionalLight::create(color, intensity);
                        } else if (ltype == "spot") {
                            float innerCone = lightDef.value("innerConeAngle", 0.0f);
                            float outerCone = lightDef.value("outerConeAngle", math::PI / 4.f);
                            float penumbra = (outerCone > 0.f) ? (1.f - innerCone / outerCone) : 0.f;
                            light = SpotLight::create(color, intensity, range, outerCone, penumbra);
                        } else {
                            // "point" or fallback
                            light = PointLight::create(color, intensity, range);
                        }
                        if (light) {
                            light->name = lightDef.value("name", "light_" + std::to_string(lightIdx));
                            light->visible = false;  // hidden by default; user opts in
                            obj->add(light);
                            std::cerr << "[GLTFLoader] Light: " << light->name
                                      << " type=" << ltype << " intensity=" << intensity
                                      << " range=" << range << std::endl;
                        }
                    }
                }

                if (nodeDef.contains("children")) {
                    for (int ci : nodeDef["children"].get<std::vector<int>>())
                        obj->add(nodeObjects[ci]);
                }
            }

            std::shared_ptr<Group> loadScene(int sceneIdx) {
                const auto& sceneDef = gltf["scenes"][sceneIdx];
                auto root = Group::create();
                root->name = sceneDef.value("name", "Scene");

                if (sceneDef.contains("nodes")) {
                    int numNodes = gltf.contains("nodes") ? static_cast<int>(gltf["nodes"].size()) : 0;
                    for (int i = 0; i < numNodes; ++i) buildNode(i);

                    for (int nodeIdx : sceneDef["nodes"].get<std::vector<int>>())
                        root->add(nodeObjects[nodeIdx]);
                }
                return root;
            }

            // -----------------------------------------------------------------------
            //  Entry points
            // -----------------------------------------------------------------------

            GLTFResult parseGLTF(const std::string& jsonText) {
                gltf = json::parse(jsonText);

                // Pre-allocate buffer slots
                int numBuffers = gltf.contains("buffers") ? static_cast<int>(gltf["buffers"].size()) : 0;
                buffers.resize(numBuffers);

                // Parse top-level KHR_materials_variants names
                if (gltf.contains("extensions") &&
                    gltf["extensions"].contains("KHR_materials_variants")) {
                    const auto& ext = gltf["extensions"]["KHR_materials_variants"];
                    if (ext.contains("variants")) {
                        for (const auto& v : ext["variants"])
                            variantNames.push_back(v.value("name", ""));
                    }
                }

                gatherJoints();
                preCreateNodes();

                GLTFResult result;
                int defaultScene = gltf.value("scene", 0);
                int numScenes = gltf.contains("scenes") ? static_cast<int>(gltf["scenes"].size()) : 0;

                for (int i = 0; i < numScenes; ++i) {
                    result.scenes.push_back(loadScene(i));
                }

                if (!result.scenes.empty()) {
                    int si = (defaultScene >= 0 && defaultScene < numScenes) ? defaultScene : 0;
                    result.scene = result.scenes[si];
                } else {
                    result.scene = Group::create();
                }

                result.animations = loadAnimations();
                resolveVariants(result);
                return result;
            }

            GLTFResult parseGLB(const std::vector<uint8_t>& data) {
                if (data.size() < 12) throw std::runtime_error("GLB too small");

                uint32_t magic, version, totalLength;
                std::memcpy(&magic, data.data(), 4);
                std::memcpy(&version, data.data() + 4, 4);
                std::memcpy(&totalLength, data.data() + 8, 4);

                if (magic != GLB_MAGIC) throw std::runtime_error("Not a GLB file (bad magic)");

                size_t offset = 12;
                std::string jsonText;
                bool gotJSON = false, gotBIN = false;

                while (offset + 8 <= data.size()) {
                    uint32_t chunkLen, chunkType;
                    std::memcpy(&chunkLen, data.data() + offset, 4);
                    std::memcpy(&chunkType, data.data() + offset + 4, 4);
                    offset += 8;

                    if (chunkType == GLB_CHUNK_JSON && !gotJSON) {
                        jsonText = std::string(reinterpret_cast<const char*>(data.data() + offset), chunkLen);
                        gotJSON = true;
                    } else if (chunkType == GLB_CHUNK_BIN && !gotBIN) {
                        // Buffer 0 is the embedded BIN chunk
                        buffers.resize(1);
                        buffers[0].assign(data.data() + offset, data.data() + offset + chunkLen);
                        gotBIN = true;
                    }

                    offset += chunkLen;
                }

                if (!gotJSON) throw std::runtime_error("GLB has no JSON chunk");

                gltf = json::parse(jsonText);
                int numBuffers = gltf.contains("buffers") ? static_cast<int>(gltf["buffers"].size()) : 0;
                if (static_cast<int>(buffers.size()) < numBuffers) buffers.resize(numBuffers);

                // Parse top-level KHR_materials_variants names
                if (gltf.contains("extensions") &&
                    gltf["extensions"].contains("KHR_materials_variants")) {
                    const auto& ext = gltf["extensions"]["KHR_materials_variants"];
                    if (ext.contains("variants")) {
                        for (const auto& v : ext["variants"])
                            variantNames.push_back(v.value("name", ""));
                    }
                }

                gatherJoints();
                preCreateNodes();

                GLTFResult result;
                int defaultScene = gltf.value("scene", 0);
                int numScenes = gltf.contains("scenes") ? static_cast<int>(gltf["scenes"].size()) : 0;

                for (int i = 0; i < numScenes; ++i) {
                    result.scenes.push_back(loadScene(i));
                }

                if (!result.scenes.empty()) {
                    int si = (defaultScene >= 0 && defaultScene < numScenes) ? defaultScene : 0;
                    result.scene = result.scenes[si];
                } else {
                    result.scene = Group::create();
                }

                result.animations = loadAnimations();
                resolveVariants(result);
                return result;
            }

            // -----------------------------------------------------------------------
            //  Animation
            // -----------------------------------------------------------------------

            std::vector<std::shared_ptr<AnimationClip>> loadAnimations() {
                if (!gltf.contains("animations")) return {};

                std::vector<std::shared_ptr<AnimationClip>> clips;

                for (size_t animIdx = 0; animIdx < gltf["animations"].size(); ++animIdx) {
                    const auto& animDef = gltf["animations"][animIdx];

                    std::string animName = animDef.value("name", "animation_" + std::to_string(animIdx));

                    if (!animDef.contains("channels") || !animDef.contains("samplers")) continue;

                    const auto& channels = animDef["channels"];
                    const auto& samplers = animDef["samplers"];

                    std::vector<std::shared_ptr<KeyframeTrack>> tracks;

                    for (const auto& channel : channels) {
                        if (!channel.contains("sampler") || !channel.contains("target")) continue;

                        const auto& target = channel["target"];
                        int nodeIdx = target.value("node", -1);
                        std::string path = target.value("path", "");

                        if (nodeIdx < 0 || path.empty()) continue;

                        int samplerIdx = channel["sampler"].get<int>();
                        if (samplerIdx < 0 || samplerIdx >= static_cast<int>(samplers.size())) continue;

                        const auto& samplerDef = samplers[samplerIdx];
                        int inputAccIdx = samplerDef["input"].get<int>();
                        int outputAccIdx = samplerDef["output"].get<int>();
                        std::string interpolation = samplerDef.value("interpolation", "LINEAR");

                        auto times = readFloats(inputAccIdx);
                        auto values = readFloats(outputAccIdx);

                        if (times.empty()) continue;

                        // CUBICSPLINE: strip in/out tangents, keep only the spline vertex (middle value)
                        if (interpolation == "CUBICSPLINE") {
                            int nFrames = static_cast<int>(times.size());
                            int totalComponents = static_cast<int>(values.size()) / (3 * nFrames);
                            if (totalComponents > 0) {
                                std::vector<float> stripped;
                                stripped.reserve(nFrames * totalComponents);
                                for (int f = 0; f < nFrames; ++f) {
                                    int base = f * 3 * totalComponents + totalComponents;// skip in-tangent
                                    for (int c = 0; c < totalComponents; ++c)
                                        stripped.push_back(values[base + c]);
                                }
                                values = std::move(stripped);
                            }
                            interpolation = "LINEAR";
                        }

                        Interpolation interp = Interpolation::Linear;
                        if (interpolation == "STEP") interp = Interpolation::Discrete;

                        // Resolve node name for track path
                        std::string nodeName;
                        auto nit = nodeObjects.find(nodeIdx);
                        if (nit != nodeObjects.end()) {
                            nodeName = nit->second->name;
                            if (nodeName.empty()) nodeName = "node_" + std::to_string(nodeIdx);
                        } else {
                            nodeName = "node_" + std::to_string(nodeIdx);
                        }

                        std::shared_ptr<KeyframeTrack> track;

                        if (path == "translation") {
                            track = std::make_shared<VectorKeyframeTrack>(
                                    nodeName + ".position", times, values, interp);
                        } else if (path == "rotation") {
                            track = std::make_shared<QuaternionKeyframeTrack>(
                                    nodeName + ".quaternion", times, values);
                        } else if (path == "scale") {
                            track = std::make_shared<VectorKeyframeTrack>(
                                    nodeName + ".scale", times, values, interp);
                        } else if (path == "weights") {
                            // Morph target weights: one NumberKeyframeTrack per morph target
                            // The values interleave weights for all morph targets per frame
                            int nFrames = static_cast<int>(times.size());
                            int nTargets = (nFrames > 0) ? static_cast<int>(values.size()) / nFrames : 0;
                            for (int t = 0; t < nTargets; ++t) {
                                std::vector<float> targetValues;
                                targetValues.reserve(nFrames);
                                for (int f = 0; f < nFrames; ++f)
                                    targetValues.push_back(values[f * nTargets + t]);
                                auto weightTrack = std::make_shared<NumberKeyframeTrack>(
                                        nodeName + ".morphTargetInfluences[" + std::to_string(t) + "]",
                                        times, targetValues, interp);
                                tracks.push_back(weightTrack);
                            }
                            continue;
                        }

                        if (track) tracks.push_back(track);
                    }

                    if (tracks.empty()) continue;

                    auto clip = std::make_shared<AnimationClip>(animName, -1.f, tracks);
                    clip->resetDuration();
                    clips.push_back(clip);
                }

                return clips;
            }

            void resolveVariants(GLTFResult& result) {
                if (variantNames.empty() || !result.scene) return;
                result.variants.names = variantNames;
                result.scene->traverse([&](Object3D& obj) {
                    auto* owm = dynamic_cast<ObjectWithMaterials*>(&obj);
                    if (!owm) return;
                    auto mi = obj.userData.find("__gltfMeshIdx");
                    auto pi = obj.userData.find("__gltfPrimIdx");
                    if (mi == obj.userData.end()) return;
                    int mIdx = std::any_cast<int>(mi->second);
                    int pIdx = std::any_cast<int>(pi->second);
                    obj.userData.erase("__gltfMeshIdx");
                    obj.userData.erase("__gltfPrimIdx");
                    result.variants.defaults[obj.uuid] = owm->material();
                    auto meshIt = primVariantData.find(mIdx);
                    if (meshIt == primVariantData.end()) return;
                    auto primIt = meshIt->second.find(pIdx);
                    if (primIt == meshIt->second.end()) return;
                    for (const auto& pvm : primIt->second) {
                        auto mat = loadMaterial(pvm.materialIdx);
                        for (int vi : pvm.variantIndices) {
                            if (vi < 0 || vi >= static_cast<int>(variantNames.size())) continue;
                            result.variants.table[variantNames[vi]].push_back({obj.uuid, mat});
                        }
                    }
                });
            }
        };

    }// anonymous namespace

    // ===========================================================================
    //  GLTFLoader public API
    // ===========================================================================

    std::optional<GLTFResult> GLTFLoader::load(const fs::path& path) {
        try {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open file: " + path.string());
            std::vector<uint8_t> data(std::istreambuf_iterator<char>(f), {});

            GLTFParser parser;
            parser.basePath = path.parent_path();
            parser.buffers = {};

            std::string ext = path.extension().string();
            // lowercase extension
            for (auto& c : ext) c = static_cast<char>(std::tolower(c));

            if (ext == ".glb") {
                return parser.parseGLB(data);
            }

            // .gltf — plain JSON
            std::string jsonText(data.begin(), data.end());
            return parser.parseGLTF(jsonText);
        } catch (const std::exception& e) {
            std::cerr << "[GLTFLoader] Error loading " << path << ": " << e.what() << "\n";
            return std::nullopt;
        }
    }

}// namespace threepp
