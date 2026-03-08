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

#include <threepp/loaders/ImageLoader.hpp>
#include <threepp/textures/DataTexture.hpp>

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

            // -----------------------------------------------------------------------
            //  Material
            // -----------------------------------------------------------------------

            std::shared_ptr<Material> loadMaterial(int matIdx) {
                auto it = materialCache.find(matIdx);
                if (it != materialCache.end()) return it->second;

                const auto& matDef = gltf["materials"][matIdx];
                auto mat = MeshStandardMaterial::create();
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
                        mat->map = loadTexture(ti);
                    }

                    // Metalness / roughness
                    mat->metalness = pbr.value("metallicFactor", 1.0f);
                    mat->roughness = pbr.value("roughnessFactor", 1.0f);

                    // Metallic-roughness texture (G=roughness, B=metalness per spec)
                    if (pbr.contains("metallicRoughnessTexture")) {
                        int ti = pbr["metallicRoughnessTexture"]["index"].get<int>();
                        auto tex = loadTexture(ti);
                        mat->metalnessMap = tex;
                        mat->roughnessMap = tex;
                    }
                }

                // Normal map
                if (matDef.contains("normalTexture")) {
                    int ti = matDef["normalTexture"]["index"].get<int>();
                    mat->normalMap = loadTexture(ti);
                    float scale = matDef["normalTexture"].value("scale", 1.0f);
                    mat->normalScale = Vector2{scale, scale};
                }

                // Occlusion map
                if (matDef.contains("occlusionTexture")) {
                    int ti = matDef["occlusionTexture"]["index"].get<int>();
                    mat->aoMap = loadTexture(ti);
                    mat->aoMapIntensity = matDef["occlusionTexture"].value("strength", 1.0f);
                }

                // Emissive
                if (matDef.contains("emissiveFactor")) {
                    auto e = matDef["emissiveFactor"].get<std::vector<float>>();
                    mat->emissive = Color(e[0], e[1], e[2]);
                }
                if (matDef.contains("emissiveTexture")) {
                    int ti = matDef["emissiveTexture"]["index"].get<int>();
                    mat->emissiveMap = loadTexture(ti);
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

                materialCache[matIdx] = mat;
                return mat;
            }

            // -----------------------------------------------------------------------
            //  Mesh
            // -----------------------------------------------------------------------

            std::shared_ptr<Object3D> loadMesh(int meshIdx) {
                const auto& meshDef = gltf["meshes"][meshIdx];
                const auto& primitives = meshDef["primitives"];

                // Multiple primitives → Group
                auto group = Group::create();
                group->name = meshDef.value("name", "");

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
                    addFloatAttr("COLOR_0", "color", 4);// may be RGB or RGBA; 4 is safe
                    addFloatAttr("TANGENT", "tangent", 4);

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

                    auto mesh = Mesh::create(geometry, mat);
                    group->add(mesh);
                }

                // If only one primitive, unwrap the group
                if (group->children.size() == 1) {
                    auto child = group->children[0];
                    child->name = group->name;
                    return child->clone();
                }
                return group;
            }

            // -----------------------------------------------------------------------
            //  Node / Scene hierarchy
            // -----------------------------------------------------------------------

            std::shared_ptr<Object3D> loadNode(int nodeIdx) {
                const auto& nodeDef = gltf["nodes"][nodeIdx];
                std::shared_ptr<Object3D> obj;

                if (nodeDef.contains("mesh")) {
                    obj = loadMesh(nodeDef["mesh"].get<int>());
                } else {
                    obj = Group::create();
                }

                obj->name = nodeDef.value("name", "");

                // Transform — matrix takes priority over TRS
                if (nodeDef.contains("matrix")) {
                    auto m = nodeDef["matrix"].get<std::vector<float>>();
                    Matrix4 mat4;
                    mat4.set(m[0], m[4], m[8], m[12],
                             m[1], m[5], m[9], m[13],
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

                // Recurse children
                if (nodeDef.contains("children")) {
                    for (int childIdx : nodeDef["children"].get<std::vector<int>>()) {
                        obj->add(loadNode(childIdx));
                    }
                }

                return obj;
            }

            std::shared_ptr<Group> loadScene(int sceneIdx) {
                const auto& sceneDef = gltf["scenes"][sceneIdx];
                auto root = Group::create();
                root->name = sceneDef.value("name", "Scene");

                if (sceneDef.contains("nodes")) {
                    for (int nodeIdx : sceneDef["nodes"].get<std::vector<int>>()) {
                        root->add(loadNode(nodeIdx));
                    }
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

                return result;
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
