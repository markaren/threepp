
#include "threepp/loaders/ColladaLoader.hpp"

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/tracks/QuaternionKeyframeTrack.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/constants.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "pugixml.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace threepp;

namespace {

    // Bulk parsers for whitespace-separated numeric XML arrays. COLLADA files
    // carry arrays with hundreds of thousands of entries, so these avoid the
    // istringstream-per-array overhead. reserveHint typically comes from the
    // element's count attribute; 0 means unknown.

    std::vector<float> parseFloatArray(const std::string& text, size_t reserveHint = 0) {
        std::vector<float> result;
        result.reserve(reserveHint);
        const char* p = text.c_str();
        const char* const end = p + text.size();
        while (p != end) {
            if (std::isspace(static_cast<unsigned char>(*p))) {
                ++p;
                continue;
            }
#if defined(__cpp_lib_to_chars)
            float val{};
            if (const auto [next, ec] = std::from_chars(p, end, val); ec == std::errc{} && next != p) {
                result.push_back(val);
                p = next;
                continue;
            }
#else
            // libc++ without float from_chars: strtof stops at the string's
            // terminating null, so reading from within c_str() is safe.
            char* next = nullptr;
            const float val = std::strtof(p, &next);
            if (next != p) {
                result.push_back(val);
                p = next;
                continue;
            }
#endif
            // skip malformed token
            while (p != end && !std::isspace(static_cast<unsigned char>(*p))) ++p;
        }
        return result;
    }

    std::vector<int> parseIntArray(const std::string& text, size_t reserveHint = 0) {
        std::vector<int> result;
        result.reserve(reserveHint);
        const char* p = text.c_str();
        const char* const end = p + text.size();
        while (p != end) {
            if (std::isspace(static_cast<unsigned char>(*p))) {
                ++p;
                continue;
            }
            int val{};
            if (const auto [next, ec] = std::from_chars(p, end, val); ec == std::errc{} && next != p) {
                result.push_back(val);
                p = next;
                continue;
            }
            while (p != end && !std::isspace(static_cast<unsigned char>(*p))) ++p;
        }
        return result;
    }

    std::string stripHash(const std::string& s) {
        if (!s.empty() && s[0] == '#') return s.substr(1);
        return s;
    }

    struct Source {
        std::vector<float> data;
        int stride{3};
        int offset{0};// accessor offset into data
    };

    struct Input {
        std::string semantic;
        std::string source;// id without '#'
        int offset{0};
        int set{0};
    };

    struct Primitive {
        std::string type;// "triangles" or "polylist"
        std::string material;
        int count{0};
        std::vector<Input> inputs;
        std::vector<int> p;
        std::vector<int> vcount;// for polylist
        int maxOffset{0};
    };

    struct GeometryData {
        std::string id;
        std::string name;// name attribute; falls back to id when absent
        std::unordered_map<std::string, Source> sources;
        // <vertices id="..."> maps to a source id (POSITION semantic)
        std::unordered_map<std::string, std::string> verticesMap;
        std::vector<Primitive> primitives;
    };

    struct EffectData {
        Color diffuse{1.f, 1.f, 1.f};
        Color ambient{0.f, 0.f, 0.f};
        Color specular{1.f, 1.f, 1.f};
        float shininess{30.f};
        float transparency{1.f};
        std::string diffuseTextureId;// image id
        TextureWrapping wrapS{TextureWrapping::Repeat};
        TextureWrapping wrapT{TextureWrapping::Repeat};
    };

    TextureWrapping parseWrapMode(const std::string& value) {
        if (value == "MIRROR") return TextureWrapping::MirroredRepeat;
        if (value == "CLAMP" || value == "BORDER" || value == "NONE") return TextureWrapping::ClampToEdge;
        return TextureWrapping::Repeat;// WRAP or unspecified
    }

    Color parseColorNode(const pugi::xml_node& node) {
        const std::string text = node.child_value();
        auto parts = utils::split(text, ' ');
        if (parts.size() >= 3) {
            return Color(utils::parseFloat(parts[0]),
                         utils::parseFloat(parts[1]),
                         utils::parseFloat(parts[2]));
        }
        return Color(1.f, 1.f, 1.f);
    }

    void parseSources(const pugi::xml_node& meshNode, GeometryData& geo) {
        for (const auto& source : meshNode.children("source")) {
            const std::string id = source.attribute("id").value();
            const auto floatArray = source.child("float_array");
            if (!floatArray) continue;

            const auto technique = source.child("technique_common");
            const auto accessor = technique ? technique.child("accessor") : pugi::xml_node{};

            Source src;
            if (accessor) {
                src.stride = accessor.attribute("stride").as_int(3);
                src.offset = accessor.attribute("offset").as_int(0);
            }
            src.data = parseFloatArray(floatArray.child_value(), floatArray.attribute("count").as_uint(0));
            geo.sources[id] = std::move(src);
        }
    }

    void parseVertices(const pugi::xml_node& meshNode, GeometryData& geo) {
        for (const auto& verts : meshNode.children("vertices")) {
            const std::string id = verts.attribute("id").value();
            for (const auto& input : verts.children("input")) {
                const std::string semantic = input.attribute("semantic").value();
                if (semantic == "POSITION") {
                    geo.verticesMap[id] = stripHash(input.attribute("source").value());
                }
            }
        }
    }

    std::vector<Input> parseInputs(const pugi::xml_node& prim) {
        std::vector<Input> inputs;
        for (const auto& inp : prim.children("input")) {
            Input i;
            i.semantic = inp.attribute("semantic").value();
            i.source = stripHash(inp.attribute("source").value());
            i.offset = inp.attribute("offset").as_int(0);
            i.set = inp.attribute("set").as_int(0);
            inputs.push_back(i);
        }
        return inputs;
    }

    int maxOffset(const std::vector<Input>& inputs) {
        int m = 0;
        for (const auto& i : inputs) m = std::max(m, i.offset);
        return m;
    }

    void parsePrimitives(const pugi::xml_node& meshNode, GeometryData& geo) {
        // triangles
        for (const auto& tri : meshNode.children("triangles")) {
            Primitive p;
            p.type = "triangles";
            p.material = tri.attribute("material").value();
            p.count = tri.attribute("count").as_int(0);
            p.inputs = parseInputs(tri);
            p.maxOffset = maxOffset(p.inputs);
            p.p = parseIntArray(tri.child("p").child_value(),
                                static_cast<size_t>(p.count) * 3 * (p.maxOffset + 1));
            geo.primitives.push_back(std::move(p));
        }

        // polylist
        for (const auto& poly : meshNode.children("polylist")) {
            Primitive p;
            p.type = "polylist";
            p.material = poly.attribute("material").value();
            p.count = poly.attribute("count").as_int(0);
            p.inputs = parseInputs(poly);
            p.maxOffset = maxOffset(p.inputs);
            p.vcount = parseIntArray(poly.child("vcount").child_value(), p.count);
            p.p = parseIntArray(poly.child("p").child_value(),
                                static_cast<size_t>(p.count) * 4 * (p.maxOffset + 1));
            geo.primitives.push_back(std::move(p));
        }

        // lines
        // (not handled here — only triangles and polylist)
    }

    GeometryData parseGeometry(const pugi::xml_node& geomNode) {
        GeometryData geo;
        geo.id = geomNode.attribute("id").value();
        geo.name = geomNode.attribute("name").value();
        if (geo.name.empty()) geo.name = geo.id;

        const auto meshNode = geomNode.child("mesh");
        if (!meshNode) return geo;

        parseSources(meshNode, geo);
        parseVertices(meshNode, geo);
        parsePrimitives(meshNode, geo);

        return geo;
    }

    // Resolve the actual source id for a given input, accounting for <vertices> indirection
    const Source* resolveSource(const Input& input, const GeometryData& geo) {
        std::string sourceId = input.source;

        // If this is a VERTEX input, look up via vertices map
        if (input.semantic == "VERTEX") {
            auto it = geo.verticesMap.find(sourceId);
            if (it != geo.verticesMap.end()) sourceId = it->second;
        }

        auto it = geo.sources.find(sourceId);
        if (it == geo.sources.end()) return nullptr;
        return &it->second;
    }

    struct SkinData {
        Matrix4 bindShapeMatrix;// identity default
        std::vector<std::string> jointNames;
        std::vector<Matrix4> invBindMatrices;
        std::vector<float> weightValues;
        std::vector<int> vcount;
        std::vector<int> v;
        int inputStride{2};
        int jointOffset{0};
        int weightOffset{1};

        // Precomputed top-4 influences per original vertex, packed as floats for BufferAttribute.
        std::vector<std::array<float, 4>> jointIndices;
        std::vector<std::array<float, 4>> jointWeights;
    };

    // Compute the top-4 joint influences per original vertex and normalize weights.
    void buildSkinInfluences(SkinData& sd) {
        const size_t numVerts = sd.vcount.size();
        sd.jointIndices.assign(numVerts, {0.f, 0.f, 0.f, 0.f});
        sd.jointWeights.assign(numVerts, {0.f, 0.f, 0.f, 0.f});

        size_t vCursor = 0;
        for (size_t vi = 0; vi < numVerts; ++vi) {
            const int n = sd.vcount[vi];

            // Keep the 4 strongest influences via sorted insertion — no per-vertex
            // allocation or full sort.
            std::array<int, 4> ji{0, 0, 0, 0};
            std::array<float, 4> jw{0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < n; ++k) {
                const size_t base = (vCursor + k) * sd.inputStride;
                if (base + sd.inputStride > sd.v.size()) break;
                const int jointIdx = sd.v[base + sd.jointOffset];
                const int weightIdx = sd.v[base + sd.weightOffset];
                const float w = (weightIdx >= 0 && weightIdx < static_cast<int>(sd.weightValues.size()))
                                        ? sd.weightValues[weightIdx]
                                        : 0.f;
                for (int s = 0; s < 4; ++s) {
                    if (w > jw[s]) {
                        for (int m = 3; m > s; --m) {
                            jw[m] = jw[m - 1];
                            ji[m] = ji[m - 1];
                        }
                        jw[s] = w;
                        ji[s] = jointIdx;
                        break;
                    }
                }
            }
            vCursor += n;

            const float sum = jw[0] + jw[1] + jw[2] + jw[3];
            const float invSum = sum > 0.f ? 1.f / sum : 0.f;
            for (int k = 0; k < 4; ++k) {
                sd.jointIndices[vi][k] = static_cast<float>(ji[k]);
                sd.jointWeights[vi][k] = jw[k] * invSum;
            }
        }
    }

    // Key of one output vertex: the (position, normal, uv) source-index tuple a
    // face corner references. Corners sharing a tuple share one indexed vertex.
    struct VertexKey {
        int pos{-1};
        int normal{-1};
        int uv{-1};

        bool operator==(const VertexKey&) const = default;
    };

    struct VertexKeyHash {
        size_t operator()(const VertexKey& k) const {
            size_t h = std::hash<int>{}(k.pos);
            h ^= std::hash<int>{}(k.normal) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(k.uv) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::shared_ptr<BufferGeometry> buildGeometry(const GeometryData& geo, const Primitive& prim,
                                                  const SkinData* skin = nullptr) {
        // Find inputs
        const Input* posInput = nullptr;
        const Input* normalInput = nullptr;
        const Input* uvInput = nullptr;
        for (const auto& inp : prim.inputs) {
            if (inp.semantic == "VERTEX") posInput = &inp;
            else if (inp.semantic == "NORMAL")
                normalInput = &inp;
            else if (inp.semantic == "TEXCOORD" && inp.set == 0)
                uvInput = &inp;
        }

        const Source* posSource = posInput ? resolveSource(*posInput, geo) : nullptr;
        const Source* normalSource = normalInput ? resolveSource(*normalInput, geo) : nullptr;
        const Source* uvSource = uvInput ? resolveSource(*uvInput, geo) : nullptr;

        if (!posSource) return nullptr;

        const int stride = prim.maxOffset + 1;
        // Corners available in the (possibly malformed/truncated) <p> stream.
        const size_t maxCorners = stride > 0 ? prim.p.size() / stride : 0;

        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> uvs;
        std::vector<float> skinIndices;
        std::vector<float> skinWeights;
        std::vector<unsigned int> index;
        std::unordered_map<VertexKey, unsigned int, VertexKeyHash> remap;
        index.reserve(maxCorners);
        remap.reserve(maxCorners);

        // Copy `n` floats from an accessor position, zero-filling when the source
        // index is out of range (malformed input must not read out of bounds).
        auto fetch = [](const Source& src, int idx, float* out, int n) {
            const size_t base = static_cast<size_t>(src.offset) + static_cast<size_t>(idx) * src.stride;
            if (idx < 0 || base + n > src.data.size()) {
                std::fill_n(out, n, 0.f);
                return;
            }
            std::copy_n(src.data.begin() + base, n, out);
        };

        auto addCorner = [&](size_t corner) {
            const size_t pBase = corner * stride;

            VertexKey key;
            key.pos = prim.p[pBase + posInput->offset];
            if (normalSource && normalInput) key.normal = prim.p[pBase + normalInput->offset];
            if (uvSource && uvInput) key.uv = prim.p[pBase + uvInput->offset];

            const auto [it, inserted] = remap.try_emplace(key, static_cast<unsigned int>(positions.size() / 3));
            if (inserted) {
                float buf[3];
                fetch(*posSource, key.pos, buf, 3);
                positions.insert(positions.end(), buf, buf + 3);
                if (normalSource && normalInput) {
                    fetch(*normalSource, key.normal, buf, 3);
                    normals.insert(normals.end(), buf, buf + 3);
                }
                if (uvSource && uvInput) {
                    fetch(*uvSource, key.uv, buf, 2);
                    uvs.insert(uvs.end(), buf, buf + 2);
                }
                if (skin) {
                    if (key.pos >= 0 && key.pos < static_cast<int>(skin->jointIndices.size())) {
                        const auto& ji = skin->jointIndices[key.pos];
                        const auto& jw = skin->jointWeights[key.pos];
                        skinIndices.insert(skinIndices.end(), ji.begin(), ji.end());
                        skinWeights.insert(skinWeights.end(), jw.begin(), jw.end());
                    } else {
                        skinIndices.insert(skinIndices.end(), {0.f, 0.f, 0.f, 0.f});
                        skinWeights.insert(skinWeights.end(), {0.f, 0.f, 0.f, 0.f});
                    }
                }
            }
            index.push_back(it->second);
        };

        if (prim.type == "triangles") {
            size_t vertexCount = std::min(static_cast<size_t>(prim.count) * 3, maxCorners);
            vertexCount -= vertexCount % 3;// only whole triangles
            for (size_t i = 0; i < vertexCount; i++) {
                addCorner(i);
            }
        } else if (prim.type == "polylist") {
            size_t vi = 0;// corner index into p
            for (int polyIdx = 0; polyIdx < prim.count; polyIdx++) {
                const int vc = prim.vcount.empty() ? 3
                                                   : (polyIdx < static_cast<int>(prim.vcount.size()) ? prim.vcount[polyIdx] : 0);
                if (vc <= 0 || vi + vc > maxCorners) break;// truncated/malformed
                // Fan triangulation from vertex 0
                for (int t = 1; t < vc - 1; t++) {
                    addCorner(vi);
                    addCorner(vi + t);
                    addCorner(vi + t + 1);
                }
                vi += vc;
            }
        }

        if (positions.empty()) return nullptr;

        // Bake bind-shape matrix into positions/normals so bindMatrix on the
        // SkinnedMesh can be identity (matches three.js GLTFLoader convention)
        // and avoids scaling the skinned mesh by bindShape at runtime.
        if (skin) {
            const Matrix4& bsm = skin->bindShapeMatrix;
            Matrix3 nmat;
            nmat.getNormalMatrix(bsm);
            for (size_t i = 0; i + 2 < positions.size(); i += 3) {
                Vector3 p{positions[i], positions[i + 1], positions[i + 2]};
                p.applyMatrix4(bsm);
                positions[i] = p.x;
                positions[i + 1] = p.y;
                positions[i + 2] = p.z;
            }
            for (size_t i = 0; i + 2 < normals.size(); i += 3) {
                Vector3 n{normals[i], normals[i + 1], normals[i + 2]};
                n.applyMatrix3(nmat).normalize();
                normals[i] = n.x;
                normals[i + 1] = n.y;
                normals[i + 2] = n.z;
            }
        }

        auto geometry = std::make_shared<BufferGeometry>();
        geometry->setIndex(std::move(index));
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        if (!normals.empty()) {
            geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
        }
        if (!uvs.empty()) {
            geometry->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
        }
        if (!skinIndices.empty()) {
            geometry->setAttribute("skinIndex", FloatBufferAttribute::create(skinIndices, 4));
            geometry->setAttribute("skinWeight", FloatBufferAttribute::create(skinWeights, 4));
        }

        return geometry;
    }

    EffectData parseEffect(const pugi::xml_node& effectNode) {
        EffectData effect;

        // Look for phong/lambert/blinn shading model
        auto profile = effectNode.child("profile_COMMON");
        if (!profile) return effect;

        // Build newparam maps: sampler sid -> surface sid, surface sid -> image id
        std::unordered_map<std::string, std::string> samplerToSurface;
        std::unordered_map<std::string, std::string> surfaceToImage;
        // sampler sid -> {surface sid, wrap_s, wrap_t}
        struct SamplerInfo {
            std::string surface;
            std::string wrapS, wrapT;
        };
        std::unordered_map<std::string, SamplerInfo> samplers;

        for (const auto& param : profile.children("newparam")) {
            const std::string sid = param.attribute("sid").value();
            if (const auto sampler = param.child("sampler2D")) {
                SamplerInfo info;
                info.surface = sampler.child_value("source");
                info.wrapS = sampler.child_value("wrap_s");
                info.wrapT = sampler.child_value("wrap_t");
                samplers[sid] = std::move(info);
                samplerToSurface[sid] = samplers[sid].surface;
            } else if (const auto surface = param.child("surface")) {
                surfaceToImage[sid] = surface.child_value("init_from");
            }
        }

        // Helper to resolve a texture attribute value to an image id, and capture wrap modes
        auto resolveTexture = [&](const std::string& texAttr, EffectData& eff) -> std::string {
            auto sampIt = samplers.find(texAttr);
            if (sampIt != samplers.end()) {
                if (!sampIt->second.wrapS.empty()) eff.wrapS = parseWrapMode(sampIt->second.wrapS);
                if (!sampIt->second.wrapT.empty()) eff.wrapT = parseWrapMode(sampIt->second.wrapT);
                auto imgIt = surfaceToImage.find(sampIt->second.surface);
                if (imgIt != surfaceToImage.end()) return imgIt->second;
                return "";
            }
            return texAttr;// might already be an image id
        };

        // Check for technique
        auto technique = profile.child("technique");
        if (!technique) return effect;

        pugi::xml_node shading;
        for (const char* model : {"phong", "lambert", "blinn", "constant"}) {
            shading = technique.child(model);
            if (shading) break;
        }
        if (!shading) return effect;

        // diffuse
        auto diffuseNode = shading.child("diffuse");
        if (diffuseNode) {
            auto colorNode = diffuseNode.child("color");
            if (colorNode) effect.diffuse = parseColorNode(colorNode);
            auto texNode = diffuseNode.child("texture");
            if (texNode) effect.diffuseTextureId = resolveTexture(texNode.attribute("texture").value(), effect);
        }

        // ambient
        auto ambientNode = shading.child("ambient");
        if (ambientNode) {
            auto colorNode = ambientNode.child("color");
            if (colorNode) effect.ambient = parseColorNode(colorNode);
        }

        // specular
        auto specularNode = shading.child("specular");
        if (specularNode) {
            auto colorNode = specularNode.child("color");
            if (colorNode) effect.specular = parseColorNode(colorNode);
        }

        // shininess
        auto shininessNode = shading.child("shininess");
        if (shininessNode) {
            auto floatNode = shininessNode.child("float");
            if (floatNode) effect.shininess = utils::parseFloat(floatNode.child_value());
        }

        // transparency
        auto transparencyNode = shading.child("transparency");
        if (transparencyNode) {
            auto floatNode = transparencyNode.child("float");
            if (floatNode) effect.transparency = utils::parseFloat(floatNode.child_value());
        }

        return effect;
    }

    std::shared_ptr<MeshPhongMaterial> buildMaterial(const EffectData& effect) {
        auto mat = MeshPhongMaterial::create();
        mat->color.copy(effect.diffuse);
        mat->specular.copy(effect.specular);
        mat->shininess = effect.shininess;
        if (effect.transparency < 1.f) {
            mat->transparent = true;
            mat->opacity = effect.transparency;
        }
        return mat;
    }

    // --------- Animations ---------

    struct AnimSource {
        std::vector<float> floats;
        std::vector<std::string> names;
        int stride{1};
        std::string paramType;// "float", "float4x4", "name", ...
    };

    struct AnimSampler {
        std::string inputId;
        std::string outputId;
        std::string interpolationId;
    };

    struct AnimChannel {
        std::string samplerId;
        std::string target;// "nodeId/sid[.component]"
    };

    struct AnimationEntry {
        std::unordered_map<std::string, AnimSource> sources;
        std::unordered_map<std::string, AnimSampler> samplers;
        std::vector<AnimChannel> channels;
    };

    std::vector<std::string> parseNameArray(const std::string& text) {
        std::vector<std::string> result;
        std::istringstream ss(text);
        std::string val;
        while (ss >> val) result.push_back(val);
        return result;
    }

    // Parse all <source> nodes inside an <animation>
    void parseAnimSources(const pugi::xml_node& animNode, std::unordered_map<std::string, AnimSource>& sources) {
        for (const auto& src : animNode.children("source")) {
            const std::string id = src.attribute("id").value();
            AnimSource s;

            const auto technique = src.child("technique_common");
            if (const auto accessor = technique ? technique.child("accessor") : pugi::xml_node{}) {
                s.stride = accessor.attribute("stride").as_int(1);
                if (const auto param = accessor.child("param")) {
                    s.paramType = param.attribute("type").value();
                }
            }

            if (const auto fa = src.child("float_array")) {
                s.floats = parseFloatArray(fa.child_value(), fa.attribute("count").as_uint(0));
            } else if (const auto na = src.child("Name_array")) {
                s.names = parseNameArray(na.child_value());
            }

            sources[id] = std::move(s);
        }
    }

    void parseAnimSamplers(const pugi::xml_node& animNode, std::unordered_map<std::string, AnimSampler>& samplers) {
        for (const auto& sampNode : animNode.children("sampler")) {
            const std::string id = sampNode.attribute("id").value();
            AnimSampler sampler;
            for (const auto& inp : sampNode.children("input")) {
                const std::string sem = inp.attribute("semantic").value();
                const std::string src = stripHash(inp.attribute("source").value());
                if (sem == "INPUT") sampler.inputId = src;
                else if (sem == "OUTPUT")
                    sampler.outputId = src;
                else if (sem == "INTERPOLATION")
                    sampler.interpolationId = src;
            }
            samplers[id] = sampler;
        }
    }

    void parseAnimChannels(const pugi::xml_node& animNode, std::vector<AnimChannel>& channels) {
        for (const auto& chanNode : animNode.children("channel")) {
            AnimChannel c;
            c.samplerId = stripHash(chanNode.attribute("source").value());
            c.target = chanNode.attribute("target").value();
            channels.push_back(std::move(c));
        }
    }

    // Recursively collect animation leaves: a Collada <animation> can nest sub-animations
    // that reuse the parent's sources/samplers/channels. We flatten by gathering all
    // sources/samplers/channels encountered under one top-level animation.
    void collectAnim(const pugi::xml_node& animNode, AnimationEntry& entry) {
        parseAnimSources(animNode, entry.sources);
        parseAnimSamplers(animNode, entry.samplers);
        parseAnimChannels(animNode, entry.channels);
        for (const auto& child : animNode.children("animation")) {
            collectAnim(child, entry);
        }
    }

    // Split a channel target like "nodeId/sid.component" into its parts
    struct ParsedTarget {
        std::string nodeId;
        std::string sid;
        std::string component;// may be empty
    };

    ParsedTarget parseTarget(const std::string& target) {
        ParsedTarget t;
        const auto slash = target.find('/');
        if (slash == std::string::npos) {
            t.nodeId = target;
            return t;
        }
        t.nodeId = target.substr(0, slash);
        std::string rest = target.substr(slash + 1);

        // Component can be separated by '.' (e.g. "location.X") or '(' (e.g. "transform(3)(0)")
        const auto dot = rest.find_first_of(".(");
        if (dot == std::string::npos) {
            t.sid = rest;
        } else {
            t.sid = rest.substr(0, dot);
            t.component = rest.substr(dot);
        }
        return t;
    }

    // Apply all transform children in order and return the combined Matrix4
    Matrix4 buildNodeTransform(const pugi::xml_node& node) {
        Matrix4 result;

        for (const auto& child : node.children()) {
            const std::string name = child.name();

            if (name == "matrix") {
                auto vals = parseFloatArray(child.child_value());
                if (vals.size() >= 16) {
                    Matrix4 m;
                    m.set(vals[0], vals[1], vals[2], vals[3],
                          vals[4], vals[5], vals[6], vals[7],
                          vals[8], vals[9], vals[10], vals[11],
                          vals[12], vals[13], vals[14], vals[15]);
                    result.multiply(m);
                }
            } else if (name == "translate") {
                auto vals = parseFloatArray(child.child_value());
                if (vals.size() >= 3) {
                    Matrix4 m;
                    m.makeTranslation(vals[0], vals[1], vals[2]);
                    result.multiply(m);
                }
            } else if (name == "rotate") {
                auto vals = parseFloatArray(child.child_value());
                if (vals.size() >= 4) {
                    Vector3 axis(vals[0], vals[1], vals[2]);
                    float angle = vals[3] * math::DEG2RAD;
                    Matrix4 m;
                    Quaternion q;
                    q.setFromAxisAngle(axis, angle);
                    m.makeRotationFromQuaternion(q);
                    result.multiply(m);
                }
            } else if (name == "scale") {
                auto vals = parseFloatArray(child.child_value());
                if (vals.size() >= 3) {
                    Matrix4 m;
                    m.makeScale(vals[0], vals[1], vals[2]);
                    result.multiply(m);
                }
            }
        }

        return result;
    }

}// namespace

struct ColladaLoader::Impl {

    bool ignoreUpDirection = false;

    std::shared_ptr<Group> load(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            std::cerr << "[ColladaLoader] No such file: '" << std::filesystem::absolute(path).string() << "'!" << std::endl;
            return nullptr;
        }

        pugi::xml_document doc;
        const auto result = doc.load_file(path.string().c_str());
        if (!result) {
            std::cerr << "[ColladaLoader] Failed to parse XML: " << result.description() << std::endl;
            return nullptr;
        }

        const auto root = doc.child("COLLADA");
        if (!root) {
            std::cerr << "[ColladaLoader] Not a valid COLLADA file." << std::endl;
            return nullptr;
        }

        // Parse up-axis (default Y_UP per Collada spec)
        std::string upAxis = "Y_UP";
        if (const auto asset = root.child("asset")) {
            const std::string ua = asset.child_value("up_axis");
            if (!ua.empty()) upAxis = ua;
        }

        // Parse geometries
        std::unordered_map<std::string, GeometryData> geometries;
        for (const auto& geom : root.child("library_geometries").children("geometry")) {
            const std::string id = geom.attribute("id").value();
            geometries[id] = parseGeometry(geom);
        }

        // Parse controllers: map controller id -> geometry id and skin data.
        std::unordered_map<std::string, std::string> controllerToGeometry;
        std::unordered_map<std::string, SkinData> skinsByController;
        for (const auto& ctrl : root.child("library_controllers").children("controller")) {
            const std::string id = ctrl.attribute("id").value();
            const auto skin = ctrl.child("skin");
            if (!skin) continue;

            controllerToGeometry[id] = stripHash(skin.attribute("source").value());

            SkinData sd;
            if (const auto bsm = skin.child("bind_shape_matrix")) {
                auto vals = parseFloatArray(bsm.child_value());
                if (vals.size() >= 16) {
                    sd.bindShapeMatrix.set(vals[0], vals[1], vals[2], vals[3],
                                           vals[4], vals[5], vals[6], vals[7],
                                           vals[8], vals[9], vals[10], vals[11],
                                           vals[12], vals[13], vals[14], vals[15]);
                }
            }

            // Gather skin <source> elements.
            std::unordered_map<std::string, std::vector<float>> floatSources;
            std::unordered_map<std::string, std::vector<std::string>> nameSources;
            for (const auto& src : skin.children("source")) {
                const std::string sid = src.attribute("id").value();
                if (const auto fa = src.child("float_array")) {
                    floatSources[sid] = parseFloatArray(fa.child_value(), fa.attribute("count").as_uint(0));
                } else if (const auto na = src.child("Name_array")) {
                    nameSources[sid] = parseNameArray(na.child_value());
                }
            }

            // <joints> -> JOINT + INV_BIND_MATRIX source ids
            std::string jointsSrcId, invBindSrcId;
            if (const auto joints = skin.child("joints")) {
                for (const auto& inp : joints.children("input")) {
                    const std::string sem = inp.attribute("semantic").value();
                    const std::string src = stripHash(inp.attribute("source").value());
                    if (sem == "JOINT") jointsSrcId = src;
                    else if (sem == "INV_BIND_MATRIX")
                        invBindSrcId = src;
                }
            }
            if (const auto it = nameSources.find(jointsSrcId); it != nameSources.end()) {
                sd.jointNames = it->second;
            }
            if (const auto it = floatSources.find(invBindSrcId); it != floatSources.end()) {
                const auto& mats = it->second;
                for (size_t i = 0; i + 16 <= mats.size(); i += 16) {
                    Matrix4 m;
                    m.set(mats[i + 0], mats[i + 1], mats[i + 2], mats[i + 3],
                          mats[i + 4], mats[i + 5], mats[i + 6], mats[i + 7],
                          mats[i + 8], mats[i + 9], mats[i + 10], mats[i + 11],
                          mats[i + 12], mats[i + 13], mats[i + 14], mats[i + 15]);
                    sd.invBindMatrices.emplace_back(m);
                }
            }

            // <vertex_weights> -> per-vertex (joint, weight) pairs
            if (const auto vw = skin.child("vertex_weights")) {
                std::string weightSrcId;
                int maxOff = 0;
                for (const auto& inp : vw.children("input")) {
                    const std::string sem = inp.attribute("semantic").value();
                    const int off = inp.attribute("offset").as_int(0);
                    maxOff = std::max(maxOff, off);
                    if (sem == "JOINT") sd.jointOffset = off;
                    else if (sem == "WEIGHT") {
                        sd.weightOffset = off;
                        weightSrcId = stripHash(inp.attribute("source").value());
                    }
                }
                sd.inputStride = maxOff + 1;
                if (const auto it = floatSources.find(weightSrcId); it != floatSources.end()) {
                    sd.weightValues = it->second;
                }
                if (const auto vc = vw.child("vcount")) sd.vcount = parseIntArray(vc.child_value(), vw.attribute("count").as_uint(0));
                if (const auto vv = vw.child("v")) sd.v = parseIntArray(vv.child_value());
            }

            if (!sd.vcount.empty()) {
                buildSkinInfluences(sd);
                skinsByController[id] = std::move(sd);
            }
        }

        // Parse effects
        std::unordered_map<std::string, EffectData> effects;
        for (const auto& eff : root.child("library_effects").children("effect")) {
            const std::string id = eff.attribute("id").value();
            effects[id] = parseEffect(eff);
        }

        // Parse materials: material id -> effect id
        std::unordered_map<std::string, std::string> materialToEffect;
        for (const auto& mat : root.child("library_materials").children("material")) {
            const std::string id = mat.attribute("id").value();
            const auto instEffect = mat.child("instance_effect");
            if (instEffect) {
                materialToEffect[id] = stripHash(instEffect.attribute("url").value());
            }
        }

        // Parse images: image id -> filename
        const auto baseDir = path.parent_path();
        std::unordered_map<std::string, std::filesystem::path> images;
        for (const auto& img : root.child("library_images").children("image")) {
            const std::string id = img.attribute("id").value();
            const std::string filename = img.child_value("init_from");
            if (!filename.empty()) {
                images[id] = baseDir / filename;
            }
        }

        TextureLoader texLoader;

        // Helper: resolve effect + texture -> Material. Instances referencing the
        // same material id share one Material (textures are deduped by texLoader).
        std::unordered_map<std::string, std::shared_ptr<Material>> materialCache;
        auto loadMaterial = [&](const std::string& matId) -> std::shared_ptr<Material> {
            if (const auto it = materialCache.find(matId); it != materialCache.end()) {
                return it->second;
            }
            std::shared_ptr<Material> mat;
            const std::string effectId = materialToEffect.contains(matId) ? materialToEffect.at(matId) : "";
            if (effectId.empty() || !effects.contains(effectId)) {
                mat = MeshPhongMaterial::create();
            } else {
                const EffectData& effect = effects.at(effectId);
                auto phong = buildMaterial(effect);
                if (!effect.diffuseTextureId.empty() && images.contains(effect.diffuseTextureId)) {
                    auto tex = texLoader.load(images.at(effect.diffuseTextureId));
                    if (tex) {
                        tex->wrapS = effect.wrapS;
                        tex->wrapT = effect.wrapT;
                        tex->needsUpdate();
                        phong->map = tex;
                    }
                }
                mat = phong;
            }
            materialCache[matId] = mat;
            return mat;
        };

        // Repeated <instance_geometry>/<instance_controller> of the same source
        // share one BufferGeometry instead of re-expanding the index streams.
        std::unordered_map<std::string, std::shared_ptr<BufferGeometry>> geometryCache;

        // Build scene
        auto sceneGroup = Group::create();

        // nodesById maps each <node id="..."> to its created Object3D (either a
        // Group or a Bone for type="JOINT" nodes) so that animation channel targets
        // can resolve to a track name the AnimationMixer can bind.
        std::unordered_map<std::string, std::shared_ptr<Object3D>> nodesById;

        const auto visualScenes = root.child("library_visual_scenes");
        const auto& scene = root.child("scene");
        std::string sceneId;
        if (scene) {
            const auto instScene = scene.child("instance_visual_scene");
            if (instScene) sceneId = stripHash(instScene.attribute("url").value());
        }

        pugi::xml_node visualScene;
        for (const auto& vs : visualScenes.children("visual_scene")) {
            if (sceneId.empty() || std::string(vs.attribute("id").value()) == sceneId) {
                visualScene = vs;
                break;
            }
        }

        if (!visualScene) {
            std::cerr << "[ColladaLoader] No visual scene found." << std::endl;
            return sceneGroup;
        }

        // SkinnedMeshes can't bind their skeleton until every joint node has been
        // parsed (a skin may reference bones that live outside the current node's
        // subtree). Queue them here and resolve after the visual-scene walk.
        struct PendingSkin {
            std::shared_ptr<SkinnedMesh> mesh;
            std::string controllerId;
        };
        std::vector<PendingSkin> pendingSkins;

        // Helper: parse a node recursively
        std::function<void(const pugi::xml_node&, Object3D&)> parseNode;
        parseNode = [&](const pugi::xml_node& nodeXml, Object3D& parent) {
            const std::string nodeId = nodeXml.attribute("id").value();
            const std::string nodeSid = nodeXml.attribute("sid").value();
            const std::string nodeType = nodeXml.attribute("type").value();

            // JOINT nodes become Bones so that future SkinnedMesh support can
            // resolve them via Skeleton::getBoneByName. Other nodes are Groups.
            std::shared_ptr<Object3D> group = (nodeType == "JOINT")
                                                      ? std::static_pointer_cast<Object3D>(Bone::create())
                                                      : std::static_pointer_cast<Object3D>(Group::create());

            // Use id/sid (underscore form) rather than the `name` attribute for the
            // Object3D name. The `name` may contain ':' (e.g. "mixamorig:Hips") that
            // mismatches the joint names used by <skin> Name_array and animation
            // channel targets, which reference the id/sid form.
            std::string nodeName = nodeId.empty() ? nodeSid : nodeId;
            if (nodeName.empty()) nodeName = nodeXml.attribute("name").value();
            group->name = nodeName;
            if (!nodeId.empty()) nodesById[nodeId] = group;
            if (!nodeSid.empty() && nodeSid != nodeId) nodesById[nodeSid] = group;

            Matrix4 localMatrix = buildNodeTransform(nodeXml);
            Vector3 pos, scale;
            Quaternion quat;
            localMatrix.decompose(pos, quat, scale);
            group->position.copy(pos);
            group->quaternion.copy(quat);
            group->scale.copy(scale);

            // instance_geometry
            for (const auto& instGeom : nodeXml.children("instance_geometry")) {
                const std::string geomId = stripHash(instGeom.attribute("url").value());
                auto geomIt = geometries.find(geomId);
                if (geomIt == geometries.end()) continue;
                const GeometryData& geodata = geomIt->second;

                // Build material symbol -> material id map from bind_material
                std::unordered_map<std::string, std::string> symbolToMaterialId;
                const auto bindMat = instGeom.child("bind_material");
                if (bindMat) {
                    for (const auto& instMat : bindMat.child("technique_common").children("instance_material")) {
                        const std::string symbol = instMat.attribute("symbol").value();
                        const std::string target = stripHash(instMat.attribute("target").value());
                        symbolToMaterialId[symbol] = target;
                    }
                }

                // Create a mesh per primitive
                for (size_t pi = 0; pi < geodata.primitives.size(); ++pi) {
                    const auto& prim = geodata.primitives[pi];

                    const std::string geoKey = geomId + "/" + std::to_string(pi);
                    std::shared_ptr<BufferGeometry> geom;
                    if (const auto it = geometryCache.find(geoKey); it != geometryCache.end()) {
                        geom = it->second;
                    } else {
                        geom = buildGeometry(geodata, prim);
                        geometryCache[geoKey] = geom;
                    }
                    if (!geom) continue;

                    const std::string matId = symbolToMaterialId.contains(prim.material) ? symbolToMaterialId.at(prim.material) : prim.material;
                    auto mesh = Mesh::create(geom, loadMaterial(matId));
                    mesh->name = geodata.name;
                    group->add(mesh);
                }
            }

            // instance_controller (skinned meshes)
            for (const auto& instCtrl : nodeXml.children("instance_controller")) {
                const std::string ctrlId = stripHash(instCtrl.attribute("url").value());
                const auto ctrlGeomIt = controllerToGeometry.find(ctrlId);
                if (ctrlGeomIt == controllerToGeometry.end()) continue;

                const std::string geomId = ctrlGeomIt->second;
                const auto geomIt = geometries.find(geomId);
                if (geomIt == geometries.end()) continue;
                const GeometryData& geodata = geomIt->second;

                const auto skinIt = skinsByController.find(ctrlId);
                const SkinData* skinPtr = (skinIt != skinsByController.end()) ? &skinIt->second : nullptr;

                std::unordered_map<std::string, std::string> symbolToMaterialId;
                if (const auto bindMat = instCtrl.child("bind_material")) {
                    for (const auto& instMat : bindMat.child("technique_common").children("instance_material")) {
                        symbolToMaterialId[instMat.attribute("symbol").value()] = stripHash(instMat.attribute("target").value());
                    }
                }

                for (size_t pi = 0; pi < geodata.primitives.size(); ++pi) {
                    const auto& prim = geodata.primitives[pi];

                    const std::string geoKey = ctrlId + "@" + geomId + "/" + std::to_string(pi);
                    std::shared_ptr<BufferGeometry> geom;
                    if (const auto it = geometryCache.find(geoKey); it != geometryCache.end()) {
                        geom = it->second;
                    } else {
                        geom = buildGeometry(geodata, prim, skinPtr);
                        geometryCache[geoKey] = geom;
                    }
                    if (!geom) continue;

                    const std::string matId = symbolToMaterialId.contains(prim.material) ? symbolToMaterialId.at(prim.material) : prim.material;
                    auto material = loadMaterial(matId);
                    if (skinPtr) {
                        auto skinned = SkinnedMesh::create(geom, material);
                        skinned->name = geodata.name;
                        skinned->normalizeSkinWeights();
                        group->add(skinned);
                        pendingSkins.push_back({skinned, ctrlId});
                    } else {
                        auto mesh = Mesh::create(geom, material);
                        mesh->name = geodata.name;
                        group->add(mesh);
                    }
                }
            }

            // Recurse into child nodes
            for (const auto& child : nodeXml.children("node")) {
                parseNode(child, *group);
            }

            parent.add(group);
        };


        for (const auto& node : visualScene.children("node")) {
            parseNode(node, *sceneGroup);
        }

        // Apply up-axis conversion so the returned model is Y-up.
        // Collada permits X_UP, Y_UP or Z_UP; three.js/threepp is Y_UP.
        // Skipped when this loader is being used by an outer system (URDF/SDF/...)
        // that owns the coordinate frame.
        if (!ignoreUpDirection) {
            if (upAxis == "Z_UP") {
                sceneGroup->rotation.set(-math::PI / 2.f, 0.f, 0.f);
            } else if (upAxis == "X_UP") {
                sceneGroup->rotation.set(0.f, 0.f, math::PI / 2.f);
            }
        }

        // Resolve pending skin bindings now that every joint node exists in nodesById.
        for (const auto& pending : pendingSkins) {
            const auto skinIt = skinsByController.find(pending.controllerId);
            if (skinIt == skinsByController.end()) continue;
            const SkinData& sd = skinIt->second;

            std::vector<std::shared_ptr<Bone>> bones;
            bones.reserve(sd.jointNames.size());
            for (const auto& jName : sd.jointNames) {
                const auto it = nodesById.find(jName);
                if (it == nodesById.end()) {
                    std::cerr << "[ColladaLoader] skin joint '" << jName << "' not found in scene." << std::endl;
                    bones.clear();
                    break;
                }
                auto bone = std::dynamic_pointer_cast<Bone>(it->second);
                if (!bone) {
                    std::cerr << "[ColladaLoader] skin joint '" << jName << "' is not a Bone node." << std::endl;
                    bones.clear();
                    break;
                }
                bones.push_back(bone);
            }
            if (bones.empty() || bones.size() != sd.invBindMatrices.size()) continue;

            auto skeleton = Skeleton::create(bones, sd.invBindMatrices);
            // bindShape has been baked into the geometry in buildGeometry(),
            // so pass identity as the bind matrix.
            pending.mesh->bind(skeleton, Matrix4{});
        }

        // Parse animations. Each top-level <animation> is flattened into a list of
        // KeyframeTrack objects; those are then gathered either into the clips
        // declared by <library_animation_clips>, or into one fallback clip.
        if (const auto libAnims = root.child("library_animations")) {

            auto buildTracksFromAnim = [&](const pugi::xml_node& topAnim) {
                AnimationEntry entry;
                collectAnim(topAnim, entry);

                std::vector<std::shared_ptr<KeyframeTrack>> tracks;
                for (const auto& channel : entry.channels) {
                    const auto sampIt = entry.samplers.find(channel.samplerId);
                    if (sampIt == entry.samplers.end()) continue;
                    const AnimSampler& sampler = sampIt->second;

                    const auto inIt = entry.sources.find(sampler.inputId);
                    const auto outIt = entry.sources.find(sampler.outputId);
                    if (inIt == entry.sources.end() || outIt == entry.sources.end()) continue;

                    const AnimSource& inputSrc = inIt->second;
                    const AnimSource& outputSrc = outIt->second;
                    if (inputSrc.floats.empty() || outputSrc.floats.empty()) continue;

                    const ParsedTarget t = parseTarget(channel.target);
                    const auto nodeIt = nodesById.find(t.nodeId);
                    if (nodeIt == nodesById.end()) continue;
                    const std::string& boundName = nodeIt->second->name;

                    const std::vector<float>& times = inputSrc.floats;
                    const std::vector<float>& values = outputSrc.floats;
                    const size_t nKeys = times.size();

                    const bool isMatrix = (outputSrc.paramType == "float4x4") || (outputSrc.stride == 16);
                    if (isMatrix && values.size() >= nKeys * 16 && t.component.empty()) {
                        std::vector<float> posVals;
                        posVals.reserve(nKeys * 3);
                        std::vector<float> quatVals;
                        quatVals.reserve(nKeys * 4);
                        std::vector<float> scaleVals;
                        scaleVals.reserve(nKeys * 3);

                        for (size_t k = 0; k < nKeys; ++k) {
                            const float* m = values.data() + k * 16;
                            Matrix4 mat;
                            mat.set(m[0], m[1], m[2], m[3],
                                    m[4], m[5], m[6], m[7],
                                    m[8], m[9], m[10], m[11],
                                    m[12], m[13], m[14], m[15]);
                            Vector3 p, s;
                            Quaternion q;
                            mat.decompose(p, q, s);
                            posVals.insert(posVals.end(), {p.x, p.y, p.z});
                            quatVals.insert(quatVals.end(), {static_cast<float>(q.x),
                                                             static_cast<float>(q.y),
                                                             static_cast<float>(q.z),
                                                             static_cast<float>(q.w)});
                            scaleVals.insert(scaleVals.end(), {s.x, s.y, s.z});
                        }

                        tracks.emplace_back(std::make_shared<VectorKeyframeTrack>(
                                boundName + ".position", times, posVals));
                        tracks.emplace_back(std::make_shared<QuaternionKeyframeTrack>(
                                boundName + ".quaternion", times, quatVals));
                        tracks.emplace_back(std::make_shared<VectorKeyframeTrack>(
                                boundName + ".scale", times, scaleVals));
                        continue;
                    }

                    if (t.component.empty()) {
                        if (t.sid == "translate" || t.sid == "location" || t.sid == "translation") {
                            if (values.size() >= nKeys * 3) {
                                tracks.emplace_back(std::make_shared<VectorKeyframeTrack>(
                                        boundName + ".position", times, values));
                            }
                        } else if (t.sid == "scale") {
                            if (values.size() >= nKeys * 3) {
                                tracks.emplace_back(std::make_shared<VectorKeyframeTrack>(
                                        boundName + ".scale", times, values));
                            }
                        }
                    }
                }
                return tracks;
            };

            // Build per-animation tracks indexed by id so <animation_clip> can reference them.
            std::unordered_map<std::string, std::vector<std::shared_ptr<KeyframeTrack>>> tracksById;
            std::vector<std::string> orderedIds;
            for (const auto& topAnim : libAnims.children("animation")) {
                const std::string id = topAnim.attribute("id").value();
                auto tracks = buildTracksFromAnim(topAnim);
                if (tracks.empty()) continue;
                if (!id.empty()) {
                    tracksById[id] = std::move(tracks);
                    orderedIds.emplace_back(id);
                } else {
                    // Un-id'd animation — still merge into the fallback bucket.
                    const std::string synthetic = "__anim_" + std::to_string(orderedIds.size());
                    tracksById[synthetic] = std::move(tracks);
                    orderedIds.emplace_back(synthetic);
                }
            }

            const auto libClips = root.child("library_animation_clips");
            if (libClips && libClips.child("animation_clip")) {
                for (const auto& clipNode : libClips.children("animation_clip")) {
                    std::string clipName = clipNode.attribute("name").value();
                    if (clipName.empty()) clipName = clipNode.attribute("id").value();

                    std::vector<std::shared_ptr<KeyframeTrack>> clipTracks;
                    for (const auto& inst : clipNode.children("instance_animation")) {
                        const std::string ref = stripHash(inst.attribute("url").value());
                        auto it = tracksById.find(ref);
                        if (it == tracksById.end()) continue;
                        clipTracks.insert(clipTracks.end(), it->second.begin(), it->second.end());
                    }
                    if (clipTracks.empty()) continue;

                    auto clip = std::make_shared<AnimationClip>(clipName, -1.f, clipTracks);
                    clip->resetDuration();
                    sceneGroup->animations.emplace_back(clip);
                }
            } else {
                // No <library_animation_clips>: merge every track into a single default clip.
                std::vector<std::shared_ptr<KeyframeTrack>> allTracks;
                for (const auto& id : orderedIds) {
                    auto& t = tracksById[id];
                    allTracks.insert(allTracks.end(), t.begin(), t.end());
                }
                if (!allTracks.empty()) {
                    auto clip = std::make_shared<AnimationClip>("default", -1.f, allTracks);
                    clip->resetDuration();
                    sceneGroup->animations.emplace_back(clip);
                }
            }
        }

        return sceneGroup;
    }
};

ColladaLoader::ColladaLoader()
    : pimpl_(std::make_unique<Impl>()) {}

ColladaLoader& ColladaLoader::setIgnoreUpDirection(bool ignore) {
    pimpl_->ignoreUpDirection = ignore;
    return *this;
}

std::shared_ptr<Group> ColladaLoader::load(const std::filesystem::path& path) {
    return pimpl_->load(path);
}

ColladaLoader::~ColladaLoader() = default;