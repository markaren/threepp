// GLTFLoader's parser, shared by the translation units that implement it:
// GLTFLoader.cpp (buffers, accessors, nodes, skins, animations and the
// document itself), gltf/GLTFMaterials.cpp (images, textures, materials) and
// gltf/GLTFMeshes.cpp (geometry, meshes and the node tree). One file used to
// hold all of it, and came close to the section limit of a MinGW Debug object.
// Internal to the library.

#ifndef THREEPP_GLTFPARSER_HPP
#define THREEPP_GLTFPARSER_HPP

#include "threepp/loaders/GLTFLoader.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <threepp/animation/AnimationClip.hpp>
#include <threepp/animation/MaterialAnimationProxy.hpp>
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
#include <threepp/objects/InstancedMesh.hpp>
#include <threepp/objects/ObjectWithMaterials.hpp>
#include <threepp/objects/Skeleton.hpp>
#include <threepp/objects/SkinnedMesh.hpp>

#include "threepp/utils/Base64.hpp"

#include <unordered_set>

#include "meshoptimizer.h"

namespace threepp::gltf {

    using json = nlohmann::json;
    namespace fs = std::filesystem;

    using utils::base64Decode;

    // Percent-decode a glTF URI (RFC 3986). glTF texture URIs with spaces or
    // other reserved chars arrive as e.g. "Base%20Color.jpg"; we must decode
    // before using them as filesystem paths. Data URIs are handled separately
    // and must NOT be passed here.
    inline std::string percentDecode(const std::string& uri) {
        std::string out;
        out.reserve(uri.size());
        auto hex = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return 10 + c - 'a';
            if (c >= 'A' && c <= 'F') return 10 + c - 'A';
            return -1;
        };
        for (size_t i = 0; i < uri.size(); ++i) {
            if (uri[i] == '%' && i + 2 < uri.size()) {
                int hi = hex(uri[i + 1]);
                int lo = hex(uri[i + 2]);
                if (hi >= 0 && lo >= 0) {
                    out.push_back(static_cast<char>((hi << 4) | lo));
                    i += 2;
                    continue;
                }
            }
            out.push_back(uri[i]);
        }
        return out;
    }

    // Read an already-open binary stream into a byte vector in one shot,
    // sizing the buffer from the file length instead of growing it a
    // character at a time (as std::istreambuf_iterator does). Falls back to
    // the iterator form if the size can't be determined (e.g. a pipe).
    inline std::vector<uint8_t> readAllBytes(std::ifstream& f, const fs::path& p) {
        std::error_code ec;
        auto sz = fs::file_size(p, ec);
        std::vector<uint8_t> out;
        if (!ec) {
            out.resize(static_cast<size_t>(sz));
            f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz));
            out.resize(static_cast<size_t>(f.gcount()));
        } else {
            out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
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

    // glTF sampler filter modes (OpenGL enum values).
    constexpr int FILTER_NEAREST = 9728;
    constexpr int FILTER_LINEAR = 9729;
    constexpr int FILTER_NEAREST_MIPMAP_NEAREST = 9984;
    constexpr int FILTER_LINEAR_MIPMAP_NEAREST = 9985;
    constexpr int FILTER_NEAREST_MIPMAP_LINEAR = 9986;
    constexpr int FILTER_LINEAR_MIPMAP_LINEAR = 9987;

    inline int componentSize(int componentType) {
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

    inline int typeCount(const std::string& type) {
        if (type == "SCALAR") return 1;
        if (type == "VEC2") return 2;
        if (type == "VEC3") return 3;
        if (type == "VEC4") return 4;
        if (type == "MAT2") return 4;
        if (type == "MAT3") return 9;
        if (type == "MAT4") return 16;
        return 0;
    }

    // Decode a single accessor component to float, honouring the accessor's
    // `normalized` flag per glTF 2.0 §3.6.2.2. When normalized, integer types
    // map to [0,1] (unsigned) or [-1,1] (signed); otherwise the integer value
    // is taken verbatim (the KHR_mesh_quantization case, where a node/UV
    // transform performs the dequantization). Signed BYTE (5120) is handled
    // in both branches — the previous switch dropped it to 0.0.
    inline float decodeComponentFloat(const uint8_t* src, int ct, bool normalized) {
        switch (ct) {
            case COMP_FLOAT: {
                float t;
                std::memcpy(&t, src, 4);
                return t;
            }
            case COMP_UNSIGNED_BYTE: {
                uint8_t t = *src;
                return normalized ? t / 255.f : static_cast<float>(t);
            }
            case COMP_BYTE: {
                int8_t t;
                std::memcpy(&t, src, 1);
                return normalized ? std::max(-1.f, t / 127.f) : static_cast<float>(t);
            }
            case COMP_UNSIGNED_SHORT: {
                uint16_t t;
                std::memcpy(&t, src, 2);
                return normalized ? t / 65535.f : static_cast<float>(t);
            }
            case COMP_SHORT: {
                int16_t t;
                std::memcpy(&t, src, 2);
                return normalized ? std::max(-1.f, t / 32767.f) : static_cast<float>(t);
            }
            case COMP_UNSIGNED_INT: {
                uint32_t t;
                std::memcpy(&t, src, 4);
                return static_cast<float>(t);
            }
        }
        return 0.f;
    }

    // Decode a single integer index component (never normalized).
    inline uint32_t decodeIndex(const uint8_t* src, int ct) {
        switch (ct) {
            case COMP_UNSIGNED_BYTE:
                return *src;
            case COMP_UNSIGNED_SHORT: {
                uint16_t t;
                std::memcpy(&t, src, 2);
                return t;
            }
            case COMP_UNSIGNED_INT: {
                uint32_t t;
                std::memcpy(&t, src, 4);
                return t;
            }
        }
        return 0;
    }

    // ===========================================================================
    //  Parser state
    // ===========================================================================
    struct GLTFParser {
        json gltf;
        std::vector<std::vector<uint8_t>> buffers;
        fs::path basePath;
        bool preserveNarrowAttributes = true;// mirrors GLTFLoader's flag

        // Cache to avoid duplicate GPU uploads.
        // Textures are keyed by (texIdx, colorSpace): the same glTF texture
        // is legitimately used in multiple roles (e.g. baseColor sRGB and,
        // elsewhere, as a Linear data map), and each colour-space variant
        // needs its own tagged Texture. Caching per variant means each is
        // built (and its pixels copied) at most once instead of on every
        // request. ColorSpace is stored as its underlying int.
        std::map<std::pair<int, int>, std::shared_ptr<Texture>> textureCache;
        std::unordered_map<int, std::shared_ptr<Material>> materialCache;
        // The vertexColors = true copy of a material, for the primitives that
        // carry COLOR_0. Keyed like materialCache (-1 = the default material).
        std::unordered_map<int, std::shared_ptr<Material>> vertexColorMaterialCache;

        // Decoded-geometry cache, keyed by (meshIdx, primIdx, hasSkin).
        // loadMesh is invoked once per referencing node; without this the
        // same primitive is fully re-decoded for every node. Cached
        // BufferGeometry is shared by all referencing meshes (renderers key
        // GPU uploads on geometry id, so sharing = one upload). Morph
        // influences and variant tags stay per-Mesh; only the immutable
        // vertex/index/morph-attribute data is shared.
        std::map<std::tuple<int, int, bool>, std::shared_ptr<BufferGeometry>> geometryCache;

        // KHR_texture_transform result cache, keyed by
        // (texIdx, colorSpace, offX, offY, scaleX, scaleY, rotation, texCoord).
        // A transform requires cloning the base texture; caching by the full
        // parameter set means identical transforms clone at most once instead
        // of once per material slot / call site.
        std::map<std::tuple<int, int, float, float, float, float, float, int>,
                 std::shared_ptr<Texture>> textureTransformCache;

        // Guards one-time KHR_materials_variants mapping collection per
        // (meshIdx, primIdx): loadMesh runs per node, and the mappings are
        // appended to primVariantData, so without this guard a
        // multiply-referenced mesh would register duplicate variant entries.
        std::set<std::pair<int, int>> variantsCollected;

        // Skeleton support
        std::unordered_set<int> jointNodeSet;
        std::unordered_map<int, std::shared_ptr<Object3D>> nodeObjects;
        std::unordered_map<int, std::shared_ptr<Skeleton>> skinCache;
        std::unordered_set<int> builtNodes;

        // Non-joint mesh nodes whose pre-created Group wrapper has been
        // replaced with the actual meshObj (Mesh or multi-prim Group) —
        // avoids attaching the mesh twice in buildNode.
        std::unordered_set<int> collapsedMeshNodes;

        // KHR_animation_pointer: one invisible proxy per animated material,
        // attached to the scene root so the AnimationMixer can resolve it.
        std::unordered_map<int, std::shared_ptr<MaterialAnimationProxy>> matAnimProxies;

        // KHR_materials_variants support
        std::vector<std::string> variantNames;
        struct PrimVariantMapping {
            int materialIdx;
            std::vector<int> variantIndices;
        };
        // meshIdx -> primIdx -> list of variant mappings
        std::unordered_map<int, std::unordered_map<int, std::vector<PrimVariantMapping>>> primVariantData;

        // Cache of decoded EXT_meshopt_compression bufferViews (keyed by bufferView index)
        std::unordered_map<int, std::vector<uint8_t>> meshoptCache;

        // -----------------------------------------------------------------------
        //  Buffer/accessor helpers
        // -----------------------------------------------------------------------

        const std::vector<uint8_t>& resolveBuffer(int idx);

        struct AccessorData {
            const uint8_t* ptr;// nullptr for a sparse accessor with no base bufferView
            size_t byteStride;
            size_t count;
            int componentType;
            int numComponents;
            bool normalized;
        };

        // Byte span occupied by `count` elements of `elemSize` bytes at `stride`.
        static size_t accessorSpan(size_t count, size_t stride, size_t elemSize) {
            return count == 0 ? 0 : (count - 1) * stride + elemSize;
        }

        // Meshopt compression on a bufferView: EXT_meshopt_compression is the
        // ratified name; KHR_meshopt_compression is the earlier draft name
        // still found in the wild. Accept both. Returns nullptr when absent.
        static const json* meshoptExtension(const json& def) {
            auto ext = def.find("extensions");
            if (ext == def.end()) return nullptr;
            if (auto e = ext->find("EXT_meshopt_compression"); e != ext->end()) return &*e;
            if (auto e = ext->find("KHR_meshopt_compression"); e != ext->end()) return &*e;
            return nullptr;
        }

        const std::vector<uint8_t>& decodeMeshoptBV(int bvIdx);

        // Accessors declared by the document (0 if the array is absent).
        size_t accessorCount() const {
            return gltf.contains("accessors") ? gltf["accessors"].size() : 0;
        }

        bool accessorIndexValid(int accessorIdx) const {
            return accessorIdx >= 0 &&
                   static_cast<size_t>(accessorIdx) < accessorCount();
        }

        // Same check straight off a json value: a reference that isn't even
        // an integer is as unusable as an out-of-range one.
        bool accessorIndexValid(const json& v) const {
            return v.is_number_integer() && accessorIndexValid(v.get<int>());
        }

        AccessorData getAccessor(int accessorIdx);

        const uint8_t* plainBufferViewPtr(int bvIdx, size_t extraOffset, size_t neededBytes);

        // Apply an accessor's `sparse` overlay onto an already-materialised
        // flat buffer. `writeElement(index, valuePtr)` writes the nc
        // components at sparse position `index`, reading them from `valuePtr`
        // (values share the accessor's componentType). Shared by the float and
        // index decoders.
        template<class WriteFn>
        void applySparse(const json& sparse, int accCt, int nc, WriteFn&& writeElement) {
            size_t sCount = sparse["count"].get<size_t>();
            if (sCount == 0) return;

            const auto& idxDef = sparse["indices"];
            int idxBv = idxDef["bufferView"].get<int>();
            size_t idxOff = idxDef.value("byteOffset", 0);
            int idxCt = idxDef["componentType"].get<int>();
            const size_t idxCompSize = static_cast<size_t>(componentSize(idxCt));

            const auto& valDef = sparse["values"];
            int valBv = valDef["bufferView"].get<int>();
            size_t valOff = valDef.value("byteOffset", 0);
            const size_t valElemSize = static_cast<size_t>(componentSize(accCt) * nc);

            const uint8_t* idxPtr = plainBufferViewPtr(idxBv, idxOff, sCount * idxCompSize);
            const uint8_t* valPtr = plainBufferViewPtr(valBv, valOff, sCount * valElemSize);

            for (size_t k = 0; k < sCount; ++k) {
                uint32_t target = decodeIndex(idxPtr + k * idxCompSize, idxCt);
                writeElement(target, valPtr + k * valElemSize);
            }
        }

        std::vector<float> readFloats(int accessorIdx);

        // Read an accessor's components in their native integer width,
        // honouring stride and any sparse overlay (sparse values share the
        // accessor's componentType, so a plain byte copy is exact). Callers
        // must have checked that componentType matches sizeof(T) — this is
        // the storage behind narrow BufferAttributes, where the normalized
        // flag travels on the attribute instead of being baked into floats.
        template<class T>
        std::vector<T> readNarrow(int accessorIdx) {
            auto [ptr, stride, count, ct, nc, normalized] = getAccessor(accessorIdx);
            std::vector<T> out(count * static_cast<size_t>(nc), T{});

            if (ptr) {
                if (stride == sizeof(T) * static_cast<size_t>(nc)) {
                    std::memcpy(out.data(), ptr, out.size() * sizeof(T));
                } else {
                    for (size_t i = 0; i < count; ++i) {
                        std::memcpy(&out[i * nc], ptr + i * stride, nc * sizeof(T));
                    }
                }
            }

            const auto& acc = gltf["accessors"][accessorIdx];
            if (acc.contains("sparse")) {
                const int lnc = nc;
                applySparse(acc["sparse"], ct, nc,
                            [&, lnc](uint32_t target, const uint8_t* valPtr) {
                                if (static_cast<size_t>(target) * lnc + lnc > out.size()) return;
                                std::memcpy(&out[target * lnc], valPtr, lnc * sizeof(T));
                            });
            }
            return out;
        }

        std::vector<uint32_t> readIndices(int accessorIdx);

        std::vector<float> readJointIndicesAsFloat(int accessorIdx);

        // -----------------------------------------------------------------------
        //  Skeleton helpers
        // -----------------------------------------------------------------------

        void gatherJoints();

        void applyNodeTransform(const std::shared_ptr<Object3D>& obj, const json& nodeDef);

        void preCreateNodes();

        std::vector<int> collectReachable(const json& sceneDef);

        void resetSceneBuildState();

        std::shared_ptr<Skeleton> loadSkin(int skinIdx);

        // -----------------------------------------------------------------------
        //  Image / Texture loading
        // -----------------------------------------------------------------------

        std::optional<Image> loadImageData(int imageIdx, std::vector<uint8_t>* embedded = nullptr);

        std::shared_ptr<Texture> loadTexture(int texIdx, ColorSpace cs = ColorSpace::sRGB);

        std::shared_ptr<Texture> applyTextureTransform(
                const json& texInfo, int texIdx, ColorSpace cs = ColorSpace::sRGB);

        // -----------------------------------------------------------------------
        //  Material
        // -----------------------------------------------------------------------

        std::shared_ptr<Material> loadMaterial(int matIdx);

        // -----------------------------------------------------------------------
        //  Mesh
        // -----------------------------------------------------------------------

        std::shared_ptr<BufferGeometry> buildPrimitiveGeometry(
                int meshIdx, int primIdx, const json& prim, bool hasSkin);

        bool primitiveAccessorsValid(int meshIdx, const std::string& meshName,
                                     int primIdx, const json& prim) const;

        std::shared_ptr<Object3D> loadMesh(int meshIdx, bool hasSkin = false);

        // -----------------------------------------------------------------------
        //  Node / Scene hierarchy
        // -----------------------------------------------------------------------

        std::shared_ptr<Object3D> applyGpuInstancing(const json& extData,
                                                    const std::shared_ptr<Object3D>& meshObj);

        std::shared_ptr<Object3D> buildMeshObjForNode(int nodeIdx);

        void tryCollapseMeshWrapper(int nodeIdx);

        void buildNode(int nodeIdx);

        std::shared_ptr<Group> loadScene(int sceneIdx);

        // -----------------------------------------------------------------------
        //  Entry points
        // -----------------------------------------------------------------------

        GLTFResult buildResult();

        GLTFResult parseGLTF(const std::string& jsonText);

        GLTFResult parseGLB(const std::vector<uint8_t>& data);

        // -----------------------------------------------------------------------
        //  Animation
        // -----------------------------------------------------------------------

        std::vector<std::shared_ptr<AnimationClip>> loadAnimations();

        void resolveVariants(GLTFResult& result);
    };

}// namespace threepp::gltf

#endif//THREEPP_GLTFPARSER_HPP
