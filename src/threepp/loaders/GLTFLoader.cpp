// glTF 2.0 loader for threepp

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

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace threepp {

    namespace {

        using utils::base64Decode;

        // Percent-decode a glTF URI (RFC 3986). glTF texture URIs with spaces or
        // other reserved chars arrive as e.g. "Base%20Color.jpg"; we must decode
        // before using them as filesystem paths. Data URIs are handled separately
        // and must NOT be passed here.
        std::string percentDecode(const std::string& uri) {
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
        std::vector<uint8_t> readAllBytes(std::ifstream& f, const fs::path& p) {
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
                    fs::path p = basePath / percentDecode(uri);
                    std::ifstream f(p, std::ios::binary);
                    if (!f) throw std::runtime_error("Cannot open buffer file: " + p.string());
                    buffers[idx] = readAllBytes(f, p);
                }
                return buffers[idx];
            }

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

            // Decode a meshopt-compressed bufferView on first access and cache it.
            const std::vector<uint8_t>& decodeMeshoptBV(int bvIdx) {
                auto it = meshoptCache.find(bvIdx);
                if (it != meshoptCache.end()) return it->second;

                const auto& ext = *meshoptExtension(gltf["bufferViews"][bvIdx]);
                int bufIdx        = ext["buffer"].get<int>();
                size_t byteOffset = ext.value("byteOffset", 0);
                size_t byteLength = ext["byteLength"].get<size_t>();
                size_t byteStride = ext["byteStride"].get<size_t>();
                size_t count      = ext["count"].get<size_t>();
                std::string mode  = ext["mode"].get<std::string>();
                std::string filter = ext.value("filter", "NONE");

                const auto& compressed = resolveBuffer(bufIdx);
                if (byteOffset + byteLength > compressed.size())
                    throw std::runtime_error("EXT_meshopt_compression: source range out of bounds (bufferView " +
                                             std::to_string(bvIdx) + ")");
                const uint8_t* src = compressed.data() + byteOffset;

                auto& decoded = meshoptCache[bvIdx];
                decoded.resize(count * byteStride);

                int rc = 0;
                if (mode == "ATTRIBUTES")
                    rc = meshopt_decodeVertexBuffer(decoded.data(), count, byteStride, src, byteLength);
                else if (mode == "TRIANGLES")
                    rc = meshopt_decodeIndexBuffer(decoded.data(), count, byteStride, src, byteLength);
                else if (mode == "INDICES")
                    rc = meshopt_decodeIndexSequence(decoded.data(), count, byteStride, src, byteLength);
                else
                    throw std::runtime_error("EXT_meshopt_compression: unknown mode '" + mode + "'");

                if (rc != 0)
                    throw std::runtime_error("EXT_meshopt_compression: decode failed (bufferView " + std::to_string(bvIdx) + ")");

                if (filter == "OCTAHEDRAL")
                    meshopt_decodeFilterOct(decoded.data(), count, byteStride);
                else if (filter == "QUATERNION")
                    meshopt_decodeFilterQuat(decoded.data(), count, byteStride);
                else if (filter == "EXPONENTIAL")
                    meshopt_decodeFilterExp(decoded.data(), count, byteStride);
                else if (filter == "COLOR")
                    meshopt_decodeFilterColor(decoded.data(), count, byteStride);

                return decoded;
            }

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

            AccessorData getAccessor(int accessorIdx) {
                // Every read helper funnels through here, so one range check
                // turns a bad reference into a named error instead of a
                // nlohmann type_error out of `gltf["accessors"][-1]`.
                if (!accessorIndexValid(accessorIdx))
                    throw std::runtime_error("Accessor index " + std::to_string(accessorIdx) +
                                             " out of range [0, " + std::to_string(accessorCount()) + ")");

                const auto& acc = gltf["accessors"][accessorIdx];
                size_t accOff = acc.value("byteOffset", 0);
                size_t count = acc["count"].get<size_t>();
                int ct = acc["componentType"].get<int>();
                std::string type = acc["type"].get<std::string>();
                int nc = typeCount(type);
                bool normalized = acc.value("normalized", false);
                const size_t elemSize = static_cast<size_t>(componentSize(ct) * nc);

                // A sparse accessor may omit bufferView entirely (its base is all
                // zeros, fully replaced by the sparse overlay). Callers handle a
                // null pointer by zero-filling the base.
                if (!acc.contains("bufferView"))
                    return {nullptr, elemSize, count, ct, nc, normalized};

                int bvIdx = acc["bufferView"].get<int>();
                const auto& bv = gltf["bufferViews"][bvIdx];

                if (const json* mo = meshoptExtension(bv)) {
                    size_t byteStride = (*mo)["byteStride"].get<size_t>();
                    const auto& decoded = decodeMeshoptBV(bvIdx);
                    if (accOff + accessorSpan(count, byteStride, elemSize) > decoded.size())
                        throw std::runtime_error("Accessor " + std::to_string(accessorIdx) +
                                                 " out of bounds of decoded meshopt bufferView");
                    return {decoded.data() + accOff, byteStride, count, ct, nc, normalized};
                }

                int bufIdx = bv["buffer"].get<int>();
                size_t bvOffset = bv.value("byteOffset", 0);
                size_t bvStride = bv.value("byteStride", 0);
                const auto& buf = resolveBuffer(bufIdx);
                size_t stride = bvStride > 0 ? bvStride : elemSize;

                // Validate the accessor's byte span fits both its bufferView (if a
                // byteLength is declared) and the backing buffer, so malformed
                // offsets/strides/counts fail cleanly instead of reading OOB.
                const size_t span = accessorSpan(count, stride, elemSize);
                if (bv.contains("byteLength")) {
                    size_t bvLen = bv["byteLength"].get<size_t>();
                    if (accOff + span > bvLen)
                        throw std::runtime_error("Accessor " + std::to_string(accessorIdx) +
                                                 " out of bounds of bufferView " + std::to_string(bvIdx));
                }
                if (bvOffset + accOff + span > buf.size())
                    throw std::runtime_error("Accessor " + std::to_string(accessorIdx) +
                                             " out of bounds of buffer " + std::to_string(bufIdx));

                const uint8_t* base = buf.data() + bvOffset + accOff;
                return {base, stride, count, ct, nc, normalized};
            }

            // Raw pointer into a plain (uncompressed) bufferView at an extra byte
            // offset, validating that `neededBytes` fit. Used for sparse index /
            // value bufferViews.
            const uint8_t* plainBufferViewPtr(int bvIdx, size_t extraOffset, size_t neededBytes) {
                const auto& bv = gltf["bufferViews"][bvIdx];
                int bufIdx = bv["buffer"].get<int>();
                size_t bvOffset = bv.value("byteOffset", 0);
                const auto& buf = resolveBuffer(bufIdx);
                const size_t start = bvOffset + extraOffset;
                if (start + neededBytes > buf.size())
                    throw std::runtime_error("Sparse bufferView " + std::to_string(bvIdx) +
                                             " out of bounds");
                return buf.data() + start;
            }

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

            // Read accessor into a flat float vector, honouring `normalized` and
            // applying any sparse overlay. A sparse accessor with no base
            // bufferView starts zero-filled.
            std::vector<float> readFloats(int accessorIdx) {
                auto [ptr, stride, count, ct, nc, normalized] = getAccessor(accessorIdx);
                const size_t compSize = static_cast<size_t>(componentSize(ct));
                std::vector<float> out(count * nc, 0.f);

                if (ptr) {
                    if (ct == COMP_FLOAT && stride == static_cast<size_t>(nc) * 4) {
                        // Tightly packed FLOAT data (the common case) — a single
                        // bulk copy. getAccessor validated the span fits.
                        std::memcpy(out.data(), ptr, count * nc * sizeof(float));
                    } else {
                        for (size_t i = 0; i < count; ++i) {
                            const uint8_t* row = ptr + i * stride;
                            for (int j = 0; j < nc; ++j)
                                out[i * nc + j] = decodeComponentFloat(row + j * compSize, ct, normalized);
                        }
                    }
                }

                const auto& acc = gltf["accessors"][accessorIdx];
                if (acc.contains("sparse")) {
                    // Structured bindings can't be captured by a lambda pre-C++20
                    // in a fully portable way; copy the ones the lambda needs.
                    const int lct = ct, lnc = nc;
                    const bool lnorm = normalized;
                    const size_t lcomp = compSize;
                    applySparse(acc["sparse"], ct, nc,
                                [&, lct, lnc, lnorm, lcomp](uint32_t target, const uint8_t* valPtr) {
                                    if (static_cast<size_t>(target) * lnc + lnc > out.size()) return;
                                    for (int j = 0; j < lnc; ++j)
                                        out[target * lnc + j] =
                                                decodeComponentFloat(valPtr + j * lcomp, lct, lnorm);
                                });
                }
                return out;
            }

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

            std::vector<uint32_t> readIndices(int accessorIdx) {
                auto [ptr, stride, count, ct, nc, accNormalized] = getAccessor(accessorIdx);
                std::vector<uint32_t> out(count, 0u);

                if (ptr) {
                    if (ct == COMP_UNSIGNED_INT && stride == 4) {
                        // Tightly packed uint32 indices — bulk copy.
                        std::memcpy(out.data(), ptr, count * sizeof(uint32_t));
                    } else {
                        for (size_t i = 0; i < count; ++i)
                            out[i] = decodeIndex(ptr + i * stride, ct);
                    }
                }

                const auto& acc = gltf["accessors"][accessorIdx];
                if (acc.contains("sparse")) {
                    const int lct = ct;
                    applySparse(acc["sparse"], ct, nc,
                                [&, lct](uint32_t target, const uint8_t* valPtr) {
                                    if (target >= out.size()) return;
                                    out[target] = decodeIndex(valPtr, lct);
                                });
                }
                return out;
            }

            // Read JOINTS_0 accessor as float without normalisation (joint
            // indices are integers, never in [0,1]). Sparse overlay applied raw.
            std::vector<float> readJointIndicesAsFloat(int accessorIdx) {
                auto [ptr, stride, count, ct, nc, accNormalized] = getAccessor(accessorIdx);
                const size_t compSize = static_cast<size_t>(componentSize(ct));
                std::vector<float> out(count * nc, 0.f);

                auto decodeRaw = [](const uint8_t* src, int ct) -> float {
                    if (ct == COMP_UNSIGNED_BYTE) return static_cast<float>(*src);
                    if (ct == COMP_UNSIGNED_SHORT) {
                        uint16_t t;
                        std::memcpy(&t, src, 2);
                        return static_cast<float>(t);
                    }
                    return 0.f;
                };

                if (ptr) {
                    for (size_t i = 0; i < count; ++i) {
                        const uint8_t* row = ptr + i * stride;
                        for (int j = 0; j < nc; ++j)
                            out[i * nc + j] = decodeRaw(row + j * compSize, ct);
                    }
                }

                const auto& acc = gltf["accessors"][accessorIdx];
                if (acc.contains("sparse")) {
                    const int lct = ct, lnc = nc;
                    const size_t lcomp = compSize;
                    applySparse(acc["sparse"], ct, nc,
                                [&, lct, lnc, lcomp](uint32_t target, const uint8_t* valPtr) {
                                    if (static_cast<size_t>(target) * lnc + lnc > out.size()) return;
                                    for (int j = 0; j < lnc; ++j)
                                        out[target * lnc + j] = decodeRaw(valPtr + j * lcomp, lct);
                                });
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
                    // Fall back to a synthetic "node_N" name for unnamed glTF nodes
                    // (e.g. BrainStem). Animation tracks also use this synthetic
                    // name (see loadAnimations), so PropertyBinding::findNode
                    // resolves bones by matching names instead of defaulting to
                    // root and silently losing the animation.
                    obj->name = nodeDef.value("name", "node_" + std::to_string(i));
                    applyNodeTransform(obj, nodeDef);
                    nodeObjects[i] = obj;
                }
            }

            // Node indices reachable from a scene's roots (DFS through children),
            // returned in ascending order. Only reachable nodes are built (and
            // their meshes decoded); unreachable nodes are never part of the
            // returned scene graph, so skipping them changes nothing observable
            // — it just avoids decoding meshes that would be discarded. Guards
            // against cycles and out-of-range indices in malformed files.
            std::vector<int> collectReachable(const json& sceneDef) {
                std::vector<int> order;
                if (!sceneDef.contains("nodes")) return order;
                const int numNodes = gltf.contains("nodes") ? static_cast<int>(gltf["nodes"].size()) : 0;
                std::unordered_set<int> visited;
                std::vector<int> stack = sceneDef["nodes"].get<std::vector<int>>();
                while (!stack.empty()) {
                    int i = stack.back();
                    stack.pop_back();
                    if (i < 0 || i >= numNodes) continue;
                    if (!visited.insert(i).second) continue;
                    const auto& nd = gltf["nodes"][i];
                    if (nd.contains("children"))
                        for (int c : nd["children"].get<std::vector<int>>())
                            stack.push_back(c);
                }
                order.assign(visited.begin(), visited.end());
                std::sort(order.begin(), order.end());
                return order;
            }

            // Re-instantiate per-scene node objects so multiple scenes build
            // independent graphs (a node referenced by two scenes must not be
            // reparented/stolen from the first). Heavy, scene-independent caches
            // (geometry / material / texture / image / variant data) persist and
            // are shared across scenes; only the per-scene graph state resets.
            // Skeletons are cleared because they bind to this scene's bone
            // instances.
            void resetSceneBuildState() {
                nodeObjects.clear();
                builtNodes.clear();
                collapsedMeshNodes.clear();
                skinCache.clear();
                preCreateNodes();
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

            // `embedded`, when given, receives the encoded bytes for an image
            // the .glb carries INSIDE itself (a bufferView, or a data: uri) —
            // the ones with no file anywhere for an exporter to point at. An
            // image referenced by path keeps having a file, so it gets nothing
            // and the caller stays on the path it always had.
            std::optional<Image> loadImageData(int imageIdx, std::vector<uint8_t>* embedded = nullptr) {
                const auto& imgDef = gltf["images"][imageIdx];
                std::vector<uint8_t> raw;
                bool isEmbedded = true;

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
                        fs::path p = basePath / percentDecode(uri);
                        std::ifstream f(p, std::ios::binary);
                        if (!f) throw std::runtime_error("Cannot open image: " + p.string());
                        raw = readAllBytes(f, p);
                        isEmbedded = false;
                    }
                } else {
                    throw std::runtime_error("Image " + std::to_string(imageIdx) + " has no source");
                }

                ImageLoader loader;
                // flipY false: glTF's UV origin is the top-left corner, so the
                // rows are used in the order the file stores them. Whatever
                // reads the retained bytes back has to agree.
                auto image = loader.load(raw, 4, false);
                if (image && embedded && isEmbedded) *embedded = std::move(raw);

                return image;
            }

            std::shared_ptr<Texture> loadTexture(int texIdx, ColorSpace cs = ColorSpace::sRGB) {
                const std::pair<int, int> key{texIdx, static_cast<int>(cs)};
                if (auto it = textureCache.find(key); it != textureCache.end())
                    return it->second;

                const auto& texDef = gltf["textures"][texIdx];
                int imageIdx = texDef.value("source", -1);
                if (imageIdx < 0) return nullptr;

                std::vector<uint8_t> encoded;
                auto image = loadImageData(imageIdx, &encoded);
                if (!image) return nullptr;

                // Move the decoded pixels straight into the texture — no copy and
                // no lingering decode buffer. The (texIdx, colorSpace) texture
                // cache above means each variant is built exactly once, so the
                // rare second colour-space role of one texture re-decodes rather
                // than retaining every decoded image for the whole load (which
                // would inflate peak memory on texture-heavy scenes).
                auto tex = Texture::create(std::vector<Image>{std::move(*image)});
                tex->colorSpace = cs;
                // The bytes this came in as, for a texture the .glb keeps inside
                // itself and that therefore has no sourceFile. Carries the same
                // flipY the decode above used (see Texture::encodedSource).
                tex->encodedSource = Texture::EncodedImage::from(std::move(encoded), false);
                tex->needsUpdate();

                // glTF 2.0 §3.8.4: when sampler is undefined, repeat wrapping
                // and auto filtering must be used. Texture's C++ default is
                // ClampToEdge, so always default to Repeat for glTF assets.
                tex->wrapS = TextureWrapping::Repeat;
                tex->wrapT = TextureWrapping::Repeat;

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

                    // Filters: only override threepp's defaults (magFilter=Linear,
                    // minFilter=LinearMipmapLinear, generateMipmaps=true) when the
                    // sampler explicitly specifies one — an unspecified filter is
                    // glTF "auto", which those defaults already match.
                    if (samp.contains("magFilter")) {
                        tex->magFilter = samp["magFilter"].get<int>() == FILTER_NEAREST
                                                 ? Filter::Nearest
                                                 : Filter::Linear;
                    }
                    if (samp.contains("minFilter")) {
                        switch (samp["minFilter"].get<int>()) {
                            case FILTER_NEAREST:                tex->minFilter = Filter::Nearest; break;
                            case FILTER_LINEAR:                 tex->minFilter = Filter::Linear; break;
                            case FILTER_NEAREST_MIPMAP_NEAREST: tex->minFilter = Filter::NearestMipmapNearest; break;
                            case FILTER_LINEAR_MIPMAP_NEAREST:  tex->minFilter = Filter::LinearMipmapNearest; break;
                            case FILTER_NEAREST_MIPMAP_LINEAR:  tex->minFilter = Filter::NearestMipmapLinear; break;
                            case FILTER_LINEAR_MIPMAP_LINEAR:   tex->minFilter = Filter::LinearMipmapLinear; break;
                            default: break;
                        }
                        // A non-mipmap min filter samples only the base level, so
                        // mipmap generation is pointless (and the GL renderer keys
                        // its glGenerateMipmap on this flag).
                        const int mf = samp["minFilter"].get<int>();
                        tex->generateMipmaps =
                                (mf != FILTER_NEAREST && mf != FILTER_LINEAR);
                    }
                }

                textureCache[key] = tex;
                return tex;
            }

            // Load a texture and apply KHR_texture_transform (+ non-zero
            // texCoord) from a textureInfo JSON node. When no transform is
            // present and texCoord == 0 the shared base texture is returned
            // unchanged (no clone). Otherwise a transformed clone is produced
            // and cached by (texIdx, colorSpace, offset/scale/rotation/texCoord),
            // so identical transforms clone at most once instead of per slot.
            std::shared_ptr<Texture> applyTextureTransform(
                    const json& texInfo, int texIdx, ColorSpace cs = ColorSpace::sRGB) {
                auto tex = loadTexture(texIdx, cs);
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

                const std::tuple<int, int, float, float, float, float, float, int> key{
                        texIdx, static_cast<int>(cs), offX, offY, scX, scY, rot, texCoordVal};
                if (auto it = textureTransformCache.find(key); it != textureTransformCache.end())
                    return it->second;

                // Clone to avoid sharing transforms between channels
                auto clone = tex->clone();
                clone->offset = {offX, offY};
                clone->repeat = {scX, scY};
                clone->rotation = rot;
                clone->center = {0, 0};
                clone->texCoord = texCoordVal;
                clone->updateMatrix();
                textureTransformCache[key] = clone;
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
                            basicMat->map = applyTextureTransform(pbr["baseColorTexture"], ti);
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
                        ext.contains("KHR_materials_volume") ||
                        ext.contains("KHR_materials_iridescence")) {
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
                        mat->map = applyTextureTransform(pbr["baseColorTexture"], ti);
                    }

                    // Metalness / roughness
                    mat->metalness = pbr.value("metallicFactor", 1.0f);
                    mat->roughness = pbr.value("roughnessFactor", 1.0f);

                    // Metallic-roughness texture (G=roughness, B=metalness per spec)
                    if (pbr.contains("metallicRoughnessTexture")) {
                        int ti = pbr["metallicRoughnessTexture"]["index"].get<int>();
                        auto tex = applyTextureTransform(pbr["metallicRoughnessTexture"], ti, ColorSpace::Linear);
                        mat->metalnessMap = tex;
                        mat->roughnessMap = tex;
                    }
                }

                // Normal map
                if (matDef.contains("normalTexture")) {
                    int ti = matDef["normalTexture"]["index"].get<int>();
                    mat->normalMap = applyTextureTransform(matDef["normalTexture"], ti, ColorSpace::Linear);
                    float scale = matDef["normalTexture"].value("scale", 1.0f);
                    mat->normalScale = Vector2{scale, scale};
                }

                // Occlusion map
                if (matDef.contains("occlusionTexture")) {
                    int ti = matDef["occlusionTexture"]["index"].get<int>();
                    mat->aoMap = applyTextureTransform(matDef["occlusionTexture"], ti, ColorSpace::Linear);
                    mat->aoMapIntensity = matDef["occlusionTexture"].value("strength", 1.0f);
                }

                // Emissive
                if (matDef.contains("emissiveFactor")) {
                    auto e = matDef["emissiveFactor"].get<std::vector<float>>();
                    mat->emissive = Color(e[0], e[1], e[2]);
                }
                if (matDef.contains("emissiveTexture")) {
                    int ti = matDef["emissiveTexture"]["index"].get<int>();
                    mat->emissiveMap = applyTextureTransform(matDef["emissiveTexture"], ti);
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
                            physMat->transmissionMap = loadTexture(ti, ColorSpace::Linear);
                        }
                        // glTF: transmission WITHOUT KHR_materials_volume = a THIN-WALLED
                        // surface (infinitely thin, e.g. a watch crystal / car window).
                        // The volume block below resets this to false when volume is
                        // present (a closed solid). Without this the renderer treated
                        // every transmissive surface as solid → 2-interface refraction
                        // that warps/destroys whatever is just behind a thin pane.
                        physMat->thinWalled = true;
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
                        // A volume = a closed solid → 2-interface (closed-mesh) refraction,
                        // not the thin-shell path the transmission block assumed above.
                        physMat->thinWalled = (physMat->thickness <= 0.0f);
                    }

                    // KHR_materials_clearcoat
                    if (ext.contains("KHR_materials_clearcoat")) {
                        const auto& cc = ext["KHR_materials_clearcoat"];
                        physMat->clearcoat = cc.value("clearcoatFactor", 0.0f);
                        if (cc.contains("clearcoatTexture")) {
                            int ti = cc["clearcoatTexture"]["index"].get<int>();
                            physMat->clearcoatMap = loadTexture(ti, ColorSpace::Linear);
                        }
                        physMat->clearcoatRoughness = cc.value("clearcoatRoughnessFactor", 0.0f);
                        if (cc.contains("clearcoatRoughnessTexture")) {
                            int ti = cc["clearcoatRoughnessTexture"]["index"].get<int>();
                            physMat->clearcoatRoughnessMap = loadTexture(ti, ColorSpace::Linear);
                        }
                        if (cc.contains("clearcoatNormalTexture")) {
                            int ti = cc["clearcoatNormalTexture"]["index"].get<int>();
                            physMat->clearcoatNormalMap = loadTexture(ti, ColorSpace::Linear);
                            float scale = cc["clearcoatNormalTexture"].value("scale", 1.0f);
                            physMat->clearcoatNormalScale = Vector2{scale, scale};
                        }
                    }

                    // KHR_materials_iridescence
                    if (ext.contains("KHR_materials_iridescence")) {
                        const auto& ir = ext["KHR_materials_iridescence"];
                        physMat->iridescence = ir.value("iridescenceFactor", 0.0f);
                        physMat->iridescenceIOR = ir.value("iridescenceIor", 1.3f);
                        // Spec: thicknessMaximum is used as the constant thickness
                        // when no thickness texture is present (which we don't sample yet).
                        physMat->iridescenceThicknessNm = ir.value("iridescenceThicknessMaximum", 400.0f);
                    }
                }

                materialCache[matIdx] = mat;
                return mat;
            }

            // -----------------------------------------------------------------------
            //  Mesh
            // -----------------------------------------------------------------------

            // Decode one primitive's BufferGeometry (attributes, morph deltas,
            // index, computed normals). Cached by (meshIdx, primIdx, hasSkin) and
            // shared by every Mesh that references the primitive — the immutable
            // vertex data is decoded exactly once.
            std::shared_ptr<BufferGeometry> buildPrimitiveGeometry(
                    int meshIdx, int primIdx, const json& prim, bool hasSkin) {
                const std::tuple<int, int, bool> key{meshIdx, primIdx, hasSkin};
                if (auto it = geometryCache.find(key); it != geometryCache.end())
                    return it->second;

                auto geometry = BufferGeometry::create();
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
                // COLOR_0: use actual accessor component count (VEC3 or VEC4).
                // The glTF-recommended encodings are normalized uint8/uint16 —
                // keep those narrow (1/4 resp. 1/2 the float footprint) instead
                // of baking the normalization into widened floats. Both
                // renderers expand them on fetch/upload; the normalized flag
                // rides on the attribute.
                if (attrs.contains("COLOR_0")) {
                    int accIdx = attrs["COLOR_0"].get<int>();
                    auto [ptr, stride, count, ct, nc, accNormalized] = getAccessor(accIdx);
                    if (preserveNarrowAttributes && ct == COMP_UNSIGNED_BYTE && accNormalized) {
                        geometry->setAttribute("color",
                                Uint8BufferAttribute::create(readNarrow<uint8_t>(accIdx), nc, true));
                    } else if (preserveNarrowAttributes && ct == COMP_UNSIGNED_SHORT && accNormalized) {
                        geometry->setAttribute("color",
                                Uint16BufferAttribute::create(readNarrow<uint16_t>(accIdx), nc, true));
                    } else {
                        geometry->setAttribute("color",
                                FloatBufferAttribute::create(readFloats(accIdx), nc));
                    }
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

                // --- Morph targets (POSITION/NORMAL deltas) ---
                // glTF morph targets are always relative (deltas from the base attribute).
                if (prim.contains("targets")) {
                    const auto& targets = prim["targets"];
                    for (const auto& target : targets) {
                        if (target.contains("POSITION")) {
                            auto data = readFloats(target["POSITION"].get<int>());
                            geometry->getOrCreateMorphAttribute("position")
                                    ->emplace_back(FloatBufferAttribute::create(std::move(data), 3));
                        }
                        if (target.contains("NORMAL")) {
                            auto data = readFloats(target["NORMAL"].get<int>());
                            geometry->getOrCreateMorphAttribute("normal")
                                    ->emplace_back(FloatBufferAttribute::create(std::move(data), 3));
                        }
                    }
                    if (!targets.empty()) {
                        geometry->morphTargetsRelative = true;
                    }
                }

                // --- Indices ---
                if (prim.contains("indices")) {
                    auto indices = readIndices(prim["indices"].get<int>());
                    geometry->setIndex(std::move(indices));
                }

                // Compute vertex normals if absent
                if (!attrs.contains("NORMAL")) {
                    geometry->computeVertexNormals();
                }

                geometryCache[key] = geometry;
                return geometry;
            }

            // Exporters can emit primitives whose accessor references are
            // simply invalid: OpenCASCADE's RWGltf_CafWriter writes
            // {"attributes":{"POSITION":-1},"indices":-1} for faces it failed to
            // triangulate. Decoding one aborts the whole document, so a 165 MB
            // assembly is lost over a single degenerate face. Validate every
            // index a primitive references up front and skip just that
            // primitive, keeping the rest of the mesh.
            bool primitiveAccessorsValid(int meshIdx, const std::string& meshName,
                                         int primIdx, const json& prim) const {
                auto reject = [&](const std::string& why) {
                    std::cerr << "GLTFLoader: skipping primitive " << primIdx
                              << " of mesh " << meshIdx;
                    if (!meshName.empty()) std::cerr << " ('" << meshName << "')";
                    std::cerr << " - " << why << std::endl;
                    return false;
                };

                if (!prim.contains("attributes") || !prim["attributes"].is_object())
                    return reject("no attributes");

                for (auto it = prim["attributes"].begin(); it != prim["attributes"].end(); ++it) {
                    if (!accessorIndexValid(it.value()))
                        return reject("attribute " + it.key() + " references invalid accessor " +
                                      it.value().dump());
                }

                if (prim.contains("indices") && !accessorIndexValid(prim["indices"]))
                    return reject("indices reference invalid accessor " + prim["indices"].dump());

                if (prim.contains("targets")) {
                    for (const auto& target : prim["targets"]) {
                        if (!target.is_object()) continue;
                        for (auto it = target.begin(); it != target.end(); ++it) {
                            if (!accessorIndexValid(it.value()))
                                return reject("morph target " + it.key() +
                                              " references invalid accessor " + it.value().dump());
                        }
                    }
                }

                return true;
            }

            std::shared_ptr<Object3D> loadMesh(int meshIdx, bool hasSkin = false) {
                const auto& meshDef = gltf["meshes"][meshIdx];
                const auto& primitives = meshDef["primitives"];

                const std::string meshName = meshDef.value("name", "");
                std::vector<std::shared_ptr<Mesh>> meshes;

                int primIdx = 0;
                for (const auto& prim : primitives) {
                    // glTF "mode" (default 4/TRIANGLES if absent): POINTS(0),
                    // LINES(1), LINE_LOOP(2), LINE_STRIP(3), TRIANGLES(4),
                    // TRIANGLE_STRIP(5), TRIANGLE_FAN(6). The renderers here
                    // only understand flat triangle lists (raster draw calls
                    // and the Vulkan RT BLAS both assume indexCount/vertexCount
                    // is a triangle count) — feeding them a point cloud or line
                    // strip mis-decodes as garbage triangles and, for vertex
                    // counts not a multiple of 3, previously overran the
                    // position buffer in computeVertexNormals(). Skip
                    // unsupported topologies rather than misrender them.
                    const int primMode = prim.value("mode", 4);
                    if (primMode != 4) {
                        std::cerr << "GLTFLoader: skipping primitive " << primIdx
                                  << " of mesh " << meshIdx << " - unsupported mode "
                                  << primMode << " (only TRIANGLES is supported)" << std::endl;
                        ++primIdx;
                        continue;
                    }

                    if (!primitiveAccessorsValid(meshIdx, meshName, primIdx, prim)) {
                        ++primIdx;
                        continue;
                    }

                    // Shared, cached geometry (decoded once per mesh/prim/skin).
                    auto geometry = buildPrimitiveGeometry(meshIdx, primIdx, prim, hasSkin);
                    const auto& attrs = prim["attributes"];

                    // --- Material (cached by matIdx) ---
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

                    // Morph influences are per-Mesh: the morph delta attributes
                    // live on the shared geometry; the weights selecting them do
                    // not. Count comes from the primitive's target list.
                    const size_t numMorphTargets =
                            prim.contains("targets") ? prim["targets"].size() : 0;
                    if (numMorphTargets > 0) {
                        auto& influences = mesh->morphTargetInfluences();
                        influences.assign(numMorphTargets, 0.0f);
                        // Initial weights live on the mesh, not the primitive.
                        if (meshDef.contains("weights")) {
                            const auto& weights = meshDef["weights"];
                            const size_t n = std::min(numMorphTargets, weights.size());
                            for (size_t i = 0; i < n; ++i) {
                                influences[i] = weights[i].get<float>();
                            }
                        }
                    }

                    // Tag for KHR_materials_variants post-load resolution (per-Mesh)
                    mesh->userData["__gltfMeshIdx"] = meshIdx;
                    mesh->userData["__gltfPrimIdx"] = primIdx;

                    // Collect per-primitive variant mappings exactly once per
                    // (mesh, prim): loadMesh may run for several referencing nodes,
                    // and the mappings are appended — the guard prevents duplicates.
                    if (!variantNames.empty() &&
                        prim.contains("extensions") &&
                        prim["extensions"].contains("KHR_materials_variants") &&
                        variantsCollected.insert({meshIdx, primIdx}).second) {
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

                    meshes.push_back(mesh);
                    ++primIdx;
                }

                // Single non-skinned primitive → return the Mesh directly (named
                // after the mesh), skipping a redundant Group wrapper + clone (the
                // old path cloned only because it built the Group first). Skinned
                // meshes stay wrapped in a Group so buildMeshObjForNode can bind
                // each SkinnedMesh child to the skeleton.
                if (meshes.size() == 1 && !hasSkin) {
                    meshes[0]->name = meshName;
                    return meshes[0];
                }

                auto group = Group::create();
                group->name = meshName;
                for (auto& m : meshes) group->add(m);
                return group;
            }

            // -----------------------------------------------------------------------
            //  Node / Scene hierarchy
            // -----------------------------------------------------------------------

            // EXT_mesh_gpu_instancing — node-level extension carrying per-instance
            // TRANSLATION (vec3) / ROTATION (vec4 quat xyzw) / SCALE (vec3) accessor
            // arrays. Any subset may be present; missing components default to
            // identity. All provided accessors must share the same count.
            //
            // We walk `meshObj` (single Mesh or Group of Meshes from loadMesh) and
            // replace each non-skinned Mesh with an InstancedMesh sharing its
            // geometry + material. Per-instance matrices are composed from the TRS
            // accessors and uploaded into instanceMatrix. The replacement preserves
            // mesh name + userData so variant resolution and friends keep working.
            std::shared_ptr<Object3D> applyGpuInstancing(const json& extData,
                                                        const std::shared_ptr<Object3D>& meshObj) {
                if (!extData.contains("attributes") || !meshObj) return meshObj;
                const auto& attrs = extData["attributes"];

                // Count comes from any provided accessor (spec: all must match).
                size_t count = 0;
                for (const char* key : {"TRANSLATION", "ROTATION", "SCALE"}) {
                    if (attrs.contains(key)) {
                        count = gltf["accessors"][attrs[key].get<int>()]["count"].get<size_t>();
                        break;
                    }
                }
                if (count == 0) return meshObj;

                const std::vector<float> t = attrs.contains("TRANSLATION")
                        ? readFloats(attrs["TRANSLATION"].get<int>()) : std::vector<float>();
                const std::vector<float> r = attrs.contains("ROTATION")
                        ? readFloats(attrs["ROTATION"].get<int>()) : std::vector<float>();
                const std::vector<float> s = attrs.contains("SCALE")
                        ? readFloats(attrs["SCALE"].get<int>()) : std::vector<float>();

                std::vector<Matrix4> mats(count);
                for (size_t i = 0; i < count; ++i) {
                    Vector3 tv(0, 0, 0), sv(1, 1, 1);
                    Quaternion qv;
                    if (!t.empty()) tv.set(t[i * 3], t[i * 3 + 1], t[i * 3 + 2]);
                    if (!r.empty()) qv.set(r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]);
                    if (!s.empty()) sv.set(s[i * 3], s[i * 3 + 1], s[i * 3 + 2]);
                    mats[i].compose(tv, qv, sv);
                }

                auto convert = [&](Mesh& m) -> std::shared_ptr<InstancedMesh> {
                    // Skinned meshes can't reasonably share a single pose buffer
                    // across instances; leave them as the regular Mesh.
                    if (dynamic_cast<SkinnedMesh*>(&m)) return nullptr;
                    auto inst = InstancedMesh::create(m.geometry(), m.material(), count);
                    inst->name = m.name;
                    inst->userData = m.userData;
                    for (size_t i = 0; i < count; ++i) inst->setMatrixAt(i, mats[i]);
                    inst->instanceMatrix()->needsUpdate();
                    inst->computeBoundingSphere();
                    return inst;
                };

                // Single-Mesh case (loadMesh unwrapped a single-primitive mesh).
                // Mesh and Group are siblings under Object3D, so as<Mesh>() is
                // null on a Group.
                if (auto* m = meshObj->as<Mesh>()) {
                    if (auto inst = convert(*m)) return inst;
                    return meshObj;
                }

                // Group-of-Meshes case: rebuild the group with InstancedMesh children.
                auto group = Group::create();
                group->name = meshObj->name;
                group->userData = meshObj->userData;
                for (auto* child : meshObj->children) {
                    if (auto* cm = child->as<Mesh>()) {
                        if (auto inst = convert(*cm)) {
                            group->add(inst);
                            continue;
                        }
                    }
                    // Non-Mesh / SkinnedMesh: keep as-is, share the existing pointer.
                    // children stores raw pointers; promote via shared_from_this if
                    // available, otherwise we can't reattach safely — fall back to a
                    // clone so we don't leave dangling refs.
                    group->add(child->clone());
                }
                return group;
            }

            // Build the Mesh / multi-prim Group for a node (with skin + instancing
            // applied). Caller decides whether to attach it as a child of the node's
            // wrapper (joint case) or replace the wrapper outright (collapse case).
            std::shared_ptr<Object3D> buildMeshObjForNode(int nodeIdx) {
                const auto& nodeDef = gltf["nodes"][nodeIdx];
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

                // EXT_mesh_gpu_instancing — replace Mesh children with
                // InstancedMesh driven by the extension's per-instance TRS
                // accessor arrays. Skipped for skinned meshes (spec advises
                // against combining with KHR_skin).
                if (!hasSkin && nodeDef.contains("extensions") &&
                    nodeDef["extensions"].contains("EXT_mesh_gpu_instancing")) {
                    meshObj = applyGpuInstancing(
                            nodeDef["extensions"]["EXT_mesh_gpu_instancing"], meshObj);
                }

                return meshObj;
            }

            // For non-joint nodes with a mesh, replace the pre-created Group
            // wrapper with the actual meshObj — transferring transform + name.
            // This eliminates a redundant Object3D layer per mesh node
            // (Group("X") → Mesh("X") collapses to a single Mesh("X")).
            //
            // Joint nodes (Bones) keep their wrapper so skin binding / skeleton
            // wiring remains intact.
            void tryCollapseMeshWrapper(int nodeIdx) {
                if (collapsedMeshNodes.count(nodeIdx)) return;
                const auto& nodeDef = gltf["nodes"][nodeIdx];
                if (!nodeDef.contains("mesh") || !gltf.contains("meshes")) return;
                if (jointNodeSet.count(nodeIdx)) return;

                auto meshObj = buildMeshObjForNode(nodeIdx);

                auto& wrapper = nodeObjects[nodeIdx];
                meshObj->position.copy(wrapper->position);
                meshObj->quaternion.copy(wrapper->quaternion);
                meshObj->scale.copy(wrapper->scale);
                // Adopt wrapper's name (set in preCreateNodes from the node's
                // glTF name or the synthetic "node_N" fallback). Animation
                // tracks reference nodes by this name — see loadAnimations.
                meshObj->name = wrapper->name;

                // Multi-primitive meshes (Group containing several Meshes):
                // propagate the node's name to inner primitives so traversal
                // by name still finds them.
                if (meshObj->children.size() > 1) {
                    int idx = 0;
                    for (auto* child : meshObj->children) {
                        child->name = wrapper->name + "_" + std::to_string(idx++);
                    }
                }

                nodeObjects[nodeIdx] = meshObj;
                collapsedMeshNodes.insert(nodeIdx);
            }

            // Build node hierarchy using pre-created node objects
            void buildNode(int nodeIdx) {
                if (!builtNodes.insert(nodeIdx).second) return; // already built
                const auto& nodeDef = gltf["nodes"][nodeIdx];
                auto& obj = nodeObjects[nodeIdx];

                // Mesh attachment is only needed here for joints (Bones) that
                // also carry a mesh — non-joint mesh nodes were already
                // collapsed (their meshObj IS nodeObjects[nodeIdx]).
                if (!collapsedMeshNodes.count(nodeIdx) &&
                    nodeDef.contains("mesh") && gltf.contains("meshes")) {

                    auto meshObj = buildMeshObjForNode(nodeIdx);

                    // DCC tools (Blender, Maya, ...) put the user-facing object name on
                    // the glTF node; the mesh name is the mesh-data name and is often
                    // generic ("Object_0"). When the node has an explicit name, prefer
                    // it on the mesh container so traversal-by-name finds Blender names.
                    if (nodeDef.contains("name")) {
                        const auto nodeName = nodeDef["name"].get<std::string>();
                        meshObj->name = nodeName;
                        // For multi-primitive meshes (Group containing several Meshes),
                        // also propagate the name to inner primitives so they're findable.
                        if (!meshObj->children.empty()) {
                            int idx = 0;
                            for (auto* child : meshObj->children) {
                                child->name = nodeName + "_" + std::to_string(idx++);
                            }
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
                            // Cone angles live in the nested "spot" object (KHR_lights_punctual §spot)
                            const json spotDef = lightDef.value("spot", json::object());
                            float innerCone = spotDef.value("innerConeAngle", 0.0f);
                            float outerCone = spotDef.value("outerConeAngle", math::PI / 4.f);
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
                    // Only nodes reachable from this scene's roots are built, so
                    // meshes of unreachable nodes are never decoded. Ascending
                    // order matches the original full 0..n iteration for the
                    // reachable subset.
                    const std::vector<int> reachable = collectReachable(sceneDef);

                    // Pass 1: collapse Group-wrapper-around-Mesh layers for
                    // every non-joint mesh node. Must run before buildNode's
                    // child-attach so parent nodes pick up the collapsed
                    // (Mesh) child, not the discarded Group wrapper.
                    for (int i : reachable) tryCollapseMeshWrapper(i);

                    // Pass 2: attach lights / joint-meshes / children.
                    for (int i : reachable) buildNode(i);

                    for (int nodeIdx : sceneDef["nodes"].get<std::vector<int>>())
                        root->add(nodeObjects[nodeIdx]);
                }

                // Sketchfab/Blender exports often wrap a Mesh in a chain of named
                // groups (e.g. "Cone_2" → "Object_8" → Mesh), where only the topmost
                // wrapper carries the user-facing name. Walk up from each Mesh through
                // single-child wrapper groups and adopt the topmost wrapper's name so
                // traverseType<Mesh> / getObjectByName see Blender's names.
                //
                // Stop at the scene root: its name is the scene's name ("Scene" fallback)
                // and shouldn't propagate onto a single-child mesh.
                Object3D* rootPtr = root.get();
                root->traverseType<Mesh>([rootPtr](Mesh& m) {
                    Object3D* candidate = &m;
                    Object3D* cur = &m;
                    while (cur->parent && cur->parent != rootPtr &&
                           cur->parent->children.size() == 1 &&
                           !dynamic_cast<Mesh*>(cur->parent)) {
                        candidate = cur->parent;
                        cur = cur->parent;
                    }
                    if (candidate != &m && !candidate->name.empty()) {
                        m.name = candidate->name;
                    }
                });

                return root;
            }

            // -----------------------------------------------------------------------
            //  Entry points
            // -----------------------------------------------------------------------

            // Shared driver: assumes `gltf` is parsed and `buffers` is sized.
            // Both entry points funnel here so the scene/animation/variant
            // assembly lives in one place.
            GLTFResult buildResult() {
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
                    // Independent scenes: re-instantiate node objects for each
                    // scene after the first so a node referenced by two scenes
                    // isn't reparented (stolen) from the earlier scene. Scene 0
                    // uses the initial preCreateNodes(); the single-scene path
                    // therefore never resets and is byte-identical to before.
                    if (numScenes > 1 && i > 0) resetSceneBuildState();
                    result.scenes.push_back(loadScene(i));
                }

                if (!result.scenes.empty()) {
                    int si = (defaultScene >= 0 && defaultScene < numScenes) ? defaultScene : 0;
                    result.scene = result.scenes[si];
                } else {
                    result.scene = Group::create();
                }

                result.animations = loadAnimations();
                for (auto& [_, proxy] : matAnimProxies) {
                    if (proxy && result.scene) result.scene->add(proxy);
                }
                resolveVariants(result);
                return result;
            }

            GLTFResult parseGLTF(const std::string& jsonText) {
                gltf = json::parse(jsonText);
                int numBuffers = gltf.contains("buffers") ? static_cast<int>(gltf["buffers"].size()) : 0;
                buffers.resize(numBuffers);
                return buildResult();
            }

            GLTFResult parseGLB(const std::vector<uint8_t>& data) {
                if (data.size() < 12) throw std::runtime_error("GLB too small");

                uint32_t magic, version, totalLength;
                std::memcpy(&magic, data.data(), 4);
                std::memcpy(&version, data.data() + 4, 4);
                std::memcpy(&totalLength, data.data() + 8, 4);

                if (magic != GLB_MAGIC) throw std::runtime_error("Not a GLB file (bad magic)");

                // The declared container length must not claim more bytes than we
                // actually have; a truncated GLB otherwise reads past the buffer.
                if (totalLength > data.size())
                    throw std::runtime_error("GLB truncated: header length " +
                                             std::to_string(totalLength) + " exceeds file size " +
                                             std::to_string(data.size()));
                // Never scan past the declared length even if the file has trailing bytes.
                const size_t end = totalLength >= 12 ? totalLength : data.size();

                size_t offset = 12;
                std::string jsonText;
                bool gotJSON = false, gotBIN = false;

                while (offset + 8 <= end) {
                    uint32_t chunkLen, chunkType;
                    std::memcpy(&chunkLen, data.data() + offset, 4);
                    std::memcpy(&chunkType, data.data() + offset + 4, 4);
                    offset += 8;

                    // Each chunk's declared payload must fit within the container.
                    if (chunkLen > end - offset)
                        throw std::runtime_error("GLB chunk overruns file (offset " +
                                                 std::to_string(offset) + ", len " +
                                                 std::to_string(chunkLen) + ")");

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
                return buildResult();
            }

            // -----------------------------------------------------------------------
            //  Animation
            // -----------------------------------------------------------------------

            std::vector<std::shared_ptr<AnimationClip>> loadAnimations() {
                if (!gltf.contains("animations")) return {};

                std::vector<std::shared_ptr<AnimationClip>> clips;

                // Memoize decoded float accessors: animation channels frequently
                // share a single input (time) accessor across many samplers, so
                // decoding it once avoids repeated buffer walks. References into
                // an unordered_map stay valid across later insertions.
                std::unordered_map<int, std::vector<float>> accCache;
                auto cachedAccessor = [&](int accIdx) -> const std::vector<float>& {
                    auto it = accCache.find(accIdx);
                    if (it != accCache.end()) return it->second;
                    return accCache.emplace(accIdx, readFloats(accIdx)).first->second;
                };

                for (size_t animIdx = 0; animIdx < gltf["animations"].size(); ++animIdx) {
                    const auto& animDef = gltf["animations"][animIdx];

                    std::string animName = animDef.value("name", "animation_" + std::to_string(animIdx));

                    if (!animDef.contains("channels") || !animDef.contains("samplers")) continue;

                    const auto& channels = animDef["channels"];
                    const auto& samplers = animDef["samplers"];

                    // Blender's glTF exporter (export_force_sampling) samples
                    // starting at frame 1, not frame 0, so every track's first
                    // keyframe lands at 1/fps instead of 0 - a small dead zone
                    // where AnimationAction::_updateTime's local clip time is
                    // still "before the first keyframe". Interpolant::evaluate
                    // correctly clamps there, but it can return the SAME clamped
                    // sample for 2+ consecutive frames right after a Loop::Repeat
                    // wrap (local time briefly revisits that zone), which makes
                    // PropertyMixer::apply's change-detection (comparing the two
                    // ping-ponged accumulator buffers) see "no change" and skip
                    // writing the property that frame - visible as a pose glitch
                    // whenever other code (e.g. a root-motion pin, an aim-tilt
                    // premultiply) mutates that same bone between mixer updates.
                    // Normalizing every track to start at t=0 removes the dead
                    // zone entirely.
                    float clipMinStart = std::numeric_limits<float>::infinity();
                    for (const auto& channel : channels) {
                        if (!channel.contains("sampler") || !channel.contains("target")) continue;
                        int samplerIdx = channel.value("sampler", -1);
                        if (samplerIdx < 0 || samplerIdx >= static_cast<int>(samplers.size())) continue;
                        int inputAccIdx = samplers[samplerIdx].value("input", -1);
                        if (inputAccIdx < 0) continue;
                        const auto& t = cachedAccessor(inputAccIdx);
                        if (!t.empty()) clipMinStart = std::min(clipMinStart, t[0]);
                    }
                    if (!std::isfinite(clipMinStart)) clipMinStart = 0.f;

                    std::vector<std::shared_ptr<KeyframeTrack>> tracks;

                    for (const auto& channel : channels) {
                        if (!channel.contains("sampler") || !channel.contains("target")) continue;

                        const auto& target = channel["target"];
                        int nodeIdx = target.value("node", -1);
                        std::string path = target.value("path", "");

                        // KHR_animation_pointer lives on target.extensions and
                        // targets materials/cameras/etc. instead of a node.
                        std::string ptr;
                        if (path == "pointer" && target.contains("extensions") &&
                            target["extensions"].contains("KHR_animation_pointer")) {
                            ptr = target["extensions"]["KHR_animation_pointer"].value("pointer", "");
                        }

                        if (ptr.empty() && (nodeIdx < 0 || path.empty())) continue;

                        int samplerIdx = channel["sampler"].get<int>();
                        if (samplerIdx < 0 || samplerIdx >= static_cast<int>(samplers.size())) continue;

                        const auto& samplerDef = samplers[samplerIdx];
                        int inputAccIdx = samplerDef["input"].get<int>();
                        int outputAccIdx = samplerDef["output"].get<int>();
                        std::string interpolation = samplerDef.value("interpolation", "LINEAR");

                        // Values are copied because CUBICSPLINE stripping and the
                        // weights split below mutate them. Times are copied (off
                        // the cache) too, so the clipMinStart shift below never
                        // mutates the shared accessor cache.
                        const std::vector<float>& rawTimes = cachedAccessor(inputAccIdx);
                        std::vector<float> values = cachedAccessor(outputAccIdx);

                        if (rawTimes.empty()) continue;

                        std::vector<float> times = rawTimes;
                        if (clipMinStart > 0.f) {
                            for (auto& tt : times) tt -= clipMinStart;
                        }

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

                        if (!ptr.empty()) {
                            // Parse /materials/N/... pointers. Anything else
                            // (cameras, lights, extensions on other types) is
                            // unsupported and silently skipped.
                            const std::string matPrefix = "/materials/";
                            if (ptr.rfind(matPrefix, 0) != 0) continue;
                            size_t slash = ptr.find('/', matPrefix.size());
                            if (slash == std::string::npos) continue;
                            int matIdx;
                            try { matIdx = std::stoi(ptr.substr(matPrefix.size(), slash - matPrefix.size())); }
                            catch (...) { continue; }
                            std::string tail = ptr.substr(slash + 1);

                            // Map glTF pointer tail -> our material property name + expected component count
                            std::string propName;
                            int expected = 0;
                            if (tail == "pbrMetallicRoughness/baseColorFactor") { propName = "baseColorFactor"; expected = 4; }
                            else if (tail == "pbrMetallicRoughness/metallicFactor")  { propName = "metalness"; expected = 1; }
                            else if (tail == "pbrMetallicRoughness/roughnessFactor") { propName = "roughness"; expected = 1; }
                            else if (tail == "emissiveFactor")                       { propName = "emissive"; expected = 3; }
                            else if (tail == "alphaCutoff")                          { propName = "alphaTest"; expected = 1; }
                            else {
                                // KHR_texture_transform: rotation/offset/scale on a per-slot
                                // texture transform. The tail looks like
                                //   <slot>/extensions/KHR_texture_transform/{rotation|offset|scale}
                                // where <slot> is the glTF texture path (e.g. "normalTexture",
                                // "pbrMetallicRoughness/baseColorTexture", or
                                // "extensions/KHR_materials_volume/thicknessTexture").
                                static const std::string kTT = "/extensions/KHR_texture_transform/";
                                size_t ttPos = tail.find(kTT);
                                if (ttPos == std::string::npos) continue;
                                const std::string slotPath = tail.substr(0, ttPos);
                                const std::string ttProp   = tail.substr(ttPos + kTT.size());

                                // glTF slot path → threepp Material texture field name.
                                // Sheen / specular / iridescence / anisotropy slot textures
                                // aren't carried on threepp's material interfaces, so they're
                                // silently ignored here.
                                std::string field;
                                if (slotPath == "normalTexture") field = "normalMap";
                                else if (slotPath == "occlusionTexture") field = "aoMap";
                                else if (slotPath == "emissiveTexture") field = "emissiveMap";
                                else if (slotPath == "pbrMetallicRoughness/baseColorTexture") field = "map";
                                else if (slotPath == "pbrMetallicRoughness/metallicRoughnessTexture") field = "metalnessMap";
                                else if (slotPath == "extensions/KHR_materials_transmission/transmissionTexture") field = "transmissionMap";
                                else if (slotPath == "extensions/KHR_materials_volume/thicknessTexture") field = "thicknessMap";
                                else if (slotPath == "extensions/KHR_materials_clearcoat/clearcoatTexture") field = "clearcoatMap";
                                else if (slotPath == "extensions/KHR_materials_clearcoat/clearcoatRoughnessTexture") field = "clearcoatRoughnessMap";
                                else if (slotPath == "extensions/KHR_materials_clearcoat/clearcoatNormalTexture") field = "clearcoatNormalMap";
                                if (field.empty()) continue;

                                // Use '/' as the inner separator: PropertyBinding splits
                                // the track name on the *last* '.' to find nodeName vs
                                // property, so any '.' inside the property would route
                                // the setter to Object3D::rotation (read-only Euler) and
                                // throw "rotation is not writable".
                                if (ttProp == "rotation")    { propName = "tex/" + field + "/rotation"; expected = 1; }
                                else if (ttProp == "offset") { propName = "tex/" + field + "/offset";   expected = 2; }
                                else if (ttProp == "scale")  { propName = "tex/" + field + "/scale";    expected = 2; }
                                else continue;
                            }

                            auto mat = loadMaterial(matIdx);
                            if (!mat) continue;

                            auto& proxy = matAnimProxies[matIdx];
                            if (!proxy) {
                                proxy = MaterialAnimationProxy::create(mat);
                                proxy->name = "__matAnim_" + std::to_string(matIdx);
                            }

                            const std::string trackName = proxy->name + "." + propName;
                            if (expected == 1) {
                                track = std::make_shared<NumberKeyframeTrack>(trackName, times, values, interp);
                            } else {
                                // VectorKeyframeTrack handles any component size;
                                // our material setter reads the expected count.
                                track = std::make_shared<VectorKeyframeTrack>(trackName, times, values, interp);
                            }
                            if (track) tracks.push_back(track);
                            continue;
                        }

                        if (path == "translation") {
                            track = std::make_shared<VectorKeyframeTrack>(
                                    nodeName + ".position", times, values, interp);
                        } else if (path == "rotation") {
                            // Pass interp here too. glTF STEP rotations were
                            // silently played as slerp, so a "Step Rotation"
                            // animation swept smoothly instead of snapping
                            // between keys. QuaternionKeyframeTrack accepts
                            // Discrete and Linear and coerces Smooth to Linear,
                            // since a cubic would denormalise the quaternion.
                            track = std::make_shared<QuaternionKeyframeTrack>(
                                    nodeName + ".quaternion", times, values, interp);
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
            std::vector<uint8_t> data = readAllBytes(f, path);

            GLTFParser parser;
            parser.basePath = path.parent_path();
            parser.buffers = {};
            parser.preserveNarrowAttributes = preserveNarrowAttributes;

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
