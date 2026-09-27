// glTF 2.0 loader for threepp

#include "threepp/loaders/GLTFLoader.hpp"

#include "gltf/GLTFParser.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace threepp::gltf {

    const std::vector<uint8_t>& GLTFParser::resolveBuffer(int idx) {
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

    // Decode a meshopt-compressed bufferView on first access and cache it.
    const std::vector<uint8_t>& GLTFParser::decodeMeshoptBV(int bvIdx) {
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

    GLTFParser::AccessorData GLTFParser::getAccessor(int accessorIdx) {
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
    const uint8_t* GLTFParser::plainBufferViewPtr(int bvIdx, size_t extraOffset, size_t neededBytes) {
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

    // Read accessor into a flat float vector, honouring `normalized` and
    // applying any sparse overlay. A sparse accessor with no base
    // bufferView starts zero-filled.
    std::vector<float> GLTFParser::readFloats(int accessorIdx) {
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

    std::vector<uint32_t> GLTFParser::readIndices(int accessorIdx) {
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
    std::vector<float> GLTFParser::readJointIndicesAsFloat(int accessorIdx) {
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

    void GLTFParser::gatherJoints() {
        if (!gltf.contains("skins")) return;
        for (const auto& skin : gltf["skins"]) {
            if (!skin.contains("joints")) continue;
            for (int ji : skin["joints"].get<std::vector<int>>())
                jointNodeSet.insert(ji);
        }
    }

    void GLTFParser::applyNodeTransform(const std::shared_ptr<Object3D>& obj, const json& nodeDef) {
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
    void GLTFParser::preCreateNodes() {
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
    std::vector<int> GLTFParser::collectReachable(const json& sceneDef) {
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
    void GLTFParser::resetSceneBuildState() {
        nodeObjects.clear();
        builtNodes.clear();
        collapsedMeshNodes.clear();
        skinCache.clear();
        preCreateNodes();
    }

    std::shared_ptr<Skeleton> GLTFParser::loadSkin(int skinIdx) {
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

    // Shared driver: assumes `gltf` is parsed and `buffers` is sized.
    // Both entry points funnel here so the scene/animation/variant
    // assembly lives in one place.
    GLTFResult GLTFParser::buildResult() {
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

    GLTFResult GLTFParser::parseGLTF(const std::string& jsonText) {
        gltf = json::parse(jsonText);
        int numBuffers = gltf.contains("buffers") ? static_cast<int>(gltf["buffers"].size()) : 0;
        buffers.resize(numBuffers);
        return buildResult();
    }

    GLTFResult GLTFParser::parseGLB(const std::vector<uint8_t>& data) {
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

    std::vector<std::shared_ptr<AnimationClip>> GLTFParser::loadAnimations() {
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

    void GLTFParser::resolveVariants(GLTFResult& result) {
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

}// namespace threepp::gltf

namespace threepp {

    using gltf::GLB_MAGIC;
    using gltf::GLTFParser;
    using gltf::readAllBytes;

    namespace {

        // Shared by both public overloads. Throws, so each caller can report the
        // failure in its own words (a file load names the file). GLB is told
        // apart from JSON by its magic rather than by a file extension, which
        // in-memory data does not have.
        GLTFResult parseBytes(const std::vector<uint8_t>& data, const fs::path& basePath,
                              bool preserveNarrowAttributes) {
            GLTFParser parser;
            parser.basePath = basePath;
            parser.buffers = {};
            parser.preserveNarrowAttributes = preserveNarrowAttributes;

            uint32_t magic = 0;
            if (data.size() >= sizeof(magic)) std::memcpy(&magic, data.data(), sizeof(magic));
            if (magic == GLB_MAGIC) {
                return parser.parseGLB(data);
            }

            // .gltf — plain JSON
            std::string jsonText(data.begin(), data.end());
            return parser.parseGLTF(jsonText);
        }

    }// anonymous namespace

    // ===========================================================================
    //  GLTFLoader public API
    // ===========================================================================

    std::optional<GLTFResult> GLTFLoader::load(const std::vector<uint8_t>& data, const fs::path& basePath) {
        try {
            return parseBytes(data, basePath, preserveNarrowAttributes);
        } catch (const std::exception& e) {
            std::cerr << "[GLTFLoader] Error loading from memory: " << e.what() << "\n";
            return std::nullopt;
        }
    }

    std::optional<GLTFResult> GLTFLoader::load(const fs::path& path) {
        try {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open file: " + path.string());
            std::vector<uint8_t> data = readAllBytes(f, path);

            return parseBytes(data, path.parent_path(), preserveNarrowAttributes);
        } catch (const std::exception& e) {
            std::cerr << "[GLTFLoader] Error loading " << path << ": " << e.what() << "\n";
            return std::nullopt;
        }
    }

}// namespace threepp
