
#include "threepp/utils/BufferGeometryUtils.hpp"

#include "threepp/core/AttributeView.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#ifdef THREEPP_WITH_VULKAN
// DisplacedMesh.cpp is only compiled into Vulkan builds (its typeinfo lives
// there), and no DisplacedMesh can exist without the Vulkan renderer.
#include "threepp/objects/DisplacedMesh.hpp"
#endif

#include <meshoptimizer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

using namespace threepp;

namespace {

    template<typename T>
    inline std::unique_ptr<BufferAttribute> mergeBufferAttributes(const std::vector<TypedBufferAttribute<T>*>& attributes) {

        std::optional<int> itemSize;
        std::optional<bool> normalized;
        size_t arrayLength = 0;

        for (unsigned i = 0; i < attributes.size(); ++i) {

            TypedBufferAttribute<T>* attribute = attributes[i];

            if (!itemSize) itemSize = attribute->itemSize();
            if (itemSize.value() != attribute->itemSize()) {

                std::cerr << "THREE.BufferGeometryUtils: .mergeBufferAttributes() failed. BufferAttribute.itemSize must be consistent across matching attributes." << std::endl;
                return nullptr;
            }

            if (!normalized) normalized = attribute->normalized();
            if (normalized.value() != attribute->normalized()) {

                std::cerr << "THREE.BufferGeometryUtils: .mergeBufferAttributes() failed. BufferAttribute.normalized must be consistent across matching attributes." << std::endl;
                return nullptr;
            }

            arrayLength += attribute->array().size();
        }


        std::vector<T> array(arrayLength);
        unsigned int offset = 0;

        for (unsigned i = 0; i < attributes.size(); ++i) {

            auto& arr = attributes[i]->array();

            for (unsigned j = 0; j < arr.size(); j++) {
                array[offset + j] = arr[j];
            }

            offset += arr.size();
        }

        return TypedBufferAttribute<T>::create(array, itemSize.value(), normalized.value());
    }

    // Merge `attr` as scalar type T if that is the first attribute's type.
    // Returns false when T doesn't match (caller tries the next type); returns
    // true with `out == nullptr` when the types are mixed across geometries.
    template<typename T>
    bool mergeAs(const std::vector<BufferAttribute*>& attr, std::unique_ptr<BufferAttribute>& out) {
        if (!attr.front()->typed<T>()) return false;
        std::vector<TypedBufferAttribute<T>*> typed;
        typed.reserve(attr.size());
        for (auto* a : attr) {
            auto* t = a->typed<T>();
            if (!t) {
                out = nullptr;
                return true;
            }
            typed.emplace_back(t);
        }
        out = mergeBufferAttributes<T>(typed);
        return true;
    }

}// namespace

std::shared_ptr<BufferGeometry> threepp::mergeBufferGeometries(const std::vector<BufferGeometry*>& geometries, bool useGroups) {

    if (geometries.empty()) return nullptr;

    bool isIndexed = geometries[0]->hasIndex();

    auto& attributesUsed = geometries[0]->getAttributes();


    std::unordered_map<std::string, std::vector<BufferAttribute*>> attributes;

    auto mergedGeometry = std::make_shared<BufferGeometry>();

    unsigned int offset = 0;

    for (unsigned i = 0; i < geometries.size(); ++i) {

        auto geometry = geometries[i];
        unsigned int attributesCount = 0;

        // ensure that all geometries are indexed, or none

        if (isIndexed != (geometry->hasIndex())) {

            std::cerr << "THREE.BufferGeometryUtils: .mergeBufferGeometries() failed with geometry at index " << i << ". All geometries must have compatible attributes; make sure index attribute exists among all geometries, or in none of them." << std::endl;
            return nullptr;
        }

        // gather attributes, exit early if they're different

        for (auto& [name, attr] : geometry->getAttributes()) {

            if (!attributesUsed.contains(name)) {

                std::cerr << "THREE.BufferGeometryUtils: .mergeBufferGeometries() failed with geometry at index " << i << ". All geometries must have compatible attributes; make sure \"" + name + "\" attribute exists among all geometries, or in none of them." << std::endl;
                return nullptr;
            }

            attributes[name].emplace_back(attr.get());

            attributesCount++;
        }

        // ensure geometries have the same number of attributes

        if (attributesCount != attributesUsed.size()) {

            std::cout << "THREE.BufferGeometryUtils: .mergeBufferGeometries() failed with geometry at index " << i << ". Make sure all geometries have the same number of attributes." << std::endl;
            return nullptr;
        }

        if (useGroups) {

            unsigned int count = 0;

            if (isIndexed) {

                count = geometry->getIndex()->count();

            } else if (geometry->getAttribute<float>("position")) {

                count = geometry->getAttribute<float>("position")->count();

            } else {

                std::cerr << "THREE.BufferGeometryUtils: .mergeBufferGeometries() failed with geometry at index " << i << ". The geometry must have either an index or a position attribute" << std::endl;
                return nullptr;
            }

            mergedGeometry->addGroup(offset, count, i);

            offset += count;
        }
    }

    // merge indices

    if (isIndexed) {

        unsigned int indexOffset = 0;
        std::vector<unsigned int> mergedIndex;

        for (auto geometry : geometries) {

            auto index = geometry->getIndex();

            for (auto j = 0; j < index->count(); ++j) {

                mergedIndex.emplace_back(index->getX(j) + indexOffset);
            }

            indexOffset += geometry->getAttribute<float>("position")->count();
        }

        mergedGeometry->setIndex(mergedIndex);
    }

    // merge attributes

    for (const auto& [name, attr] : attributes) {

        // Dispatch on the first geometry's scalar type; every supported
        // TypedBufferAttribute instantiation merges, so narrow (glTF-preserved
        // or compressAttributes) attributes survive a merge un-widened. A
        // geometry whose matching attribute has a DIFFERENT scalar type fails
        // the merge (typed<T> returns null → mergeAs yields nullptr).
        std::unique_ptr<BufferAttribute> mergedAttribute;
        mergeAs<unsigned int>(attr, mergedAttribute) ||
                mergeAs<float>(attr, mergedAttribute) ||
                mergeAs<std::uint16_t>(attr, mergedAttribute) ||
                mergeAs<std::int16_t>(attr, mergedAttribute) ||
                mergeAs<std::uint8_t>(attr, mergedAttribute) ||
                mergeAs<std::int8_t>(attr, mergedAttribute);

        if (!mergedAttribute) {

            std::cerr << "THREE.BufferGeometryUtils: .mergeBufferGeometries() failed while trying to merge the " << name << " attribute." << std::endl;
            return nullptr;
        }

        mergedGeometry->setAttribute(name, std::move(mergedAttribute));
    }

    return mergedGeometry;
}

std::shared_ptr<BufferGeometry> threepp::mergeBufferGeometries(const std::vector<std::shared_ptr<BufferGeometry>>& geometries, bool useGroups) {

    std::vector<BufferGeometry*> arr;
    arr.reserve(geometries.size());

    for (const auto& g : geometries) {
        arr.emplace_back(g.get());
    }

    return mergeBufferGeometries(arr, useGroups);
}

std::shared_ptr<BufferGeometry> threepp::mergeVertices(const BufferGeometry& geometry, float tolerance) {

    tolerance = std::max(tolerance, std::numeric_limits<float>::epsilon());

    const auto* positionAttr = geometry.getAttribute<float>("position");
    if (!positionAttr) {

        std::cerr << "THREE.BufferGeometryUtils: .mergeVertices() failed. Geometry is missing a position attribute." << std::endl;
        return nullptr;
    }

    const auto* index = geometry.getIndex();
    const unsigned int vertexCount = index ? index->count() : positionAttr->count();

    const auto& srcAttributes = geometry.getAttributes();

    std::unordered_map<std::string, std::vector<float>> floatArrays;
    std::unordered_map<std::string, std::vector<unsigned int>> uintArrays;
    std::unordered_map<std::string, int> itemSizes;
    std::unordered_map<std::string, bool> normalizeds;

    // Every non-uint attribute is read through a FloatAttributeView: zero-copy
    // for float sources, widened (and denormalized) once for narrow ones. The
    // welded output of a narrow source is therefore float — mergeVertices
    // trades that compression away rather than failing; re-run
    // compressAttributes() on the result to get it back.
    std::unordered_map<std::string, FloatAttributeView> floatViews;

    for (const auto& [name, attr] : srcAttributes) {

        itemSizes[name] = attr->itemSize();
        normalizeds[name] = attr->normalized();

        if (attr->typed<unsigned int>()) {
            uintArrays[name] = {};
        } else {
            FloatAttributeView view(attr.get());
            if (!view) {
                std::cerr << "THREE.BufferGeometryUtils: .mergeVertices() failed. Unsupported attribute type for \"" << name << "\"." << std::endl;
                return nullptr;
            }
            floatViews.emplace(name, std::move(view));
            floatArrays[name] = {};
        }
    }

    const double shiftMultiplier = std::pow(10.0, std::log10(1.0 / static_cast<double>(tolerance)));

    std::unordered_map<std::string, unsigned int> hashToIndex;
    std::vector<unsigned int> newIndices;
    newIndices.reserve(vertexCount);
    unsigned int nextIndex = 0;

    std::string hash;

    for (unsigned int i = 0; i < vertexCount; ++i) {

        const unsigned int srcIndex = index ? static_cast<unsigned int>(index->getX(i)) : i;

        hash.clear();

        for (const auto& [name, attr] : srcAttributes) {

            const int itemSize = itemSizes[name];

            if (auto viewIt = floatViews.find(name); viewIt != floatViews.end()) {

                const auto& view = viewIt->second;
                for (int k = 0; k < itemSize; ++k) {
                    const double v = static_cast<double>(view[srcIndex * itemSize + k]) * shiftMultiplier;
                    hash += std::to_string(static_cast<long long>(v));
                    hash += ',';
                }

            } else if (auto uAttr = attr->typed<unsigned int>()) {

                const auto& arr = uAttr->array();
                for (int k = 0; k < itemSize; ++k) {
                    hash += std::to_string(arr[srcIndex * itemSize + k]);
                    hash += ',';
                }
            }
        }

        auto it = hashToIndex.find(hash);
        if (it != hashToIndex.end()) {

            newIndices.emplace_back(it->second);

        } else {

            for (const auto& [name, attr] : srcAttributes) {

                const int itemSize = itemSizes[name];

                if (auto viewIt = floatViews.find(name); viewIt != floatViews.end()) {

                    const auto& view = viewIt->second;
                    auto& dst = floatArrays[name];
                    for (int k = 0; k < itemSize; ++k) {
                        dst.emplace_back(view[srcIndex * itemSize + k]);
                    }

                } else if (auto uAttr = attr->typed<unsigned int>()) {

                    const auto& src = uAttr->array();
                    auto& dst = uintArrays[name];
                    for (int k = 0; k < itemSize; ++k) {
                        dst.emplace_back(src[srcIndex * itemSize + k]);
                    }
                }
            }

            hashToIndex[hash] = nextIndex;
            newIndices.emplace_back(nextIndex);
            ++nextIndex;
        }
    }

    auto result = BufferGeometry::create();

    for (const auto& [name, attr] : srcAttributes) {

        if (floatViews.contains(name)) {
            // A widened narrow source was denormalized by the view, so the
            // output floats are plain values — clear the normalized flag.
            const bool wasNarrow = attr->type() != AttributeType::Float;
            result->setAttribute(name, TypedBufferAttribute<float>::create(
                                               floatArrays[name], itemSizes[name],
                                               wasNarrow ? false : normalizeds[name]));
        } else if (attr->typed<unsigned int>()) {
            result->setAttribute(name, TypedBufferAttribute<unsigned int>::create(uintArrays[name], itemSizes[name], normalizeds[name]));
        }
    }

    result->setIndex(newIndices);

    for (const auto& group : geometry.groups) {
        result->addGroup(group.start, group.count, group.materialIndex);
    }

    return result;
}

std::shared_ptr<BufferGeometry> threepp::simplifyGeometry(const BufferGeometry& geometry, float ratio, float error) {

    std::shared_ptr<BufferGeometry> geom;
    if (!geometry.getIndex()) {
        geom = mergeVertices(geometry);
    } else {
        geom = geometry.clone();
    }

    auto* pos = geom->getAttribute<float>("position");
    auto* uv = geom->getAttribute<float>("uv");
    auto* idx = geom->getIndex();
    if (!pos || !idx) return geom;

    const size_t vertexCount = pos->count();
    const size_t indexCount = idx->count();
    const auto* indices = idx->array().data();

    std::vector<float> positions(vertexCount * 3);
    for (size_t i = 0; i < vertexCount; ++i) {
        positions[i * 3 + 0] = pos->getX(i);
        positions[i * 3 + 1] = pos->getY(i);
        positions[i * 3 + 2] = pos->getZ(i);
    }

    std::vector<float> uvs;
    const bool hasUV = uv != nullptr;
    if (hasUV) {
        uvs.resize(vertexCount * 2);
        for (size_t i = 0; i < vertexCount; ++i) {
            uvs[i * 2 + 0] = uv->getX(i);
            uvs[i * 2 + 1] = uv->getY(i);
        }
    }

    size_t targetIndexCount = (static_cast<size_t>(static_cast<float>(indexCount) * ratio) / 3) * 3;
    std::vector<unsigned int> outIndices(indexCount);
    float resultError = 0.f;

    size_t outCount;
    if (hasUV) {
        float attrWeights[2] = {0.2f, 0.2f};
        outCount = meshopt_simplifyWithAttributes(
                outIndices.data(), indices, indexCount,
                positions.data(), vertexCount, sizeof(float) * 3,
                uvs.data(), sizeof(float) * 2, attrWeights, 2,
                nullptr, targetIndexCount, error, 0, &resultError);
    } else {
        outCount = meshopt_simplify(
                outIndices.data(), indices, indexCount,
                positions.data(), vertexCount, sizeof(float) * 3,
                targetIndexCount, error, 0, &resultError);
    }
    outIndices.resize(outCount);

    struct Vertex {
        float px, py, pz, u, v;
    };
    std::vector<Vertex> vertices(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i) {
        vertices[i] = {
                pos->getX(i), pos->getY(i), pos->getZ(i),
                hasUV ? uv->getX(i) : 0.f,
                hasUV ? uv->getY(i) : 0.f};
    }

    std::vector<unsigned int> remap(vertexCount);
    size_t newVertexCount = meshopt_generateVertexRemap(
            remap.data(), outIndices.data(), outCount,
            vertices.data(), vertexCount, sizeof(Vertex));
    meshopt_remapIndexBuffer(outIndices.data(), outIndices.data(), outCount, remap.data());
    std::vector<Vertex> newVerts(newVertexCount);
    meshopt_remapVertexBuffer(newVerts.data(), vertices.data(), vertexCount,
                              sizeof(Vertex), remap.data());

    std::vector<float> outPos, outUV;
    outPos.reserve(newVertexCount * 3);
    outUV.reserve(newVertexCount * 2);
    for (const auto& v : newVerts) {
        outPos.insert(outPos.end(), {v.px, v.py, v.pz});
        outUV.insert(outUV.end(), {v.u, v.v});
    }

    auto out = BufferGeometry::create();
    out->setAttribute("position", FloatBufferAttribute::create(outPos, 3));
    if (hasUV) out->setAttribute("uv", FloatBufferAttribute::create(outUV, 2));
    out->setIndex(outIndices);
    return out;
}

namespace {

    // Quantise a float attribute into `Narrow`, using `scale` as the full-range
    // multiplier of the target normalized format. Returns nullptr when the
    // attribute is missing or is not float (already narrowed).
    template<class Narrow>
    std::unique_ptr<TypedBufferAttribute<Narrow>> quantize(
            const BufferAttribute* attribute, float scale, bool signedRange) {

        if (!attribute || attribute->type() != AttributeType::Float) return nullptr;
        if (attribute->normalized()) return nullptr;// already in a normalized domain

        const auto* src = static_cast<const float*>(attribute->data());
        const size_t n = attribute->byteLength() / sizeof(float);

        const float lo = signedRange ? -1.f : 0.f;

        std::vector<Narrow> out(n);
        for (size_t i = 0; i < n; ++i) {
            const float v = std::clamp(src[i], lo, 1.f);
            out[i] = static_cast<Narrow>(std::lround(v * scale));
        }

        return TypedBufferAttribute<Narrow>::create(std::move(out), attribute->itemSize(), true);
    }

    // Every component must already sit inside [lo, 1] for quantisation to be
    // lossless-in-range; otherwise clamping would silently move geometry.
    bool withinRange(const BufferAttribute* attribute, float lo) {

        if (!attribute || attribute->type() != AttributeType::Float) return false;

        const auto* src = static_cast<const float*>(attribute->data());
        const size_t n = attribute->byteLength() / sizeof(float);

        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(src[i]) || src[i] < lo || src[i] > 1.f) return false;
        }
        return true;
    }

    size_t replaceIfSmaller(BufferGeometry& geometry, const std::string& name,
                            std::unique_ptr<BufferAttribute> narrowed) {

        if (!narrowed) return 0;

        const size_t before = geometry.getAttribute(name)->byteLength();
        const size_t after = narrowed->byteLength();
        if (after >= before) return 0;

        geometry.setAttribute(name, std::move(narrowed));
        return before - after;
    }

}// namespace

size_t threepp::compressAttributes(BufferGeometry& geometry, const AttributeCompression& what) {

    size_t saved = 0;

    if (what.normal && geometry.hasAttribute("normal")) {
        const auto* attr = geometry.getAttribute("normal");
        // Normals are unit-length, so -1..1 is guaranteed rather than checked;
        // a non-finite component would still poison the quantisation.
        if (withinRange(attr, -1.f)) {
            saved += replaceIfSmaller(geometry, "normal", quantize<std::int16_t>(attr, 32767.f, true));
        }
    }

    if (what.tangent && geometry.hasAttribute("tangent")) {
        const auto* attr = geometry.getAttribute("tangent");
        if (withinRange(attr, -1.f)) {
            saved += replaceIfSmaller(geometry, "tangent", quantize<std::int16_t>(attr, 32767.f, true));
        }
    }

    if (what.uv && geometry.hasAttribute("uv")) {
        const auto* attr = geometry.getAttribute("uv");
        // Tiled or atlas UVs run outside [0,1]; unorm16 would clamp them onto the
        // texture edge, so those geometries keep their float UVs.
        if (withinRange(attr, 0.f)) {
            saved += replaceIfSmaller(geometry, "uv", quantize<std::uint16_t>(attr, 65535.f, false));
        }
    }

    if (what.color && geometry.hasAttribute("color")) {
        const auto* attr = geometry.getAttribute("color");
        if (withinRange(attr, 0.f)) {
            saved += replaceIfSmaller(geometry, "color", quantize<std::uint8_t>(attr, 255.f, false));
        }
    }

    return saved;
}

size_t threepp::compressSceneAttributes(Object3D& root, const AttributeCompression& what) {

    size_t saved = 0;

    root.traverseType<Mesh>([&](Mesh& m) {
        auto geom = m.geometry();
        if (!geom) return;

        // Deforming geometry must stay float: the Vulkan skinned/displaced/
        // grass-wind/morph/softbody paths rewrite float device buffers every
        // frame, and CPU-side skinning (boneTransform) reads typed float
        // attributes. Ocean derives DisplacedMesh and is covered by that check.
        if (dynamic_cast<SkinnedMesh*>(&m)) return;
#ifdef THREEPP_WITH_VULKAN
        if (dynamic_cast<DisplacedMesh*>(&m)) return;
#endif
        if (dynamic_cast<GrassMesh*>(&m)) return;
        if (!geom->getMorphAttributes().empty()) return;
        if (geom->hasAttribute("tetIndex")) return;// SoftBody::enableGpuSkinning

        saved += compressAttributes(*geom, what);
    });

    return saved;
}
