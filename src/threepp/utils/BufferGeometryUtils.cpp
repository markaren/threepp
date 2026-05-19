
#include "threepp/utils/BufferGeometryUtils.hpp"

#ifdef THREEPP_WITH_MESHOPT
#include <meshoptimizer.h>
#endif

#include <cmath>
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

        std::unique_ptr<BufferAttribute> mergedAttribute;
        if (attr.front()->typed<unsigned int>()) {
            std::vector<TypedBufferAttribute<unsigned int>*> typed;
            for (auto& a : attr) {
                typed.emplace_back(a->typed<unsigned int>());
            }
            mergedAttribute = mergeBufferAttributes<unsigned int>(typed);
        } else if (attr.front()->typed<float>()) {
            std::vector<TypedBufferAttribute<float>*> typed;
            for (auto& a : attr) {
                typed.emplace_back(a->typed<float>());
            }
            mergedAttribute = mergeBufferAttributes<float>(typed);
        }

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

    for (const auto& [name, attr] : srcAttributes) {

        itemSizes[name] = attr->itemSize();
        normalizeds[name] = attr->normalized();

        if (attr->typed<float>()) {
            floatArrays[name] = {};
        } else if (attr->typed<unsigned int>()) {
            uintArrays[name] = {};
        } else {

            std::cerr << "THREE.BufferGeometryUtils: .mergeVertices() failed. Unsupported attribute type for \"" << name << "\"." << std::endl;
            return nullptr;
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

            if (auto fAttr = attr->typed<float>()) {

                const auto& arr = fAttr->array();
                for (int k = 0; k < itemSize; ++k) {
                    const double v = static_cast<double>(arr[srcIndex * itemSize + k]) * shiftMultiplier;
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

                if (auto fAttr = attr->typed<float>()) {

                    const auto& src = fAttr->array();
                    auto& dst = floatArrays[name];
                    for (int k = 0; k < itemSize; ++k) {
                        dst.emplace_back(src[srcIndex * itemSize + k]);
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

        if (attr->typed<float>()) {
            result->setAttribute(name, TypedBufferAttribute<float>::create(floatArrays[name], itemSizes[name], normalizeds[name]));
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

#ifdef THREEPP_WITH_MESHOPT

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

#endif
