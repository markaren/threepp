
#include "threepp/utils/BufferGeometryUtils.hpp"

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
