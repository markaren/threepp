
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <iostream>

using namespace threepp;


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

            if (!attributesUsed.count(name)) {

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

            for (unsigned j = 0; j < index->count(); ++j) {

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

