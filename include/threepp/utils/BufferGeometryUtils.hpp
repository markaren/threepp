// https://github.com/mrdoob/three.js/blob/r150/examples/jsm/utils/BufferGeometryUtils.js

#ifndef THREEPP_BUFFERGEOMETRYUTILS_HPP
#define THREEPP_BUFFERGEOMETRYUTILS_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/InterleavedBufferAttribute.hpp"

#include <memory>
#include <vector>

namespace threepp {

    std::shared_ptr<BufferGeometry> mergeBufferGeometries(const std::vector<BufferGeometry*>& geometries, bool useGroups = false);

    inline std::shared_ptr<BufferGeometry> mergeBufferGeometries(const std::vector<std::shared_ptr<BufferGeometry>>& geometries, bool useGroups = false) {
        std::vector<BufferGeometry*> arr;
        for (auto& g : geometries) {
            arr.emplace_back(g.get());
        }
        return mergeBufferGeometries(arr, useGroups);
    }

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
                array[offset+j] = arr[j];
            }

            offset += arr.size();
        }

        return TypedBufferAttribute<T>::create(array, itemSize.value(), normalized.value());
    }

}// namespace threepp

#endif//THREEPP_BUFFERGEOMETRYUTILS_HPP
