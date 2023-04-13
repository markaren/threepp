// https://github.com/mrdoob/three.js/blob/r129/src/core/InstancedBufferAttribute.js

#ifndef THREEPP_INSTANCEDBUFFERATTRIBUTE_HPP
#define THREEPP_INSTANCEDBUFFERATTRIBUTE_HPP

#include "BufferAttribute.hpp"

namespace threepp {

    template<class T>
    class InstancedBufferAttribute: public TypedBufferAttribute<T> {

    protected:
        int meshPerAttribute;

        InstancedBufferAttribute(std::vector<T> array, int itemSize, bool normalized, int meshPerAttribute = 1)
            : TypedBufferAttribute<T>(array, itemSize, normalized), meshPerAttribute(meshPerAttribute) {}
    };

}// namespace threepp

#endif//THREEPP_INSTANCEDBUFFERATTRIBUTE_HPP
