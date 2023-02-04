// https://github.com/mrdoob/three.js/blob/r129/src/core/InterleavedBufferAttribute.js

#ifndef THREEPP_INTERLEAVEDBUFFERATTRIBUTE_HPP
#define THREEPP_INTERLEAVEDBUFFERATTRIBUTE_HPP

#include <vector>
#include <memory>

#include "threepp/math/Matrix4.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/InterleavedBuffer.hpp"

namespace threepp {

    class InterleavedBufferAttribute: public TypedBufferAttribute<float> {

    public:
        unsigned int offset;
        std::shared_ptr<InterleavedBuffer> data;

        InterleavedBufferAttribute(std::shared_ptr<InterleavedBuffer> data, int itemSize, unsigned int offset, bool normalized)
            :  data(std::move(data)), offset(offset), TypedBufferAttribute<float>({}, itemSize, normalized) {}

        [[nodiscard]] const std::vector<float>& array() const override {
            return data->array();
        }

        [[nodiscard]] int count() const override {
            return data->count();
        }

    };

}

#endif//THREEPP_INTERLEAVEDBUFFERATTRIBUTE_HPP
