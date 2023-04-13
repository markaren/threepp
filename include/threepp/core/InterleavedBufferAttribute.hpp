// https://github.com/mrdoob/three.js/blob/r129/src/core/InterleavedBufferAttribute.js

#ifndef THREEPP_INTERLEAVEDBUFFERATTRIBUTE_HPP
#define THREEPP_INTERLEAVEDBUFFERATTRIBUTE_HPP

#include <memory>
#include <vector>

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/InterleavedBuffer.hpp"
#include "threepp/math/Matrix4.hpp"

namespace threepp {

    class InterleavedBufferAttribute: public FloatBufferAttribute {

    public:
        unsigned int offset;
        std::shared_ptr<InterleavedBuffer> data;

        InterleavedBufferAttribute(std::shared_ptr<InterleavedBuffer> data, int itemSize, unsigned int offset, bool normalized)
            : data(std::move(data)), offset(offset), TypedBufferAttribute<float>({}, itemSize, normalized) {}

        [[nodiscard]] std::vector<float>& array() override {

            return data->array();
        }

        [[nodiscard]] int count() const override {

            return data->count();
        }
    };

}// namespace threepp

#endif//THREEPP_INTERLEAVEDBUFFERATTRIBUTE_HPP
