// https://github.com/mrdoob/three.js/blob/r129/src/core/InterleavedBuffer.js

#ifndef THREEPP_INTERLEAVEDBUFFER_HPP
#define THREEPP_INTERLEAVEDBUFFER_HPP

#include "threepp/constants.hpp"
#include "threepp/core/misc.hpp"
#include "threepp/utils/uuid.hpp"

#include <vector>

namespace threepp {

    class InterleavedBuffer: public TypedBufferAttribute<float> {

    public:
        [[nodiscard]] int stride() const {
            return stride_;
        }

        static std::shared_ptr<InterleavedBuffer> create(const std::vector<float>& array, int stride) {

            return std::shared_ptr<InterleavedBuffer>(new InterleavedBuffer(array, stride));
        }

    protected:
        InterleavedBuffer(const std::vector<float>& array, int stride)
            : TypedBufferAttribute<float>(array, static_cast<int>(array.size() / stride)),
              stride_(stride) {}

        int stride_;
    };

}// namespace threepp

#endif//THREEPP_INTERLEAVEDBUFFER_HPP
