// https://github.com/mrdoob/three.js/blob/r129/src/core/InterleavedBuffer.js

#ifndef THREEPP_INTERLEAVEDBUFFER_HPP
#define THREEPP_INTERLEAVEDBUFFER_HPP

#include "threepp/constants.hpp"
#include "threepp/utils/uuid.hpp"
#include "threepp/core/misc.hpp"

#include <vector>

namespace threepp {

    class InterleavedBuffer: public TypedBufferAttribute<float> {

    public:

        InterleavedBuffer(const std::vector<float> &array, int stride)
            : TypedBufferAttribute<float>(array, static_cast<int>(array.size() / stride)),
              stride_(stride) {}

        [[nodiscard]] int stride() const {
            return stride_;
        }

    private:

        int stride_;
    };

}

#endif//THREEPP_INTERLEAVEDBUFFER_HPP
