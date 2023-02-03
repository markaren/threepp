// https://github.com/mrdoob/three.js/blob/r129/src/core/InterleavedBuffer.js

#ifndef THREEPP_INTERLEAVEDBUFFER_HPP
#define THREEPP_INTERLEAVEDBUFFER_HPP

#include "threepp/constants.hpp"
#include "threepp/utils/uuid.hpp"
#include "threepp/core/misc.hpp"

#include <vector>

namespace threepp {

    class InterleavedBuffer {

    public:
        unsigned int version = 0;

        std::vector<float> array;

        int stride;
        int count;

        int usage_{StaticDrawUsage};
        UpdateRange updateRange_{0, -1};

        const std::string uuid = utils::generateUUID();

        InterleavedBuffer(const std::vector<float>& array, int stride);

        InterleavedBuffer& setUsage(int usage) {
            this->usage_ = usage;

            return *this;
        }

        void needsUpdate() {
            ++version;
        }

    };

}

#endif//THREEPP_INTERLEAVEDBUFFER_HPP
