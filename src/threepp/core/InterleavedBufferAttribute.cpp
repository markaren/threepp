
#include <utility>

#include "threepp/core/InterleavedBufferAttribute.hpp"

using namespace threepp;

InterleavedBufferAttribute::InterleavedBufferAttribute(std::shared_ptr<InterleavedBuffer> data, int itemSize, unsigned int offset, bool normalized)
    : BufferAttribute(itemSize, normalized), data(std::move(data)), offset(offset) {}

