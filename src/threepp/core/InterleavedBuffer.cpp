
#include "threepp/core/InterleavedBuffer.hpp"

using namespace threepp;

InterleavedBuffer::InterleavedBuffer(const std::vector<float> &array, int stride)
    : array(array),
      stride(stride),
      count(static_cast<int>(array.size() / stride)) {}
