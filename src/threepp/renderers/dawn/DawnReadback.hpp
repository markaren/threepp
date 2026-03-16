// GPU-to-CPU pixel readback for the Dawn renderer.
// Copies a render target's color texture to a staging buffer and
// maps it back to CPU memory, converting BGRA8 to RGB8.

#ifndef THREEPP_DAWNREADBACK_HPP
#define THREEPP_DAWNREADBACK_HPP

#include <cstdint>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp::dawn {

    // Read the contents of a color texture as RGB pixels.
    // Returns an empty vector on failure.
    std::vector<unsigned char> readRGBPixels(WGPUDevice device, WGPUQueue queue,
                                             WGPUTexture colorTexture,
                                             uint32_t width, uint32_t height);

}// namespace threepp::dawn

#endif//THREEPP_DAWNREADBACK_HPP
