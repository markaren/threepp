// GPU-to-CPU pixel readback for the Wgpu renderer.
// Copies a render target's color texture to a staging buffer and
// maps it back to CPU memory, converting BGRA8 to RGB8.

#ifndef THREEPP_WGPUREADBACK_HPP
#define THREEPP_WGPUREADBACK_HPP

#include <cstdint>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp::wgpu {

    // Read the contents of a color texture as RGB pixels.
    // `bgra` selects the channel order of the source texture (render targets
    // and most surfaces are BGRA8; some platforms hand out RGBA8 surfaces).
    // Returns an empty vector on failure.
    std::vector<unsigned char> readRGBPixels(WGPUDevice device, WGPUQueue queue,
                                             WGPUTexture colorTexture,
                                             uint32_t width, uint32_t height,
                                             bool bgra = true);

}// namespace threepp::wgpu

#endif//THREEPP_WGPUREADBACK_HPP
