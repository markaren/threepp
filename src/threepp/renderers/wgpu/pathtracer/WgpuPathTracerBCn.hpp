#ifndef THREEPP_WGPUPATHTRACERBCN_HPP
#define THREEPP_WGPUPATHTRACERBCN_HPP

// Private header — software BCn / DXT decompressor used by the path tracer's
// texture atlas builder. Not part of the public API.

#include <cstdint>
#include <vector>

namespace threepp::wgpu_pt {

    /// Decompress an entire BCn / DXT texture level to RGBA8.
    /// Supports DXT1 (BC1), DXT3 (BC2), DXT5 (BC3), BC4, BC5, BC7 and the sRGB
    /// variants of DXT1/DXT5/BC7 (sRGB is only a sampling flag, not a different
    /// codec). Returns an empty vector if the format is unsupported (e.g. BC6H).
    std::vector<std::uint8_t> bcnDecompress(const std::uint8_t* blocks, int w, int h, unsigned int glFmt);

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERBCN_HPP
