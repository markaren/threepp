#ifndef THREEPP_BCNDECODE_HPP
#define THREEPP_BCNDECODE_HPP

// Private header — software BCn/DXT decode for compressed-texture upload in
// the Vulkan renderer. Not part of the public API.

#include <cstdint>
#include <vector>

namespace threepp::bcn {

    /// Decompress an entire BCn / DXT texture level to RGBA8.
    /// Supports DXT1 (BC1), DXT3 (BC2), DXT5 (BC3), BC4, BC5, BC7 and the sRGB
    /// variants of DXT1/DXT5/BC7 (sRGB is only a sampling flag, not a different
    /// codec). Returns an empty vector if the format is unsupported (e.g. BC6H).
    std::vector<std::uint8_t> bcnDecompress(const std::uint8_t* blocks, int w, int h, unsigned int glFmt);

}// namespace threepp::bcn

#endif//THREEPP_BCNDECODE_HPP
