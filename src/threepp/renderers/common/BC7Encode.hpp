
#ifndef THREEPP_BC7ENCODE_HPP
#define THREEPP_BC7ENCODE_HPP

#include <cstdint>
#include <vector>

namespace threepp::bcn {

    // First-party BC7 encoder, mode 6 only.
    //
    // Mode 6 (single subset, 7.7.7.7 endpoints + per-endpoint P-bit, 4-bit
    // indices) is the workhorse mode for PBR content: full RGBA at 16 levels
    // of interpolation covers albedo, normal, roughness/metalness and cutout
    // alpha well. A multi-mode encoder buys a few dB on hard blocks at ~50x
    // the encode cost — not worth it for load-time transcoding, and mode 6 is
    // exactly the mode BCnDecode.cpp implements bit-exactly, so the round-trip
    // test can pin encoder and decoder against each other.
    //
    // Output is 16 bytes per 4x4 block, blocks in row-major order,
    // ceil(w/4) * ceil(h/4) blocks total. Edge blocks replicate the border
    // texel (encoder sees clamped samples; the GPU never reads outside w x h).
    //
    // Multithreaded over block rows for images with enough work to matter.
    std::vector<std::uint8_t> bc7EncodeMode6(const std::uint8_t* rgba, int w, int h);

    // Build a full RGBA8 mip chain down to 1x1 (box filter). When `srgb` the
    // RGB channels are averaged in linear space through lookup tables (alpha
    // is always linear). Level 0 is NOT included in the result.
    std::vector<std::vector<std::uint8_t>> buildMipChainRGBA8(
            const std::uint8_t* rgba, int w, int h, bool srgb);

    // The same chain, area-weighted: a level of odd size spreads each texel of
    // the next over 2 + 1/floor(n/2) of its own instead of pairing them up and
    // leaving the last row or column out. Every level then covers the whole
    // image, as a GL driver's glGenerateMipmap does. For images that are drawn
    // minified as they are (a sprite's glyph atlas is neither a power of two
    // nor even): the paired-up chain shifts each odd level against the one
    // above it, and a chain of bilinear blits skips rows outright, which drops
    // a glyph's horizontal strokes. Equal to buildMipChainRGBA8 where every
    // level is even (to a code, for sRGB).
    std::vector<std::vector<std::uint8_t>> buildMipChainRGBA8Area(
            const std::uint8_t* rgba, int w, int h, bool srgb);

}// namespace threepp::bcn

#endif//THREEPP_BC7ENCODE_HPP
