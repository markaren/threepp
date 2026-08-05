
#ifndef THREEPP_EXR_PIZDECODE_HPP
#define THREEPP_EXR_PIZDECODE_HPP

#include <cstddef>
#include <vector>

namespace threepp::detail {

    // Decode one PIZ-compressed scanline block.
    //
    // PIZ knows nothing about pixel types — it compresses a stream of 16-bit
    // words — so `shortsPerSample` says only how many words one sample of each
    // channel occupies, in chlist order: 1 for HALF, 2 for FLOAT and UINT.
    // Channels are assumed unsubsampled; EXRLoader rejects the rest before
    // getting here.
    //
    // `out` receives the block in the same layout an uncompressed chunk has —
    // for each scanline, each channel's whole row in turn — so the caller can
    // treat every codec's output identically. `outSize` must be the block's
    // uncompressed byte count; a mismatch is an error rather than a truncation.
    //
    // Returns false on malformed input, having written nothing outside `out`.
    bool pizDecode(const unsigned char* in, std::size_t inSize,
                   const std::vector<int>& shortsPerSample,
                   int width, int lines,
                   unsigned char* out, std::size_t outSize);

}// namespace threepp::detail

#endif//THREEPP_EXR_PIZDECODE_HPP
