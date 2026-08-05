// Loader for the .ply files 3D Gaussian Splatting optimisers emit.
//
// Provenance: clean-room. The container is plain binary-little-endian PLY
// (Turk's format, public since 1994); the per-property conventions are those
// described in arXiv 2308.04079 and reproduced in the glTF KHR_gaussian_splatting
// and OpenUSD splat specifications. No third-party splatting code was consulted.
//
// The parser is driven by the header's property table — offsets and stride are
// computed from the declared properties, never assumed. Files in the wild carry
// different property sets (with or without normals, with per-splat extras from
// downstream tools, at any SH degree), and a hardcoded stride silently reads
// garbage from all of them.
//
// Three conventions bite anyone writing this for the first time; all three are
// handled here and pinned by SplatLoader_test:
//
//   1. f_rest_* is CHANNEL-major on disk. For degree 3 that is the 15
//      higher-order coefficients of RED, then of GREEN, then of BLUE — not
//      coefficient 1 in rgb, coefficient 2 in rgb, ... Reading it in the wrong
//      order raises no error; it just makes view-dependent colour subtly wrong
//      in a way that only shows up while orbiting.
//   2. opacity is stored pre-sigmoid and scale_* pre-exp.
//   3. rot_* is W-FIRST (rot_0 = w) and unnormalised.

#ifndef THREEPP_SPLATLOADER_HPP
#define THREEPP_SPLATLOADER_HPP

#include "threepp/splats/SplatData.hpp"

#include <filesystem>
#include <istream>

namespace threepp {

    class SplatLoader {

    public:
        // Throws std::runtime_error, with the offending property or count in
        // the message, on anything it cannot represent.
        [[nodiscard]] static SplatData loadPly(const std::filesystem::path& path);

        // Same parser over an already-open stream, which is how the tests feed
        // it hand-built buffers. The stream must be positioned at the "ply"
        // magic and opened in binary mode.
        [[nodiscard]] static SplatData parsePly(std::istream& stream);
    };

}// namespace threepp

#endif//THREEPP_SPLATLOADER_HPP
