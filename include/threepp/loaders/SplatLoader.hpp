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

        // Does this file hold Gaussian splats rather than a mesh?
        //
        // ".ply" says nothing: the same extension carries Turk's original
        // triangle soup, a laser-scanned point cloud and a 3DGS optimiser's
        // output, and an importer that has to pick a loader gets no help from
        // the name. The header does say, and cheaply — f_dc_0 is the degree-0
        // SH coefficient every 3DGS file has and no mesh PLY does, so its
        // presence in the property table is the discriminator.
        //
        // Reads only the header (a few hundred bytes) and never throws:
        // a missing, unreadable or malformed file is simply "not a splat",
        // which leaves the existing mesh path to report the failure in its own
        // words rather than pre-empting it with a splat-flavoured complaint.
        [[nodiscard]] static bool isSplatPly(const std::filesystem::path& path);

        // Stream form, for tests and for callers that already have the bytes.
        // Consumes from the current position; the stream is left wherever the
        // scan stopped, so a caller that wants to parse afterwards must seek
        // back to the magic.
        [[nodiscard]] static bool isSplatPly(std::istream& stream);

        // â”€â”€ Colour-only point clouds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        // A PLY whose vertices carry a position and, optionally, a colour,
        // normals and scalar fields â€” what a laser scanner, CloudCompare,
        // MeshLab or Open3D writes â€” loaded as a SplatData of degree-0
        // Gaussians. Every point becomes one splat: its DC colour is the
        // point's colour (white when the file has none, grey from
        // `intensity` when it has that instead), its opacity is
        // PointCloudOptions::opacity, and its sigma is either the option's
        // `sigma` or sigmaPerSpacing times the cloud's median nearest-
        // neighbour distance (splats::medianNeighbourSpacing) â€” so at
        // SplatCloud::setPointMix 0 the cloud renders as a closed surface and
        // at 1 as its dots. A point with normals becomes a disc facing them:
        // the axis along the normal is scaled by normalThickness.
        //
        // Accepts binary_little_endian, binary_big_endian and ascii, any
        // numeric property type, and elements before or after the vertices
        // (they are skipped). A mesh PLY parses too â€” its vertices become
        // the cloud and its faces are ignored â€” which is what
        // isPointCloudPly exists to decide against for an importer.
        //
        // Properties consumed: x y z; red green blue (or r g b, or
        // diffuse_*; integer types are normalised by their range, floats
        // taken as [0, 1] unless the cloud's maximum exceeds 1, then as
        // [0, 255]); nx ny nz (or normal_*); intensity (or
        // scalar_intensity); alpha is dropped. Every other numeric property
        // lands in SplatData::extras as a float, like the splat loader's.
        struct PointCloudOptions {
            float sigma = 0.f;            // > 0: every point's sigma, in the file's units
            // sigma = this * the median nearest-neighbour distance when sigma
            // is 0. 1.0 closes a randomly sampled surface (a Poisson sampling's
            // nearest neighbour sits at about half its mean pitch); a lattice
            // reads as closed from ~0.6.
            float sigmaPerSpacing = 1.0f;
            float opacity = 1.f;
            bool useNormals = true;       // orient a disc along nx/ny/nz when present
            float normalThickness = 0.15f;// the disc's sigma along its normal, as a fraction of sigma
        };

        // What the loader found and decided; every field is also derivable
        // from the returned data, this is for the console line.
        struct PointCloudInfo {
            std::size_t count = 0;
            float spacing = 0.f;// median neighbour distance, 0 if not measured
            float sigma = 0.f;  // the sigma every point got
            bool hadColor = false;
            bool hadNormals = false;
            bool hadIntensity = false;
        };

        // Throws std::runtime_error, with the offending property in the
        // message, on anything it cannot represent. A cloud of one point
        // has no spacing and gets sigma 0.01.
        [[nodiscard]] static SplatData loadPointCloudPly(const std::filesystem::path& path,
                                                         const PointCloudOptions& options = {},
                                                         PointCloudInfo* info = nullptr);

        [[nodiscard]] static SplatData parsePointCloudPly(std::istream& stream,
                                                          const PointCloudOptions& options = {},
                                                          PointCloudInfo* info = nullptr);

        // The twin of isSplatPly for an importer choosing a loader: a vertex
        // element with x, y and z, no f_dc_0, and no `face` element declaring
        // any faces. Header only, never throws.
        [[nodiscard]] static bool isPointCloudPly(const std::filesystem::path& path);

        [[nodiscard]] static bool isPointCloudPly(std::istream& stream);

        // ── Writing ─────────────────────────────────────────────────────────
        // The INRIA layout back out, exactly what loadPly reads: binary
        // little-endian; x y z; f_dc_*; f_rest_* CHANNEL-major; opacity as a
        // logit; scale_* as logs; rot_* w-first; then every extra in name
        // order. A zero scale is written as log(1e-9) (the format has no zero)
        // and an opacity is clamped inside (0, 1) before the logit.
        static void writePly(const SplatData& data, std::ostream& out);

        // Throws std::runtime_error when the file cannot be opened.
        static void writePly(const SplatData& data, const std::filesystem::path& path);
    };

}// namespace threepp

#endif//THREEPP_SPLATLOADER_HPP
