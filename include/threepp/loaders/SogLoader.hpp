// Loader for PlayCanvas SOG v2 / SuperSplat "SSOG" Gaussian-splat scans.
//
// Provenance: clean-room. The container is the SOG v2 layout published on
// developer.playcanvas.com — a meta.json plus six or seven WebP planes — and
// every formula here was re-derived from that page, then checked splat for
// splat against the equivalent INRIA .ply of the same scan. No PlayCanvas,
// SuperSplat or splat-transform code was consulted.
//
// WHY, when SplatLoader already reads the .ply: the Sanctuaire
// Sainte-Anne-de-Beaupré scan is 89.9 MB at LOD 0 as SOG and 1,124.9 MB as the
// .ply it decodes to — bit for bit the same splats. That 12.5x is the whole
// point of the format. It is a DISK win only: both decode to the same ~59
// floats per splat, so a 5.0M-splat level is ~1.2 GB resident either way.
//
// THE PLANES, all indexed by splat i at (x = i % width, y = i / width):
//
//   means_l / means_u   16-bit q per axis, q = (u << 8) | l
//   scales              one codebook index per axis
//   quats               smallest-three; the omitted component's index in ALPHA
//   sh0                 rgb = DC codebook indices, ALPHA = opacity
//   shN_labels          16-bit palette index, r | (g << 8)
//   shN_centroids       the palette itself
//
// SEVEN CONVENTIONS BITE, and every one of them produces a plausible-looking
// cloud rather than an error. SplatData::validate() catches NONE of them: it
// passes cleanly on a cloud decoded with the wrong quaternion slot, with flipY
// left true, with the DC converted to colour, and with the means never
// expanded. All seven are pinned by SogLoader_test.
//
//   1. These images are DATA, not pictures. ImageLoader::load's flipY argument
//      DEFAULTS TO TRUE and every read here passes false. With the default,
//      every splat silently wears another splat's attributes and the scene
//      still renders as a believable point cloud. Worse on a real chunk: 0_0
//      is 744x740 with only 739 rows used, so flipY also drags the padding row
//      to the top and splat 0 reads encoder garbage.
//   2. `count` is authoritative; width * height is NOT. The planes are padded
//      (chunk 0_0 is 744x740 for 549,365 splats) and the padding is not inert:
//      quats pads with alpha 0, so a loop over width * height computes
//      mode = 0 - 252 and indexes a four-element tuple at -252.
//   3. means.mins/maxs are in LOG space. After the lerp the position is
//      sign(n) * (exp(|n|) - 1). Chunk 0_0's bounds span [-6.9, +6.6] while the
//      decoded scene spans ±1000 units, so a decoder that skips the expansion
//      renders the same scan about a hundred times too small — and renders it,
//      rather than failing. Divide by 65535, not 65536: the quantised value
//      attains both endpoints, and 65536 costs up to 0.125 world units.
//   4. quats is smallest-three. The omitted component's index is the alpha byte
//      minus 252, and it indexes the tuple in (W, X, Y, Z) order, which is not
//      threepp's (x, y, z, w). Read as (x, y, z, w) the result is still exactly
//      unit — normalizeRotations() and validate() both pass — and is wrong by a
//      median 1.6 degrees. UNIT LENGTH IS NOT A CHECK ON THIS DECODE. The
//      reconstruction clamps at sqrt(max(0, ...)), so a mis-scaled or corrupt
//      file can still exceed unit length; this loader normalises afterwards
//      rather than trusting the construction.
//   5. sh0's rgb are codebook indices for the RAW DC COEFFICIENT, and SplatData
//      stores coefficients. The spec's colour = 0.5 + c * SH_C0 is render-time
//      convenience and must NOT be applied here, or this loader disagrees with
//      SplatLoader on the same scan. Real DC values reach 8.15 on this scan —
//      those are the sky smears removeOutliers() exists to find, so they are
//      not clamped either. Alpha is opacity as a / 255 with NO sigmoid: unlike
//      the .ply's, it is already activated.
//   6. shN_centroids is COEFFICIENT-major along u, at 64 palette entries per row
//      regardless of the image width: entry n, coefficient c sits at
//      u = (n % 64) * shCoeffs + c, v = n / 64, and that pixel's r, g, b are the
//      coefficient's three channels. That is already SplatData's own layout, so
//      unlike SplatLoader's gotcha 1 there is NO channel-major reorder here.
//   7. The planes must be decoded as NON-premultiplied RGBA. quats keeps its
//      mode byte in alpha, so premultiplying would scale rgb by up to 252/255
//      and rotate 94% of the splats by a median 1.6 degrees, invisibly.
//      ImageLoader's WebP path uses MODE_RGBA, which is exactly that.
//
// ORIENTATION, and the trap in the spec. SOG v2 declares itself right-handed
// with +Y up, which reads like a promise that a scan needs no reorienting. It
// is not one: that line describes the CONTAINER, not the capture inside it.
// splat-transform re-encodes an existing 3DGS .ply without moving anything, so
// a SOG built from a COLMAP reconstruction is +Y DOWN exactly like its .ply —
// on the Sanctuaire scan the means decoded here match that .ply's to 8e-13 with
// no negation on any axis. This loader therefore returns the file's own
// coordinates untouched, and a caller that flips a COLMAP .ply on import must
// flip a COLMAP-derived SOG the same way. The editor does; skipping it put the
// basilica on its roof.

#ifndef THREEPP_SOGLOADER_HPP
#define THREEPP_SOGLOADER_HPP

#include "threepp/math/Box3.hpp"
#include "threepp/splats/SplatData.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace threepp {

    class SogLoader {

    public:
        struct Options {

            // Which detail level of a multi-level asset to read. 0 is the
            // finest and the default.
            //
            // The levels are ALTERNATIVES, not a residual pyramid: on the
            // Sanctuaire scan they hold 4,998,052 / 2,499,871 / 1,249,936 /
            // 624,968 splats, and those four sum to the asset's declared total
            // of 9,372,827 — i.e. every level independently covers the whole
            // scene at its own density. Loading more than one and concatenating
            // draws the same building four times over at four densities, and
            // the result looks merely fatter and slower rather than broken.
            // That is why this is an int and not a set, why load() returns
            // exactly one level, and why there is no "load everything" spelling.
            //
            // Must be 0 when the path resolves to a single chunk: a lone chunk
            // declares no levels.
            int lod = 0;
        };

        // One chunk of one level. Carried by describe() so a future streaming
        // path has the names, counts and bounds without re-parsing lod-meta.json
        // — 502 KB on the real asset, almost all of it octree. Nothing in
        // threepp consumes these yet; load() returns one flat SplatData.
        struct ChunkInfo {

            std::string name;   // as spelled in lod-meta.json's `filenames`
            std::size_t count{};
            Box3 bound;         // union of the tree nodes referencing this chunk
        };

        struct LevelInfo {

            int lod{};
            std::size_t count{};// splats at this level, across all its chunks
            std::vector<ChunkInfo> chunks;
        };

        // One leaf of the asset's spatial tree, and where its splats sit inside
        // each level.
        //
        // This is the ONE thing in the container that relates the levels to
        // each other: a leaf's entry for level 0 and its entry for level 2
        // describe the SAME piece of space at two densities. Per-level chunk
        // bounds cannot say that (a chunk of level 0 and a chunk of level 2
        // overlap arbitrarily), which is why per-node LOD needs the tree and
        // why LevelInfo::chunks alone was not enough.
        struct NodeRange {

            int lod{};            // the asset's own level number
            std::size_t chunk{};  // index into Info::levels[lod].chunks
            std::size_t offset{}; // splat offset INSIDE that chunk
            std::size_t count{};
        };

        struct NodeInfo {

            Box3 bound;                  // the leaf's own bound, verbatim
            std::vector<NodeRange> lods; // ascending lod; a level the leaf has
                                         // no splats at is simply absent
        };

        // What an asset holds, without decoding a single plane. Reads only the
        // json — milliseconds against the ~1.2 GB resident that load() of
        // level 0 costs, which is exactly why it exists: a caller deserves to
        // find out an asset has a 625k-splat level before paying for the 5.0M
        // one.
        struct Info {

            int lodLevels = 1;
            std::vector<LevelInfo> levels;

            // The tree's leaves, in the order lod-meta.json declares them —
            // which for every writer seen so far is also the order each chunk
            // file's splats are stored in, so a leaf's ranges within one chunk
            // are contiguous and the leaves of one chunk tile it exactly.
            // loadSogWithLod checks that rather than assuming it.
            //
            // EMPTY when the asset is a lone chunk, and empty when the tree's
            // level entries carry no `offset` member (an older writer): a
            // caller then falls back to whole-cloud selection rather than
            // guessing an offset.
            std::vector<NodeInfo> nodes;

            // The lod-meta tree's root bound, verbatim. Empty for a lone chunk,
            // which declares no bound of its own.
            Box3 bound;

            int shDegree{};
        };

        // Reads one level of a SOG asset into a single SplatData.
        //
        // `path` may be any of:
        //   - a directory holding lod-meta.json  (the whole asset)
        //   - a directory holding meta.json      (one chunk)
        //   - a meta.json or lod-meta.json file  (either of the above, named)
        //   - a ZIP archive holding the same     (.zip, .sog, any extension)
        //
        // The archive case is recognised by the PK\x03\x04 magic rather than by
        // extension, because it arrives under both names: superspl.at serves the
        // scan as a .zip download, while the SOG spec's bundled single-file form
        // is the same container called .sog. One path handles both, and neither
        // has to be unpacked by hand first.
        //
        // A directory (or archive root) is probed for lod-meta.json first, then
        // meta.json, so an asset root and a chunk inside it are never confused.
        // Nothing here parses a directory NAME: "0_0" is one writer's
        // convention, and lod-meta.json's `filenames` array is not even in that
        // order — its index 2 is "3_0". Chunks are resolved through the tree,
        // which is the only thing in the file that says which chunk belongs to
        // which level.
        //
        // Throws std::runtime_error naming the offending member, file and
        // numbers on anything it cannot represent.
        [[nodiscard]] static SplatData load(const std::filesystem::path& path);

        // Two overloads rather than a defaulted argument, for the reason
        // SplatData::removeOutliers has two: `= {}` on a nested type needs that
        // type's default member initializers complete while the enclosing class
        // still isn't, which GCC rejects.
        [[nodiscard]] static SplatData load(const std::filesystem::path& path, const Options& options);

        // Same resolution rules and the same exceptions as load(), but reads
        // only the json.
        [[nodiscard]] static Info describe(const std::filesystem::path& path);

        // Is this a SOG asset?
        //
        // The mirror of SplatLoader::isSplatPly, answering the same question the
        // same way — by content, not by name. A SOG asset usually has no
        // extension at all (it is a directory), the directory name carries no
        // information, and the archive form is a ZIP that a generic archive
        // handler would also claim. So the discriminator is the json: a
        // meta.json declaring version 2 with a `means` object carrying mins,
        // maxs and files; or a lod-meta.json declaring version 1 with lodLevels
        // and filenames.
        //
        // Never throws: a missing, unreadable or malformed asset is simply "not
        // a SOG", which leaves the caller's existing paths to report the failure
        // in their own words. Like isSplatPly it sniffs rather than validates, so
        // true means "this is a SOG asset and SogLoader owns the error message",
        // not "this will load".
        [[nodiscard]] static bool isSog(const std::filesystem::path& path);
    };

}// namespace threepp

#endif//THREEPP_SOGLOADER_HPP
