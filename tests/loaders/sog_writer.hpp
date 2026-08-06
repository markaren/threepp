// Test-only writer for the PlayCanvas SOG v2 chunk layout.
//
// Same bargain as splat_ply_writer.hpp next door: the loader gets pinned by
// round trip rather than by a checked-in binary blob, and that proves something
// only because the writer applies the format's conventions in the OPPOSITE
// direction to the loader — log-space 16-bit means split across two planes,
// log-scale codebook indices, smallest-three quaternions in (w, x, y, z) tuple
// order, raw SH DC coefficients rather than colours. A sign or an ordering
// error in one of them cannot cancel out against the same error in the other.
//
// THE TRICK THAT MAKES THIS POSSIBLE AT ALL. A SOG chunk stores its planes as
// WebP, and threepp vendors the libwebp DECODER only — src/external/libwebp has
// dec/dsp/utils/webp and no enc — so a C++ test cannot encode WebP at run time.
// It does not have to. ImageLoader dispatches on CONTENT, not on file
// extension: it sniffs the RIFF/WEBP magic and otherwise falls through to stb.
// So this writer emits every plane as a PNG while NAMING it "means_l.webp" (and
// so on) in the meta.json it writes. SogLoader resolves plane paths out of
// meta.json's files[] arrays and hands them to ImageLoader, which sniffs PNG
// and decodes it correctly. tests/loaders/ImageLoader_test.cpp already pins
// content-beats-extension, so this leans on a tested property of the loader
// rather than on an accident. The bytes on disk are PNG; the names are not.
//
// PADDING IS POISONED, and that is on by default. The image is padded to
// width * height >= count, and a correct reader iterates exactly `count` pixels
// in top-down row order. Every pixel past that gets values a correct reader
// never sees: an invalid quaternion mode (alpha 0 rather than 252..255, so the
// smallest-three index decodes to -252), zero opacity, a palette label of
// 0xffff that no palette this size contains, the largest scale in the cloud,
// and means pinned to alternating corners of the quantisation box. Two very
// specific bugs walk straight into them — a loader that iterates width * height
// instead of count, and a loader that leaves ImageLoader's flipY defaulted to
// TRUE, which mirrors the rows and so serves the bottom (padding) row as splat
// 0. Turn poisonPadding off and both bugs produce plausible-looking output
// instead.
//
// Means poison is bounded, unavoidably: the format decodes a mean by lerping
// between meta's per-axis mins/maxs, so no pixel value can name a point outside
// the cloud's own box, and the padding means land on a box CORNER rather than
// somewhere absurd. Widening the stored range to make room for an absurd value
// would spend real precision on every splat, so it is not done. The loud poison
// is in the other planes.
//
// WHAT SETS THE TEST'S TOLERANCES. Every channel here is quantised, and for a
// test writer the codebooks are laid out linearly rather than by k-means (see
// buildCodebook), so the error bounds are closed-form. With `range` the spread
// of the values being encoded:
//
//   means      16 bits over the per-axis LOG-space range. Half-step is
//              range/131070 in log space; because p = sign(n)*(exp|n| - 1), the
//              absolute position error is that times (|p| + 1).
//   scales     256 entries over the log-scale range -> a RELATIVE scale error
//              of about range/510.
//   quats      8 bits over [-1/sqrt2, +1/sqrt2] -> 1/sqrt2/255 ~= 0.00277
//              absolute per stored component, independent of the data. The
//              omitted component is rebuilt as sqrt(1 - sum of squares), so its
//              own error grows as it shrinks; compare quaternions by angle or
//              by dot product, not component-wise, when it is near zero.
//   opacity    8 bits over [0, 1], used raw with no sigmoid -> 1/510.
//   sh0, shN   256 entries over the coefficient range -> range/510.
//
// Not public API. A production encoder would k-means the codebooks and the
// palette; this one does not, and says so where it matters. `extras` is
// dropped — SOG has nowhere to put it.

#ifndef THREEPP_SOG_WRITER_HPP
#define THREEPP_SOG_WRITER_HPP

#include "threepp/splats/SplatData.hpp"

// Spelled for a TEST target, which only gets ${PROJECT_SOURCE_DIR}/src on its
// include path (tests/CMakeLists.txt). FontLoader.cpp's shorter
// "nlohmann/json.hpp" and ObjectExporter.cpp's "stb_image_write.h" work only
// inside the library, which also has src/external/* on the path. These two
// spellings are the ones ObjectLoader_test.cpp and ImageLoader_test.cpp use.
#include "external/nlohmann/nlohmann/json.hpp"

// Declarations only: STB_IMAGE_WRITE_IMPLEMENTATION is deliberately NOT defined
// here, because threepp already compiles the implementation in
// src/threepp/utils/StbImageWrite.cpp and every test links threepp. Defining it
// in a header would also duplicate the symbols across any two TUs that include
// the header. ImageLoader_test.cpp gets away with defining it only because it
// is a single TU whose executable pulls nothing else out of the archive that
// references stbi_write_*.
#include "external/stb/stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>// not pulled in by <fstream> on libstdc++, only on MSVC
#include <fstream>
#include <string>
#include <vector>

namespace splattest {

    struct SogWriteOptions {

        // Entries in the shN palette. 0 means one entry per splat (label i == i),
        // which is a perfectly legal palette and the only one that round-trips
        // exactly — a test writer has no business running k-means. A smaller
        // value builds the palette out of the FIRST paletteCount splats and
        // gives splat i entry (i % paletteCount), which exercises a shared
        // palette at the cost of only splats below the count keeping their own
        // coefficients. Ignored at shDegree 0.
        int paletteCount = 0;

        // 0 packs the image as tightly as a square-ish layout allows. Anything
        // larger pads the image AREA up to at least this many pixels, which is
        // how a test asks for padding rows to exist in the first place.
        std::size_t padTo = 0;

        // Fill the pixels past `count` with values a correct reader never sees.
        // See the header comment; leave it on unless the test is specifically
        // about benign padding.
        bool poisonPadding = true;
    };

    namespace sogdetail {

        // A 256-entry codebook laid out linearly from lo to hi, and its inverse.
        // A production encoder would k-means these; linear spacing is what makes
        // the round-trip error a closed form the test can assert against.
        inline std::vector<double> buildCodebook(double lo, double hi) {

            std::vector<double> codebook(256, lo);
            if (hi > lo) {

                for (int i = 0; i < 256; ++i) codebook[i] = lo + (hi - lo) * (i / 255.);
            }
            return codebook;
        }

        // Degenerate hi == lo collapses every entry onto the same value, so any
        // index decodes exactly and 0 is as good as another.
        inline unsigned char quantise(double value, double lo, double hi) {

            if (!(hi > lo)) return 0;

            const double t = (value - lo) / (hi - lo) * 255.;
            const long i = std::lround(t);

            return static_cast<unsigned char>(std::clamp<long>(i, 0, 255));
        }

        inline std::uint16_t quantise16(double value, double lo, double hi) {

            if (!(hi > lo)) return 0;

            const double t = (value - lo) / (hi - lo) * 65535.;
            const long i = std::lround(t);

            return static_cast<std::uint16_t>(std::clamp<long>(i, 0, 65535));
        }

        // The format's log compression, encode direction. Its inverse is
        // p = sign(n) * (exp(|n|) - 1).
        inline double logEncode(double p) {

            return p < 0 ? -std::log1p(-p) : std::log1p(p);
        }

        // Scale is stored as an index into a codebook of LOG scales, so an
        // exactly zero axis is unrepresentable — the same hole the PLY format
        // has, and for the same reason. Clamped rather than written as -inf,
        // which would poison the whole codebook's range and with it every other
        // splat; round-trip tests use non-degenerate clouds.
        constexpr float kMinScale = 1e-8f;

        inline void appendPng(void* context, void* data, int size) {

            auto* out = static_cast<std::vector<unsigned char>*>(context);
            const auto* bytes = static_cast<const unsigned char*>(data);
            out->insert(out->end(), bytes, bytes + size);
        }

        // Encodes to memory and writes the bytes through std::ofstream rather
        // than handing stb the path: std::filesystem does the right thing with
        // the path encoding, stbi_write_png's fopen would not.
        inline void writePlane(const std::filesystem::path& path,
                               unsigned int width, unsigned int height, int comp,
                               const std::vector<unsigned char>& pixels) {

            std::vector<unsigned char> png;
            stbi_write_png_to_func(&appendPng, &png,
                                   static_cast<int>(width), static_cast<int>(height), comp,
                                   pixels.data(), static_cast<int>(width) * comp);

            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(png.data()),
                      static_cast<std::streamsize>(png.size()));
        }

        // Higher-order coefficients per channel for a given band count. The
        // shN planes exist only for bands 1..3.
        inline int shCoeffsForBands(int bands) {

            return bands == 1 ? 3 : bands == 2 ? 8
                                               : 15;
        }

    }// namespace sogdetail

    // Writes meta.json plus the plane images into `dir`, creating it if absent.
    // Returns the image width, which is what a caller needs to know where splat
    // i lives: x = i % width, y = i / width, counting rows from the TOP.
    inline unsigned int writeSogChunk(const std::filesystem::path& dir,
                                      const threepp::SplatData& data,
                                      const SogWriteOptions& options = {}) {

        using namespace threepp;
        using namespace sogdetail;

        std::filesystem::create_directories(dir);

        const std::size_t count = data.count();
        const int degree = std::clamp(data.shDegree, 0, 3);
        const int coeffCount = splats::shCoeffCount(degree);
        const int shCoeffs = degree > 0 ? shCoeffsForBands(degree) : 0;

        // --- image geometry ---------------------------------------------------
        // Square-ish, then padded up. Integer sqrt rather than ceil(sqrt(x)):
        // a perfect square must not round up on the strength of one ulp.
        const std::size_t area = std::max<std::size_t>(std::max(count, options.padTo), 1);

        auto width = static_cast<unsigned int>(std::sqrt(static_cast<double>(area)));
        if (width == 0) width = 1;
        while (static_cast<std::size_t>(width) * width < area) ++width;
        while (width > 1 && static_cast<std::size_t>(width - 1) * (width - 1) >= area) --width;

        const auto height = static_cast<unsigned int>((area + width - 1) / width);
        const std::size_t pixels = static_cast<std::size_t>(width) * height;

        // --- ranges, and the codebooks built from them -------------------------
        double meanLo[3] = {0, 0, 0};
        double meanHi[3] = {0, 0, 0};
        double scaleLo = 0, scaleHi = 0;
        double dcLo = 0, dcHi = 0;

        for (std::size_t i = 0; i < count; ++i) {

            const double m[3] = {logEncode(data.means[i].x),
                                 logEncode(data.means[i].y),
                                 logEncode(data.means[i].z)};
            const double s[3] = {std::log(std::max(static_cast<float>(data.scales[i].x), kMinScale)),
                                 std::log(std::max(static_cast<float>(data.scales[i].y), kMinScale)),
                                 std::log(std::max(static_cast<float>(data.scales[i].z), kMinScale))};
            const float* c = data.shAt(i);

            for (int k = 0; k < 3; ++k) {

                if (i == 0) {

                    meanLo[k] = meanHi[k] = m[k];
                    if (k == 0) {

                        scaleLo = scaleHi = s[0];
                        dcLo = dcHi = c[0];
                    }
                }
                meanLo[k] = std::min(meanLo[k], m[k]);
                meanHi[k] = std::max(meanHi[k], m[k]);
                scaleLo = std::min(scaleLo, s[k]);
                scaleHi = std::max(scaleHi, s[k]);
                dcLo = std::min(dcLo, static_cast<double>(c[k]));
                dcHi = std::max(dcHi, static_cast<double>(c[k]));
            }
        }

        const auto scaleCodebook = buildCodebook(scaleLo, scaleHi);
        const auto dcCodebook = buildCodebook(dcLo, dcHi);

        // --- the shN palette ---------------------------------------------------
        // One entry per splat by default: legal, exact, and no k-means. Labels
        // are 16 bit, hence the cap.
        std::size_t palette = 0;
        if (degree > 0 && count > 0) {

            palette = options.paletteCount > 0
                              ? std::min(count, static_cast<std::size_t>(options.paletteCount))
                              : count;
            palette = std::min<std::size_t>(palette, 65536);
        }

        double shLo = 0, shHi = 0;
        for (std::size_t n = 0; n < palette; ++n) {

            const float* c = data.shAt(n);
            for (int k = 3; k < coeffCount * 3; ++k) {

                if (n == 0 && k == 3) shLo = shHi = c[k];
                shLo = std::min(shLo, static_cast<double>(c[k]));
                shHi = std::max(shHi, static_cast<double>(c[k]));
            }
        }
        const auto shCodebook = buildCodebook(shLo, shHi);

        // --- the per-splat planes ---------------------------------------------
        std::vector<unsigned char> meansL(pixels * 4, 0), meansU(pixels * 4, 0),
                scalesPx(pixels * 4, 0), quatsPx(pixels * 4, 0), sh0Px(pixels * 4, 0),
                labelsPx(pixels * 4, 0);

        for (std::size_t i = 0; i < count; ++i) {

            const std::size_t o = i * 4;

            const Vector3& p = data.means[i];
            const double m[3] = {logEncode(p.x), logEncode(p.y), logEncode(p.z)};
            const float sc[3] = {data.scales[i].x, data.scales[i].y, data.scales[i].z};

            for (int k = 0; k < 3; ++k) {

                const std::uint16_t q = quantise16(m[k], meanLo[k], meanHi[k]);
                meansL[o + k] = static_cast<unsigned char>(q & 0xffu);
                meansU[o + k] = static_cast<unsigned char>((q >> 8) & 0xffu);

                // "The index whose exp() is nearest the linear scale." Rounding
                // in log space can land one index off that, because exp is
                // convex and the bin midpoints do not survive it, so the
                // neighbours get checked.
                const double logScale = std::log(std::max(sc[k], kMinScale));
                int best = quantise(logScale, scaleLo, scaleHi);
                double bestErr = std::abs(std::exp(scaleCodebook[best]) - sc[k]);
                for (int d = -1; d <= 1; d += 2) {

                    const int j = best + d;
                    if (j < 0 || j > 255) continue;
                    const double err = std::abs(std::exp(scaleCodebook[j]) - sc[k]);
                    if (err < bestErr) {

                        bestErr = err;
                        best = j;
                    }
                }
                scalesPx[o + k] = static_cast<unsigned char>(best);
            }
            meansL[o + 3] = meansU[o + 3] = scalesPx[o + 3] = 255;

            // Smallest-three. The tuple is (w, x, y, z) while threepp's
            // Quaternion is (x, y, z, w) — permute here, and only here. Get it
            // wrong and the result is still exactly unit length and still
            // passes validate(), which is why this is the one worth staring at.
            const SplatQuat& r = data.rotations[i];
            double q[4] = {r.w, r.x, r.y, r.z};

            const double len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (len > 0) {

                for (double& v : q) v /= len;

            } else {

                q[0] = 1;
                q[1] = q[2] = q[3] = 0;
            }

            int largest = 0;
            for (int k = 1; k < 4; ++k) {

                if (std::abs(q[k]) > std::abs(q[largest])) largest = k;
            }
            // The omitted component is reconstructed as a positive square root,
            // so the whole quaternion flips when it is negative. q and -q are
            // the same rotation, which is what makes that free.
            if (q[largest] < 0) {

                for (double& v : q) v = -v;
            }

            constexpr double kHalfSqrt2 = 0.7071067811865476;
            int j = 0;
            for (int k = 0; k < 4; ++k) {

                if (k == largest) continue;
                const long b = std::lround((q[k] * kHalfSqrt2 + 0.5) * 255.);
                quatsPx[o + j] = static_cast<unsigned char>(std::clamp<long>(b, 0, 255));
                ++j;
            }
            quatsPx[o + 3] = static_cast<unsigned char>(252 + largest);

            // sh0's rgb are indices into a codebook of RAW DC COEFFICIENTS. The
            // spec's 0.5 + c * SH_C0 is a rendering step and is applied
            // nowhere in the file.
            const float* c = data.shAt(i);
            for (int k = 0; k < 3; ++k) sh0Px[o + k] = quantise(c[k], dcLo, dcHi);

            // Opacity is the alpha byte, used directly: no logit, no sigmoid.
            const float opacity = std::clamp(data.opacities[i], 0.f, 1.f);
            sh0Px[o + 3] = static_cast<unsigned char>(std::lround(opacity * 255.f));

            if (palette > 0) {

                const std::size_t label = i % palette;
                labelsPx[o + 0] = static_cast<unsigned char>(label & 0xffu);
                labelsPx[o + 1] = static_cast<unsigned char>((label >> 8) & 0xffu);
                labelsPx[o + 2] = 0;
                labelsPx[o + 3] = 255;
            }
        }

        // The pixels past `count`. Poisoned by default; otherwise filled with a
        // legal, neutral splat rather than left zeroed, because a zeroed pixel
        // still carries quat alpha 0 and that is an invalid mode — accidental
        // poison. The whole value of the poison is that it is deliberate, so
        // switching it off has to produce genuinely plausible padding.
        for (std::size_t i = count; i < pixels; ++i) {

            const std::size_t o = i * 4;

            if (options.poisonPadding) {

                // Alternating corners of the quantisation box: a point that is
                // legal, reachable, and occupied by no splat in particular.
                // Means poison cannot be worse than a corner — see the header.
                for (int k = 0; k < 3; ++k) {

                    const unsigned char byte = (k % 2) ? 0x00 : 0xff;
                    meansL[o + k] = meansU[o + k] = byte;
                    scalesPx[o + k] = 0xff;// the largest scale in the cloud
                }

                // alpha 0 is not a mode: 0 - 252 is -252, and a reader that
                // trusts it indexes off the front of its own component table.
                // That is the point — it is a bug the reader already had.
                quatsPx[o + 0] = 0xff;
                quatsPx[o + 1] = 0x00;
                quatsPx[o + 2] = 0xff;
                quatsPx[o + 3] = 0x00;

                sh0Px[o + 0] = 0xff;
                sh0Px[o + 1] = 0x00;
                sh0Px[o + 2] = 0xff;
                sh0Px[o + 3] = 0x00;// opacity 0

                // No palette here is 65536 entries, so this indexes past the
                // centroid image the same way.
                labelsPx[o + 0] = 0xff;
                labelsPx[o + 1] = 0xff;

            } else {

                // Codebook entry 0 on both, and the identity rotation: mode 0
                // with all three stored components at the byte that decodes
                // nearest zero.
                quatsPx[o + 0] = quatsPx[o + 1] = quatsPx[o + 2] = 128;
                quatsPx[o + 3] = 252;
                sh0Px[o + 3] = 255;
            }

            meansL[o + 3] = meansU[o + 3] = scalesPx[o + 3] = 255;
            labelsPx[o + 3] = 255;
        }

        // stbi_write_png writes rows top-down, which is the order splat index
        // implies. The flip flag is a global that another test in the same
        // binary may have left set, so it gets cleared rather than assumed.
        stbi_flip_vertically_on_write(0);

        writePlane(dir / "means_l.webp", width, height, 4, meansL);
        writePlane(dir / "means_u.webp", width, height, 4, meansU);
        writePlane(dir / "scales.webp", width, height, 4, scalesPx);
        writePlane(dir / "quats.webp", width, height, 4, quatsPx);
        writePlane(dir / "sh0.webp", width, height, 4, sh0Px);

        // --- the shN palette image --------------------------------------------
        // 64 palette entries per ROW whatever the width, so the width is fixed
        // at 64 * shCoeffs and only the height follows the palette size. Within
        // an entry the coefficients run along u, which is COEFFICIENT-MAJOR and
        // therefore already SplatData's own sh[(splat*coeffCount + c)*3 + ch]
        // layout: no transpose anywhere below.
        if (palette > 0) {

            const auto cw = static_cast<unsigned int>(64 * shCoeffs);
            const auto chh = static_cast<unsigned int>((palette + 63) / 64);
            std::vector<unsigned char> centroids(static_cast<std::size_t>(cw) * chh * 3, 0);

            for (std::size_t n = 0; n < palette; ++n) {

                const float* c = data.shAt(n);
                const std::size_t v = n / 64;
                const std::size_t u0 = (n % 64) * static_cast<std::size_t>(shCoeffs);

                for (int cc = 0; cc < shCoeffs; ++cc) {

                    const std::size_t o = ((v * cw) + u0 + cc) * 3;
                    for (int ch = 0; ch < 3; ++ch) {

                        // Coefficient cc of the file is coefficient cc + 1 of
                        // SplatData, whose 0 is the DC term sh0 already carries.
                        centroids[o + ch] = quantise(c[(cc + 1) * 3 + ch], shLo, shHi);
                    }
                }
            }

            writePlane(dir / "shN_centroids.webp", cw, chh, 3, centroids);
            writePlane(dir / "shN_labels.webp", width, height, 4, labelsPx);
        }

        // --- meta.json ---------------------------------------------------------
        // ordered_json so the file reads in the order the format documents it,
        // rather than alphabetically.
        nlohmann::ordered_json meta;
        meta["version"] = 2;
        meta["asset"] = {{"generator", "threepp sog_writer"}};
        meta["count"] = count;

        // mins/maxs are IN LOG SPACE, not world space.
        meta["means"] = {{"mins", {meanLo[0], meanLo[1], meanLo[2]}},
                         {"maxs", {meanHi[0], meanHi[1], meanHi[2]}},
                         {"files", {"means_l.webp", "means_u.webp"}}};

        // One codebook of LOG scales shared by all three axes.
        meta["scales"] = {{"codebook", scaleCodebook},
                          {"files", {"scales.webp"}}};

        // Quats carry no codebook — the components are stored directly.
        meta["quats"] = {{"files", {"quats.webp"}}};

        meta["sh0"] = {{"codebook", dcCodebook},
                       {"files", {"sh0.webp"}}};

        if (palette > 0) {

            meta["shN"] = {{"count", palette},
                           {"bands", degree},
                           {"codebook", shCodebook},
                           {"files", {"shN_centroids.webp", "shN_labels.webp"}}};
        }

        std::ofstream out(dir / "meta.json");
        out << meta.dump(1);

        return width;
    }

}// namespace splattest

#endif//THREEPP_SOG_WRITER_HPP
