
#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/textures/Texture.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb/stb_image_write.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // 2x2 RGB test pattern with four distinct, byte-exact colors:
    //   top-left  red    | top-right  green
    //   bot-left  blue   | bot-right  white
    constexpr std::array<unsigned char, 12> kPattern{
            255, 0, 0, /**/ 0, 255, 0,
            0, 0, 255, /**/ 255, 255, 255};

    std::filesystem::path writePatternPng() {
        auto path = std::filesystem::temp_directory_path() / "threepp_imageloader_test.png";
        REQUIRE(stbi_write_png(path.string().c_str(), 2, 2, 3, kPattern.data(), 2 * 3) != 0);
        return path;
    }

    // pixel accessor into a loaded image
    std::array<unsigned char, 3> rgbAt(const Image& img, unsigned x, unsigned y, int channels) {
        const auto& d = const_cast<Image&>(img).data();
        const size_t o = (static_cast<size_t>(y) * img.width() + x) * channels;
        return {d[o], d[o + 1], d[o + 2]};
    }

    unsigned char alphaAt(const Image& img, unsigned x, unsigned y) {
        const auto& d = const_cast<Image&>(img).data();
        return d[(static_cast<size_t>(y) * img.width() + x) * 4 + 3];
    }

    // --- WebP fixtures -------------------------------------------------------
    //
    // No encoder is vendored (src/external/libwebp is the decoder subset only), so these
    // are pre-generated rather than written at test time. Produced with Pillow 12.3.0,
    // whose bundled libwebp is 1.6.0 — the exact version vendored here. All three encode
    // the same 2x2 kPattern colours, so the expectations below are shared with the PNG
    // sections above.

    // Lossless RGB. Pillow: save(lossless=True, quality=100, method=6)
    constexpr std::array<unsigned char, 34> kWebpLosslessRgb{
            82, 73, 70, 70, 26, 0, 0, 0, 87, 69, 66, 80,
            86, 80, 56, 76, 14, 0, 0, 0, 47, 1, 64, 0,
            0, 152, 255, 249, 159, 255, 254, 135, 194, 3};

    // Lossless RGBA, alphas 255/128/255/0. exact=True keeps the RGB under alpha 0 intact.
    constexpr std::array<unsigned char, 38> kWebpLosslessRgba{
            82, 73, 70, 70, 30, 0, 0, 0, 87, 69, 66, 80,
            86, 80, 56, 76, 18, 0, 0, 0, 47, 1, 64, 0,
            16, 152, 255, 249, 159, 127, 1, 65, 97, 186, 231, 133,
            100, 122};

    // Lossy VP8, quality 90. 4:2:0 collapses a 2x2's chroma to a single sample, so this
    // decodes near-grey — luma still orders the texels. With fancy upsampling (the
    // WebPDecoderConfig default) libwebp 1.6.0 gives 82,81,82 / 149,149,149 / 29,29,29 /
    // 253,253,253.
    constexpr std::array<unsigned char, 72> kWebpLossyRgb{
            82, 73, 70, 70, 64, 0, 0, 0, 87, 69, 66, 80,
            86, 80, 56, 32, 52, 0, 0, 0, 144, 1, 0, 157,
            1, 42, 2, 0, 2, 0, 0, 192, 18, 37, 164, 0,
            2, 231, 79, 140, 192, 0, 254, 252, 254, 255, 239, 27,
            224, 127, 166, 247, 143, 246, 132, 127, 249, 21, 231, 60,
            230, 214, 172, 52, 11, 158, 181, 190, 71, 182, 0, 0};

    // --- TIFF fixtures -------------------------------------------------------
    //
    // The same 2x2 kPattern colours, alphas 255/128/255/0 where there is alpha. Produced
    // with Pillow 12.2.0 (libtiff) and tifffile 2026.6.1, and read back by them before
    // embedding.

    // Pillow: RGBA, LZW, horizontal predictor (tiffinfo={317: 2}), little-endian strips.
    // ExtraSamples = 2 (unassociated alpha).
    constexpr std::array<unsigned char, 182> kTiffLzwPredictorRgba{
            73, 73, 42, 0, 24, 0, 0, 0, 128, 63, 192, 0, 7, 248, 6, 4,
            129, 129, 191, 225, 80, 32, 12, 4, 12, 0, 0, 1, 3, 0, 1, 0,
            0, 0, 2, 0, 0, 0, 1, 1, 3, 0, 1, 0, 0, 0, 2, 0,
            0, 0, 2, 1, 3, 0, 4, 0, 0, 0, 174, 0, 0, 0, 3, 1,
            3, 0, 1, 0, 0, 0, 5, 0, 0, 0, 6, 1, 3, 0, 1, 0,
            0, 0, 2, 0, 0, 0, 17, 1, 4, 0, 1, 0, 0, 0, 8, 0,
            0, 0, 21, 1, 3, 0, 1, 0, 0, 0, 4, 0, 0, 0, 22, 1,
            3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 23, 1, 4, 0, 1, 0,
            0, 0, 16, 0, 0, 0, 28, 1, 3, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 61, 1, 3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 82, 1,
            3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 8, 0,
            8, 0, 8, 0, 8, 0
    };

    // tifffile: RGBA as 16-bit samples (value * 257), big-endian, planar (one plane per
    // sample), Deflate with the predictor, ExtraSamples = 0 ("unspecified", what Blender
    // writes for a texture's cutout mask).
    constexpr std::array<unsigned char, 384> kTiffBigEndianPlanar16{
            77, 77, 0, 42, 0, 0, 0, 8, 0, 17, 1, 0, 0, 4, 0, 0,
            0, 1, 0, 0, 0, 2, 1, 1, 0, 4, 0, 0, 0, 1, 0, 0,
            0, 2, 1, 2, 0, 3, 0, 0, 0, 4, 0, 0, 0, 218, 1, 3,
            0, 3, 0, 0, 0, 1, 0, 8, 0, 0, 1, 6, 0, 3, 0, 0,
            0, 1, 0, 2, 0, 0, 1, 14, 0, 2, 0, 0, 0, 21, 0, 0,
            0, 226, 1, 17, 0, 4, 0, 0, 0, 4, 0, 0, 1, 6, 1, 21,
            0, 3, 0, 0, 0, 1, 0, 4, 0, 0, 1, 22, 0, 4, 0, 0,
            0, 1, 0, 0, 0, 2, 1, 23, 0, 3, 0, 0, 0, 4, 0, 0,
            1, 22, 1, 26, 0, 5, 0, 0, 0, 1, 0, 0, 1, 30, 1, 27,
            0, 5, 0, 0, 0, 1, 0, 0, 1, 38, 1, 28, 0, 3, 0, 0,
            0, 1, 0, 2, 0, 0, 1, 40, 0, 3, 0, 0, 0, 1, 0, 1,
            0, 0, 1, 49, 0, 2, 0, 0, 0, 12, 0, 0, 1, 46, 1, 61,
            0, 3, 0, 0, 0, 1, 0, 2, 0, 0, 1, 82, 0, 3, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 16, 0, 16,
            0, 16, 123, 34, 115, 104, 97, 112, 101, 34, 58, 32, 91, 52, 44, 32,
            50, 44, 32, 50, 93, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 64, 0, 0, 1, 80, 0, 0,
            1, 96, 0, 0, 1, 112, 0, 16, 0, 16, 0, 16, 0, 16, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 116, 105,
            102, 102, 102, 105, 108, 101, 46, 112, 121, 0, 0, 0, 0, 0, 0, 0,
            120, 156, 251, 255, 159, 129, 145, 129, 225, 255, 127, 0, 17, 251, 3, 254,
            120, 156, 99, 96, 248, 255, 159, 129, 225, 255, 127, 0, 13, 250, 3, 253,
            120, 156, 99, 96, 96, 96, 248, 255, 159, 129, 1, 0, 7, 1, 1, 255,
            120, 156, 251, 255, 191, 161, 241, 255, 127, 6, 70, 0, 27, 120, 4, 255
    };

    // tifffile: RGB, one 16x16 Deflate tile holding the 2x2 image.
    constexpr std::array<unsigned char, 316> kTiffTiledDeflateRgb{
            73, 73, 42, 0, 8, 0, 0, 0, 16, 0, 0, 1, 4, 0, 1, 0,
            0, 0, 2, 0, 0, 0, 1, 1, 4, 0, 1, 0, 0, 0, 2, 0,
            0, 0, 2, 1, 3, 0, 3, 0, 0, 0, 206, 0, 0, 0, 3, 1,
            3, 0, 1, 0, 0, 0, 8, 0, 0, 0, 6, 1, 3, 0, 1, 0,
            0, 0, 2, 0, 0, 0, 14, 1, 2, 0, 21, 0, 0, 0, 212, 0,
            0, 0, 21, 1, 3, 0, 1, 0, 0, 0, 3, 0, 0, 0, 26, 1,
            5, 0, 1, 0, 0, 0, 248, 0, 0, 0, 27, 1, 5, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 28, 1, 3, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 40, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 49, 1,
            2, 0, 12, 0, 0, 0, 8, 1, 0, 0, 66, 1, 4, 0, 1, 0,
            0, 0, 16, 0, 0, 0, 67, 1, 4, 0, 1, 0, 0, 0, 16, 0,
            0, 0, 68, 1, 4, 0, 1, 0, 0, 0, 32, 1, 0, 0, 69, 1,
            4, 0, 1, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 8, 0,
            8, 0, 8, 0, 123, 34, 115, 104, 97, 112, 101, 34, 58, 32, 91, 50,
            44, 32, 50, 44, 32, 51, 93, 125, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 116, 105, 102, 102, 102, 105, 108, 101,
            46, 112, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            120, 156, 251, 207, 192, 192, 240, 159, 129, 20, 240, 255, 255, 127, 210, 52,
            140, 130, 81, 192, 48, 88, 1, 0, 32, 209, 5, 251
    };

    template<size_t N>
    std::vector<unsigned char> bytesOf(const std::array<unsigned char, N>& a) {
        return {a.begin(), a.end()};
    }

    template<size_t N>
    std::filesystem::path writeBytes(const std::array<unsigned char, N>& a, const std::string& name) {
        auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream out(path, std::ios::binary);
        REQUIRE(out.is_open());
        out.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.size()));
        out.close();
        return path;
    }

}// namespace

TEST_CASE("ImageLoader preserves pixel layout") {

    const auto path = writePatternPng();
    ImageLoader loader;

    SECTION("flipY=false, 3 channels: rows in file order") {
        auto img = loader.load(path, 3, false);
        REQUIRE(img.has_value());
        CHECK(img->width() == 2);
        CHECK(img->height() == 2);
        CHECK(img->channels() == 3);
        CHECK(rgbAt(*img, 0, 0, 3) == std::array<unsigned char, 3>{255, 0, 0});  // red
        CHECK(rgbAt(*img, 1, 0, 3) == std::array<unsigned char, 3>{0, 255, 0});  // green
        CHECK(rgbAt(*img, 0, 1, 3) == std::array<unsigned char, 3>{0, 0, 255});  // blue
        CHECK(rgbAt(*img, 1, 1, 3) == std::array<unsigned char, 3>{255, 255, 255});
    }

    SECTION("flipY=true, 3 channels: rows vertically mirrored") {
        auto img = loader.load(path, 3, true);
        REQUIRE(img.has_value());
        CHECK(rgbAt(*img, 0, 0, 3) == std::array<unsigned char, 3>{0, 0, 255});  // blue now on top
        CHECK(rgbAt(*img, 1, 0, 3) == std::array<unsigned char, 3>{255, 255, 255});
        CHECK(rgbAt(*img, 0, 1, 3) == std::array<unsigned char, 3>{255, 0, 0});
        CHECK(rgbAt(*img, 1, 1, 3) == std::array<unsigned char, 3>{0, 255, 0});
    }

    SECTION("flipY=true, 4 channels: expanded with opaque alpha, mirrored") {
        auto img = loader.load(path, 4, true);
        REQUIRE(img.has_value());
        CHECK(img->channels() == 4);
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{0, 0, 255});
        CHECK(rgbAt(*img, 1, 1, 4) == std::array<unsigned char, 3>{0, 255, 0});
        const auto& d = img->data();
        CHECK(d[3] == 255);// alpha
    }

    SECTION("loading an HDR with flipY must not affect later image loads") {
        // Regression: RGBELoader used the process-global
        // stbi_set_flip_vertically_on_load, which leaked into every later
        // stbi decode — glTF textures (loaded with flipY=false) came back
        // upside-down once any HDR environment had been loaded.
        const auto hdrPath = std::filesystem::temp_directory_path() / "threepp_imageloader_test.hdr";
        const std::array<float, 6> hdrPattern{1.f, 0.f, 0.f, /* row 1 */ 0.f, 1.f, 0.f};// red over green
        REQUIRE(stbi_write_hdr(hdrPath.string().c_str(), 1, 2, 3, hdrPattern.data()) != 0);

        RGBELoader rgbe;
        auto hdr = rgbe.load(hdrPath, true);
        REQUIRE(hdr != nullptr);
        // flipY=true: green row first
        const auto& hdrData = hdr->image().data<float>();
        CHECK(hdrData[0] == 0.f);
        CHECK(hdrData[1] == 1.f);

        // the glTF pattern: flipY=false must still be in file order
        auto img = loader.load(path, 4, false);
        REQUIRE(img.has_value());
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{255, 0, 0});// red stays on top
    }

    SECTION("memory overload matches file overload") {
        // read the png bytes back and load via the memory path
        std::ifstream f(path, std::ios::binary);
        REQUIRE(f.is_open());
        std::vector<unsigned char> bytes(std::istreambuf_iterator<char>(f), {});
        REQUIRE(!bytes.empty());

        auto a = loader.load(path, 3, true);
        auto b = loader.load(bytes, 3, true);
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        CHECK(a->data() == b->data());
    }
}

TEST_CASE("ImageLoader decodes WebP") {

    ImageLoader loader;

    SECTION("lossless RGB, flipY=false: byte-exact against the source pattern") {
        auto img = loader.load(bytesOf(kWebpLosslessRgb), 3, false);
        REQUIRE(img.has_value());
        CHECK(img->width() == 2);
        CHECK(img->height() == 2);
        CHECK(img->channels() == 3);
        // Lossless means exactly the PNG pixels, not merely close ones.
        CHECK(img->data() == std::vector<unsigned char>(kPattern.begin(), kPattern.end()));
    }

    SECTION("lossless RGB, flipY=true: rows vertically mirrored") {
        auto img = loader.load(bytesOf(kWebpLosslessRgb), 3, true);
        REQUIRE(img.has_value());
        CHECK(rgbAt(*img, 0, 0, 3) == std::array<unsigned char, 3>{0, 0, 255});// blue on top
        CHECK(rgbAt(*img, 1, 0, 3) == std::array<unsigned char, 3>{255, 255, 255});
        CHECK(rgbAt(*img, 0, 1, 3) == std::array<unsigned char, 3>{255, 0, 0});
        CHECK(rgbAt(*img, 1, 1, 3) == std::array<unsigned char, 3>{0, 255, 0});
    }

    SECTION("lossless RGB, 4 channels: expanded with opaque alpha") {
        auto img = loader.load(bytesOf(kWebpLosslessRgb), 4, false);
        REQUIRE(img.has_value());
        CHECK(img->channels() == 4);
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{255, 0, 0});
        CHECK(rgbAt(*img, 1, 1, 4) == std::array<unsigned char, 3>{255, 255, 255});
        for (unsigned y = 0; y < 2; ++y) {
            for (unsigned x = 0; x < 2; ++x) {
                CHECK(alphaAt(*img, x, y) == 255);
            }
        }
    }

    SECTION("lossless RGBA: colours and per-texel alpha both survive") {
        auto img = loader.load(bytesOf(kWebpLosslessRgba), 4, false);
        REQUIRE(img.has_value());
        CHECK(img->channels() == 4);
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{255, 0, 0});
        CHECK(rgbAt(*img, 1, 0, 4) == std::array<unsigned char, 3>{0, 255, 0});
        CHECK(rgbAt(*img, 0, 1, 4) == std::array<unsigned char, 3>{0, 0, 255});
        CHECK(rgbAt(*img, 1, 1, 4) == std::array<unsigned char, 3>{255, 255, 255});
        CHECK(alphaAt(*img, 0, 0) == 255);
        CHECK(alphaAt(*img, 1, 0) == 128);
        CHECK(alphaAt(*img, 0, 1) == 255);
        CHECK(alphaAt(*img, 1, 1) == 0);
    }

    SECTION("lossy VP8 decodes to the recorded values") {
        // 4:2:0 chroma subsampling leaves a 2x2 with one chroma sample, so the colours
        // collapse to grey and only the luma ordering survives — dark, mid, darkest,
        // brightest. The tolerance guards against upsampling-mode drift between libwebp
        // releases, not against the decode being non-deterministic.
        auto img = loader.load(bytesOf(kWebpLossyRgb), 3, false);
        REQUIRE(img.has_value());
        CHECK(img->width() == 2);
        CHECK(img->height() == 2);

        constexpr std::array<unsigned char, 12> expected{
                82, 81, 82, /**/ 149, 149, 149,
                29, 29, 29, /**/ 253, 253, 253};

        const auto& d = img->data();
        REQUIRE(d.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            INFO("byte " << i);
            CHECK(std::abs(static_cast<int>(d[i]) - static_cast<int>(expected[i])) <= 3);
        }
    }

    SECTION("file overload matches memory overload") {
        const auto path = writeBytes(kWebpLosslessRgb, "threepp_imageloader_test.webp");

        auto fromFile = loader.load(path, 4, true);
        auto fromMemory = loader.load(bytesOf(kWebpLosslessRgb), 4, true);
        REQUIRE(fromFile.has_value());
        REQUIRE(fromMemory.has_value());
        CHECK(fromFile->data() == fromMemory->data());
    }

    SECTION("content beats extension: webp bytes in a .png-named file still decode") {
        // The loader sniffs the RIFF/WEBP magic rather than trusting the name, so a
        // mislabeled asset — common enough in exported scenes — loads anyway.
        const auto path = writeBytes(kWebpLosslessRgb, "threepp_imageloader_mislabeled.png");

        auto img = loader.load(path, 3, false);
        REQUIRE(img.has_value());
        CHECK(img->data() == std::vector<unsigned char>(kPattern.begin(), kPattern.end()));
    }

    SECTION("truncated data is rejected on both overloads") {
        // A valid RIFF/WEBP header with the payload cut off: this gets past the magic
        // check and has to be caught by the decoder itself.
        std::vector<unsigned char> truncated(kWebpLosslessRgb.begin(), kWebpLosslessRgb.begin() + 20);
        CHECK_FALSE(loader.load(truncated, 3, false).has_value());

        const auto path = std::filesystem::temp_directory_path() / "threepp_imageloader_truncated.webp";
        std::ofstream out(path, std::ios::binary);
        REQUIRE(out.is_open());
        out.write(reinterpret_cast<const char*>(truncated.data()),
                  static_cast<std::streamsize>(truncated.size()));
        out.close();
        CHECK_FALSE(loader.load(path, 3, false).has_value());
    }

    SECTION("greyscale channel counts are declined rather than guessed at") {
        // Legal for the stb path, used by no caller; libwebp has no grey output mode.
        CHECK_FALSE(loader.load(bytesOf(kWebpLosslessRgb), 1, false).has_value());
        CHECK_FALSE(loader.load(bytesOf(kWebpLosslessRgb), 2, false).has_value());
    }
}

TEST_CASE("ImageLoader decodes TIFF") {

    ImageLoader loader;

    SECTION("LZW + predictor, RGBA: colours and alpha byte-exact") {
        auto img = loader.load(bytesOf(kTiffLzwPredictorRgba), 4, false);
        REQUIRE(img.has_value());
        CHECK(img->width() == 2);
        CHECK(img->height() == 2);
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{255, 0, 0});
        CHECK(rgbAt(*img, 1, 0, 4) == std::array<unsigned char, 3>{0, 255, 0});
        CHECK(rgbAt(*img, 0, 1, 4) == std::array<unsigned char, 3>{0, 0, 255});
        CHECK(rgbAt(*img, 1, 1, 4) == std::array<unsigned char, 3>{255, 255, 255});
        CHECK(alphaAt(*img, 0, 0) == 255);
        CHECK(alphaAt(*img, 1, 0) == 128);
        CHECK(alphaAt(*img, 0, 1) == 255);
        CHECK(alphaAt(*img, 1, 1) == 0);
    }

    SECTION("big-endian, planar, 16-bit, Deflate + predictor: the 'unspecified' extra sample is alpha") {
        auto img = loader.load(bytesOf(kTiffBigEndianPlanar16), 4, false);
        REQUIRE(img.has_value());
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{255, 0, 0});
        CHECK(rgbAt(*img, 1, 0, 4) == std::array<unsigned char, 3>{0, 255, 0});
        CHECK(rgbAt(*img, 0, 1, 4) == std::array<unsigned char, 3>{0, 0, 255});
        CHECK(rgbAt(*img, 1, 1, 4) == std::array<unsigned char, 3>{255, 255, 255});
        CHECK(alphaAt(*img, 1, 0) == 128);
        CHECK(alphaAt(*img, 1, 1) == 0);
    }

    SECTION("tiled Deflate RGB, flipY=true and 4 channels: mirrored, opaque") {
        auto img = loader.load(bytesOf(kTiffTiledDeflateRgb), 4, true);
        REQUIRE(img.has_value());
        CHECK(img->width() == 2);
        CHECK(img->height() == 2);
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{0, 0, 255});// blue now on top
        CHECK(rgbAt(*img, 1, 0, 4) == std::array<unsigned char, 3>{255, 255, 255});
        CHECK(rgbAt(*img, 0, 1, 4) == std::array<unsigned char, 3>{255, 0, 0});
        CHECK(alphaAt(*img, 0, 0) == 255);
    }

    SECTION("3 and 1 channels") {
        auto rgb = loader.load(bytesOf(kTiffLzwPredictorRgba), 3, false);
        REQUIRE(rgb.has_value());
        CHECK(rgb->channels() == 3);
        CHECK(rgbAt(*rgb, 1, 0, 3) == std::array<unsigned char, 3>{0, 255, 0});

        auto grey = loader.load(bytesOf(kTiffTiledDeflateRgb), 1, false);
        REQUIRE(grey.has_value());
        CHECK(grey->channels() == 1);
        const auto& d = grey->data();
        CHECK(d[0] == (255 * 77) >> 8);// red, stb's luma weights
        CHECK(d[3] == 255);            // white
    }

    SECTION("file overload matches memory overload") {
        const auto path = writeBytes(kTiffLzwPredictorRgba, "threepp_imageloader_test.tif");
        auto fromFile = loader.load(path, 4, true);
        auto fromMemory = loader.load(bytesOf(kTiffLzwPredictorRgba), 4, true);
        REQUIRE(fromFile.has_value());
        REQUIRE(fromMemory.has_value());
        CHECK(fromFile->data() == fromMemory->data());
    }

    SECTION("truncated data is rejected, not read past") {
        for (size_t keep : {4, 8, 20, 100}) {
            std::vector<unsigned char> truncated(kTiffBigEndianPlanar16.begin(), kTiffBigEndianPlanar16.begin() + keep);
            CHECK_FALSE(loader.load(truncated, 4, false).has_value());
        }
    }
}

TEST_CASE("TextureLoader accepts a .webp file") {

    const auto path = writeBytes(kWebpLosslessRgba, "threepp_textureloader_test.webp");

    TextureLoader loader{false};
    auto texture = loader.load(path, ColorSpace::sRGB, false);

    REQUIRE(texture != nullptr);
    CHECK(texture->format == Format::RGBA);
    CHECK(texture->colorSpace == ColorSpace::sRGB);
    CHECK(texture->image().width() == 2);
    CHECK(texture->image().height() == 2);
}
