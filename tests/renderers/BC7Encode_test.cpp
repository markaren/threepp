
#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/common/BC7Encode.hpp"
#include "threepp/renderers/common/BCnDecode.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace threepp;

namespace {

    constexpr unsigned int GL_BC7 = 0x8E8Cu;

    // Encode → decode with the renderer's own BCnDecode, return max per-channel
    // absolute error. This pins the encoder's palette math (kW4 weights, P-bit
    // reconstruction, anchor swap) against the decoder it must agree with.
    int roundTripMaxError(const std::vector<std::uint8_t>& rgba, int w, int h) {
        const auto blocks = bcn::bc7EncodeMode6(rgba.data(), w, h);
        const auto back = bcn::bcnDecompress(blocks.data(), w, h, GL_BC7);
        REQUIRE_FALSE(back.empty());
        REQUIRE(back.size() == rgba.size());
        int maxErr = 0;
        for (size_t i = 0; i < rgba.size(); ++i) {
            maxErr = std::max(maxErr, std::abs(int(rgba[i]) - int(back[i])));
        }
        return maxErr;
    }

}// namespace

TEST_CASE("BC7 mode 6 reproduces flat blocks near-exactly") {

    std::vector<std::uint8_t> img(8 * 8 * 4);
    for (size_t i = 0; i < img.size(); i += 4) {
        img[i + 0] = 180;
        img[i + 1] = 90;
        img[i + 2] = 40;
        img[i + 3] = 255;
    }
    // Flat colour: both endpoints land on (or bracket) the value; error is at
    // most the 7-bit+P quantization step.
    CHECK(roundTripMaxError(img, 8, 8) <= 2);
}

// A gradient along ONE direction is colinear in colour space — exactly what a
// single-subset palette represents. This is the discriminating quality case:
// a correct encoder resolves it to a few codes; endpoint/index bugs blow it up.
TEST_CASE("BC7 mode 6 nails colinear gradients") {

    const int w = 16, h = 16;
    std::vector<std::uint8_t> img(w * h * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            auto* p = img.data() + (y * w + x) * 4;
            const int v = x * 255 / (w - 1);
            p[0] = static_cast<std::uint8_t>(v);
            p[1] = static_cast<std::uint8_t>(255 - v);
            p[2] = static_cast<std::uint8_t>(v / 2 + 60);
            p[3] = 255;
        }
    }
    CHECK(roundTripMaxError(img, w, h) <= 8);
}

TEST_CASE("BC7 mode 6 bounds 2-axis gradients at the single-subset limit") {

    const int w = 16, h = 16;
    std::vector<std::uint8_t> img(w * h * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            auto* p = img.data() + (y * w + x) * 4;
            p[0] = static_cast<std::uint8_t>(x * 255 / (w - 1));
            p[1] = static_cast<std::uint8_t>(y * 255 / (h - 1));
            p[2] = 128;
            p[3] = 255;
        }
    }
    // R varies with x and G with y inside each 4x4 block: the texels form a
    // 2D grid in colour space, and ANY one-subset palette (a line) carries
    // ~half the off-axis span as irreducible error (~±17 here). Multi-mode
    // encoders switch to 2-subset modes for such blocks; mode 6 cannot. This
    // pins the error at the format limit, not at zero.
    CHECK(roundTripMaxError(img, w, h) <= 32);
}

TEST_CASE("BC7 mode 6 preserves hard alpha edges usably") {

    const int w = 8, h = 8;
    std::vector<std::uint8_t> img(w * h * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            auto* p = img.data() + (y * w + x) * 4;
            const bool solid = x < w / 2;
            p[0] = 60;
            p[1] = 200;
            p[2] = 60;
            p[3] = solid ? 255 : 0;
        }
    }
    const auto blocks = bcn::bc7EncodeMode6(img.data(), w, h);
    const auto back = bcn::bcnDecompress(blocks.data(), w, h, GL_BC7);
    REQUIRE_FALSE(back.empty());
    // Foliage-cutout requirement: solid texels stay above any sane cutoff,
    // clear texels stay below. (Exact endpoint values may shift ±few codes.)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::uint8_t a = back[(y * w + x) * 4 + 3];
            if (x < w / 2) {
                CHECK(a > 200);
            } else {
                CHECK(a < 55);
            }
        }
    }
}

TEST_CASE("BC7 mode 6 on noise stays within mode-6 quality bounds") {

    const int w = 32, h = 32;
    std::vector<std::uint8_t> img(w * h * 4);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& v : img) v = static_cast<std::uint8_t>(d(rng));

    const auto blocks = bcn::bc7EncodeMode6(img.data(), w, h);
    const auto back = bcn::bcnDecompress(blocks.data(), w, h, GL_BC7);
    REQUIRE_FALSE(back.empty());

    // Pure noise is the worst case for any block codec — check RMSE, not max.
    double se = 0;
    for (size_t i = 0; i < img.size(); ++i) {
        const double e = double(img[i]) - double(back[i]);
        se += e * e;
    }
    const double rmse = std::sqrt(se / double(img.size()));
    // Uniform 4-channel noise projected onto a 1D palette line keeps
    // sqrt(3/4) of the per-channel deviation (~64 RMSE) as irreducible
    // residual; the 16-entry palette + refinement lands ~55. Prove we sit at
    // the format floor, not above it.
    CHECK(rmse < 60.0);
}

TEST_CASE("BC7 encoder handles non-multiple-of-4 dimensions") {

    const int w = 7, h = 5;
    std::vector<std::uint8_t> img(w * h * 4);
    for (int i = 0; i < w * h; ++i) {
        img[i * 4 + 0] = static_cast<std::uint8_t>(i * 7);
        img[i * 4 + 1] = static_cast<std::uint8_t>(255 - i * 5);
        img[i * 4 + 2] = static_cast<std::uint8_t>(i * 11);
        img[i * 4 + 3] = 255;
    }
    const auto blocks = bcn::bc7EncodeMode6(img.data(), w, h);
    CHECK(blocks.size() == 2u * 2u * 16u);// ceil(7/4) x ceil(5/4) blocks
    const auto back = bcn::bcnDecompress(blocks.data(), w, h, GL_BC7);
    REQUIRE(back.size() == img.size());
}

TEST_CASE("Mip chain halves dimensions down to 1x1") {

    std::vector<std::uint8_t> img(16 * 8 * 4, 128);
    const auto mips = bcn::buildMipChainRGBA8(img.data(), 16, 8, false);

    // 16x8 -> 8x4 -> 4x2 -> 2x1 -> 1x1
    REQUIRE(mips.size() == 4);
    CHECK(mips[0].size() == 8u * 4u * 4u);
    CHECK(mips[3].size() == 1u * 1u * 4u);
    // Flat input stays flat through every level.
    CHECK(mips[3][0] == 128);
    CHECK(mips[3][3] == 128);
}

TEST_CASE("sRGB mip filtering averages in linear space") {

    // 2x1 image: black and white sRGB texels. A naive byte average gives 128;
    // linear-space averaging gives ~188 (0.5 linear back to sRGB).
    std::vector<std::uint8_t> img = {0, 0, 0, 255, 255, 255, 255, 255};
    const auto mips = bcn::buildMipChainRGBA8(img.data(), 2, 1, true);
    REQUIRE(mips.size() == 1);
    CHECK(mips[0][0] > 180);
    CHECK(mips[0][0] < 195);
    // Alpha is linear regardless.
    CHECK(mips[0][3] == 255);
}
