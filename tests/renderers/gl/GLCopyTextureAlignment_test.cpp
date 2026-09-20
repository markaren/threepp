// copyTextureToImage on a row stride that is not 4-byte aligned.
//
// The destination is resized to exactly width * height * channels — tightly
// packed. GL's default GL_PACK_ALIGNMENT is 4, so for any format and width whose
// row stride is not a multiple of 4 (RGB at any odd width), glGetTexImage pads
// every row out to the next 4-byte boundary. It therefore writes PAST the end of
// the vector, and what does land inside it is skewed by the growing padding.
//
// readPixels a few lines above already sets PACK_ALIGNMENT to 1 and says why;
// copyTextureToImage was missed.
//
// 5 x 4 RGB: rows are 15 bytes, so GL pads to 16 and wants 3*16 + 15 = 63 bytes
// against a 60-byte buffer — three bytes over, and rows 1..3 shifted by 1, 2 and
// 3 bytes respectively.

#include "gl_test_helpers.hpp"

namespace {

    constexpr int W = 5, H = 4, C = 3;

    // A pattern where every texel is distinguishable, so a one-byte shift is
    // unmistakable rather than plausible.
    std::vector<unsigned char> pattern() {
        std::vector<unsigned char> d(static_cast<size_t>(W) * H * C);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t i = (static_cast<size_t>(y) * W + x) * C;
                d[i + 0] = static_cast<unsigned char>(10 + x * 40);
                d[i + 1] = static_cast<unsigned char>(20 + y * 50);
                d[i + 2] = static_cast<unsigned char>(200 - x * 20 - y * 10);
            }
        return d;
    }

}// namespace

TEST_CASE("copyTextureToImage round-trips a non-4-aligned row stride") {

    const auto expected = pattern();

    auto tex = Texture::create(Image(std::vector<unsigned char>(expected), W, H));
    tex->format = Format::RGB;
    tex->generateMipmaps = false;
    tex->minFilter = Filter::Nearest;
    tex->magFilter = Filter::Nearest;
    // The upload side of the same convention: tightly packed source rows.
    tex->unpackAlignment = 1;
    tex->needsUpdate();

    GLRenderer renderer(glCanvas());
    renderer.copyTextureToImage(*tex);

    const auto& got = tex->image().data();
    REQUIRE(got.size() == expected.size());

    int wrong = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(int(got[i]) - int(expected[i])) > 1) ++wrong;
    }

    INFO("bytes differing after the round trip: " << wrong << " / " << expected.size());
    CHECK(wrong == 0);
}
