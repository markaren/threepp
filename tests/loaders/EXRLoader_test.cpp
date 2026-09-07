
#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/EXRLoader.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/utils/Base64.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// stb_image_write's deflate, which threepp already compiles (StbImageWrite.cpp).
// It has no declaration in the header's public section, hence the prototype;
// STBIWDEF resolves to extern "C" in a C++ build.
extern "C" unsigned char* stbi_zlib_compress(unsigned char* data, int data_len, int* out_len, int quality);

using namespace threepp;

namespace {

    // --- little-endian writers -------------------------------------------------

    void put8(std::vector<unsigned char>& v, uint8_t x) { v.push_back(x); }

    void put16(std::vector<unsigned char>& v, uint16_t x) {
        v.push_back(static_cast<unsigned char>(x & 0xff));
        v.push_back(static_cast<unsigned char>(x >> 8));
    }

    void put32(std::vector<unsigned char>& v, uint32_t x) {
        for (int i = 0; i < 4; ++i) v.push_back(static_cast<unsigned char>((x >> (i * 8)) & 0xff));
    }

    void put64(std::vector<unsigned char>& v, uint64_t x) {
        for (int i = 0; i < 8; ++i) v.push_back(static_cast<unsigned char>((x >> (i * 8)) & 0xff));
    }

    void putStr(std::vector<unsigned char>& v, const std::string& s) {
        v.insert(v.end(), s.begin(), s.end());
        v.push_back(0);
    }

    void putAttr(std::vector<unsigned char>& v, const std::string& name, const std::string& type,
                 const std::vector<unsigned char>& value) {
        putStr(v, name);
        putStr(v, type);
        put32(v, static_cast<uint32_t>(value.size()));
        v.insert(v.end(), value.begin(), value.end());
    }

    // Fixture-only float→half, for normal in-range values with a mantissa that
    // fits 10 bits — every value these fixtures use is a small multiple of 1/4.
    // Deliberately not a general converter: the awkward cases (subnormals, Inf,
    // NaN, the max half) are pinned as literal bit patterns instead, so a
    // symmetric bug here cannot hide one in the decoder.
    uint16_t toHalf(float f) {

        uint32_t bits;
        std::memcpy(&bits, &f, sizeof bits);
        const uint32_t sign = (bits >> 16) & 0x8000u;
        const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
        const uint32_t mant = bits & 0x7fffffu;

        if ((bits & 0x7fffffffu) == 0) return static_cast<uint16_t>(sign);// ±0

        REQUIRE(exp > 0);
        REQUIRE(exp < 31);
        REQUIRE((mant & 0x1fffu) == 0);

        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    }

    // --- the write side of the ZIP/RLE codecs ---------------------------------

    // Inverse of the reader's reconstruct(): split the stream into even/odd
    // halves, then delta-encode. Mirrors ImfZipCompressor::compress.
    std::vector<unsigned char> forwardTransform(const std::vector<unsigned char>& raw) {

        const size_t n = raw.size();
        std::vector<unsigned char> t(n);
        size_t i1 = 0, i2 = (n + 1) / 2;
        for (size_t i = 0; i < n; ++i) {
            if (i % 2 == 0) t[i1++] = raw[i];
            else t[i2++] = raw[i];
        }

        int p = n > 0 ? t[0] : 0;
        for (size_t i = 1; i < n; ++i) {
            const int d = static_cast<int>(t[i]) - p + (128 + 256);
            p = t[i];
            t[i] = static_cast<unsigned char>(d);
        }
        return t;
    }

    // Repeat-runs only. Enough for fixture data (constant after delta encoding)
    // and it keeps the encoder to something obviously correct.
    std::vector<unsigned char> rleCompress(const std::vector<unsigned char>& in) {

        std::vector<unsigned char> out;
        size_t i = 0;
        while (i < in.size()) {
            size_t run = 1;
            while (i + run < in.size() && in[i + run] == in[i] && run < 128) ++run;
            out.push_back(static_cast<unsigned char>(run - 1));// 0..127 => repeat run+1 times
            out.push_back(in[i]);
            i += run;
        }
        return out;
    }

    std::vector<unsigned char> zlibCompress(const std::vector<unsigned char>& in) {

        int outLen = 0;
        unsigned char* packed = stbi_zlib_compress(const_cast<unsigned char*>(in.data()),
                                                   static_cast<int>(in.size()), &outLen, 8);
        REQUIRE(packed != nullptr);
        std::vector<unsigned char> out(packed, packed + outLen);
        std::free(packed);
        return out;
    }

    // --- fixture builder -------------------------------------------------------

    enum Comp { NONE = 0, RLE = 1, ZIPS = 2, ZIP = 3 };

    struct ExrChannel {
        std::string name;    // channel list must be alphabetically sorted, per spec
        std::vector<uint16_t> samples;// width*height halves, first row first
    };

    struct ExrSpec {
        int width{};
        int height{};
        int compression{NONE};
        std::vector<ExrChannel> channels;
        uint32_t versionFlags{0};
        int xMin{0};
        int yMin{0};
    };

    std::vector<unsigned char> buildExr(const ExrSpec& spec) {

        const int linesPerBlock = spec.compression == ZIP ? 16 : 1;

        std::vector<unsigned char> chlist;
        for (const auto& c : spec.channels) {
            putStr(chlist, c.name);
            put32(chlist, 1);// HALF
            put8(chlist, 0); // pLinear
            put8(chlist, 0);
            put8(chlist, 0);
            put8(chlist, 0);// reserved
            put32(chlist, 1);// xSampling
            put32(chlist, 1);// ySampling
        }
        chlist.push_back(0);

        std::vector<unsigned char> box;
        put32(box, static_cast<uint32_t>(spec.xMin));
        put32(box, static_cast<uint32_t>(spec.yMin));
        put32(box, static_cast<uint32_t>(spec.xMin + spec.width - 1));
        put32(box, static_cast<uint32_t>(spec.yMin + spec.height - 1));

        std::vector<unsigned char> f1;
        put32(f1, 0x3f800000u);// 1.0f
        std::vector<unsigned char> v2zero;
        put32(v2zero, 0);
        put32(v2zero, 0);

        std::vector<unsigned char> out;
        put32(out, 0x01312f76u);
        put32(out, 2u | spec.versionFlags);

        putAttr(out, "channels", "chlist", chlist);
        putAttr(out, "compression", "compression", {static_cast<unsigned char>(spec.compression)});
        putAttr(out, "dataWindow", "box2i", box);
        putAttr(out, "displayWindow", "box2i", box);
        putAttr(out, "lineOrder", "lineOrder", {0});
        putAttr(out, "pixelAspectRatio", "float", f1);
        putAttr(out, "screenWindowCenter", "v2f", v2zero);
        putAttr(out, "screenWindowWidth", "float", f1);
        out.push_back(0);// end of header

        std::vector<std::vector<unsigned char>> payloads;
        std::vector<int32_t> ys;
        for (int first = 0; first < spec.height; first += linesPerBlock) {

            const int lines = std::min(linesPerBlock, spec.height - first);

            std::vector<unsigned char> raw;
            for (int line = 0; line < lines; ++line) {
                for (const auto& c : spec.channels) {
                    for (int x = 0; x < spec.width; ++x) {
                        put16(raw, c.samples[static_cast<size_t>(first + line) * spec.width + x]);
                    }
                }
            }

            std::vector<unsigned char> payload = raw;
            if (spec.compression != NONE) {
                const auto t = forwardTransform(raw);
                const auto packed = spec.compression == RLE ? rleCompress(t) : zlibCompress(t);
                // OpenEXR stores the block raw when the codec failed to shrink it,
                // and the reader distinguishes the two by size alone.
                if (packed.size() < raw.size()) payload = packed;
            }

            payloads.push_back(std::move(payload));
            ys.push_back(spec.yMin + first);
        }

        uint64_t offset = out.size() + 8ull * payloads.size();
        for (const auto& p : payloads) {
            put64(out, offset);
            offset += 8 + p.size();
        }
        for (size_t i = 0; i < payloads.size(); ++i) {
            put32(out, static_cast<uint32_t>(ys[i]));
            put32(out, static_cast<uint32_t>(payloads[i].size()));
            out.insert(out.end(), payloads[i].begin(), payloads[i].end());
        }

        return out;
    }

    // R = x + 1, G = 0.5, B = y / 4 — all exactly representable as halves, and
    // all distinct per row and column so a transposed or shifted decode shows up.
    ExrSpec gradientSpec(int compression, int width = 16, int height = 20) {

        ExrSpec spec;
        spec.width = width;
        spec.height = height;
        spec.compression = compression;

        ExrChannel b{"B", {}}, g{"G", {}}, r{"R", {}};
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                b.samples.push_back(toHalf(static_cast<float>(y) / 4.f));
                g.samples.push_back(toHalf(0.5f));
                r.samples.push_back(toHalf(static_cast<float>(x + 1)));
            }
        }
        spec.channels = {b, g, r};// alphabetical
        return spec;
    }

    const std::vector<float>& pixels(const std::shared_ptr<Texture>& t) {

        return t->image().data<float>();
    }

}// namespace

TEST_CASE("EXRLoader decodes an uncompressed scanline image") {

    EXRLoader loader;
    auto texture = loader.loadFromMemory(buildExr(gradientSpec(NONE)), "grad", false);
    REQUIRE(texture != nullptr);

    CHECK(texture->image().width() == 16u);
    CHECK(texture->image().height() == 20u);
    CHECK(texture->image().isFloat());
    CHECK(texture->format == Format::RGBA);
    CHECK(texture->type == Type::Float);
    CHECK(texture->colorSpace == ColorSpace::Linear);
    CHECK(texture->mapping == Mapping::EquirectangularReflection);

    const auto& p = pixels(texture);
    REQUIRE(p.size() == static_cast<size_t>(16 * 20 * 4));

    for (int y : {0, 1, 15, 16, 19}) {// straddles the 16-line ZIP block boundary
        for (int x : {0, 7, 15}) {
            const size_t i = (static_cast<size_t>(y) * 16 + x) * 4;
            CHECK(p[i + 0] == static_cast<float>(x + 1));
            CHECK(p[i + 1] == 0.5f);
            CHECK(p[i + 2] == static_cast<float>(y) / 4.f);
            CHECK(p[i + 3] == 1.f);// no A channel in the file
        }
    }
}

TEST_CASE("EXRLoader ZIP decodes bit-identically to the uncompressed encoding") {

    const auto plain = buildExr(gradientSpec(NONE));
    const auto zipped = buildExr(gradientSpec(ZIP));

    // Guards the test itself: if the fixture failed to compress, every block
    // would fall back to the raw escape and the inflate path would go untested.
    CHECK(zipped.size() < plain.size());

    EXRLoader loader;
    auto a = loader.loadFromMemory(plain, "a", false);
    auto b = loader.loadFromMemory(zipped, "b", false);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    CHECK(pixels(a) == pixels(b));
}

TEST_CASE("EXRLoader ZIPS and RLE decode bit-identically to the uncompressed encoding") {

    EXRLoader loader;
    auto plain = loader.loadFromMemory(buildExr(gradientSpec(NONE)), "a", false);
    REQUIRE(plain != nullptr);

    for (int compression : {ZIPS, RLE}) {
        auto packed = buildExr(gradientSpec(compression));
        CHECK(packed.size() < buildExr(gradientSpec(NONE)).size());
        auto decoded = loader.loadFromMemory(packed, "b", false);
        REQUIRE(decoded != nullptr);
        CHECK(pixels(decoded) == pixels(plain));
    }
}

TEST_CASE("EXRLoader converts half bit patterns exactly") {

    // Hand-written halves, including the cases the fixture converter refuses:
    // zero, the largest finite half, the smallest subnormal, and a negative.
    const std::vector<uint16_t> bits{0x0000, 0x3c00, 0x3800, 0xbc00,
                                     0x7bff, 0x0001, 0x0400, 0xc000};
    const std::vector<float> expected{0.f, 1.f, 0.5f, -1.f,
                                      65504.f, 5.9604645e-8f, 6.1035156e-5f, -2.f};

    ExrSpec spec;
    spec.width = static_cast<int>(bits.size());
    spec.height = 1;
    spec.compression = NONE;
    spec.channels = {ExrChannel{"R", bits}};

    EXRLoader loader;
    auto texture = loader.loadFromMemory(buildExr(spec), "halves", false);
    REQUIRE(texture != nullptr);

    const auto& p = pixels(texture);
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(p[i * 4] == expected[i]);
    }
}

TEST_CASE("EXRLoader flipY reverses row order") {

    EXRLoader loader;
    const auto data = buildExr(gradientSpec(ZIP));

    auto upright = loader.loadFromMemory(data, "a", false);
    auto flipped = loader.loadFromMemory(data, "b", true);
    REQUIRE(upright != nullptr);
    REQUIRE(flipped != nullptr);

    const auto& u = pixels(upright);
    const auto& f = pixels(flipped);
    for (int y = 0; y < 20; ++y) {
        const size_t src = static_cast<size_t>(y) * 16 * 4;
        const size_t dst = static_cast<size_t>(19 - y) * 16 * 4;
        CHECK(std::equal(u.begin() + src, u.begin() + src + 16 * 4, f.begin() + dst));
    }
}

TEST_CASE("EXRLoader reads an alpha channel when the file has one") {

    auto spec = gradientSpec(NONE, 4, 2);
    ExrChannel a{"A", {}};
    for (int i = 0; i < 8; ++i) a.samples.push_back(toHalf(0.25f));
    spec.channels.insert(spec.channels.begin(), a);// A sorts first

    EXRLoader loader;
    auto texture = loader.loadFromMemory(buildExr(spec), "alpha", false);
    REQUIRE(texture != nullptr);

    const auto& p = pixels(texture);
    for (size_t i = 0; i < 8; ++i) {
        CHECK(p[i * 4 + 3] == 0.25f);
        CHECK(p[i * 4 + 0] == static_cast<float>(i % 4 + 1));// RGB still land right
    }
}

TEST_CASE("EXRLoader broadcasts a luminance-only image to RGB") {

    ExrSpec spec;
    spec.width = 2;
    spec.height = 2;
    spec.compression = NONE;
    spec.channels = {ExrChannel{"Y", {toHalf(0.5f), toHalf(1.f), toHalf(2.f), toHalf(4.f)}}};

    EXRLoader loader;
    auto texture = loader.loadFromMemory(buildExr(spec), "lum", false);
    REQUIRE(texture != nullptr);

    const auto& p = pixels(texture);
    const std::vector<float> expected{0.5f, 1.f, 2.f, 4.f};
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(p[i * 4 + 0] == expected[i]);
        CHECK(p[i * 4 + 1] == expected[i]);
        CHECK(p[i * 4 + 2] == expected[i]);
        CHECK(p[i * 4 + 3] == 1.f);
    }
}

TEST_CASE("EXRLoader places pixels by dataWindow origin, not by chunk order") {

    auto spec = gradientSpec(ZIP);
    spec.xMin = 7;
    spec.yMin = 5;// chunks now claim scanlines 5..24

    EXRLoader loader;
    auto shifted = loader.loadFromMemory(buildExr(spec), "shifted", false);
    auto origin = loader.loadFromMemory(buildExr(gradientSpec(ZIP)), "origin", false);
    REQUIRE(shifted != nullptr);
    REQUIRE(origin != nullptr);

    CHECK(shifted->image().width() == 16u);
    CHECK(shifted->image().height() == 20u);
    CHECK(pixels(shifted) == pixels(origin));
}

TEST_CASE("EXRLoader rejects codecs it cannot decode instead of returning garbage") {

    EXRLoader loader;

    // PXR24, B44, B44A, DWAA, DWAB — everything past the four this loader
    // synthesizes and the PIZ file it embeds.
    for (int codec : {5, 6, 7, 8, 9}) {
        auto spec = gradientSpec(NONE);
        spec.compression = codec;
        CHECK(loader.loadFromMemory(buildExr(spec)) == nullptr);
    }
}

TEST_CASE("EXRLoader rejects a PIZ chunk that is not really PIZ") {

    // Tagged PIZ, but the chunks were written by the ZIP path. The Huffman
    // decoder has to fail closed on them rather than emit whatever the bytes
    // happen to decode to.
    auto spec = gradientSpec(NONE);
    spec.compression = 4;

    EXRLoader loader;
    CHECK(loader.loadFromMemory(buildExr(spec)) == nullptr);
}

TEST_CASE("EXRLoader rejects tiled, deep and multi-part files") {

    EXRLoader loader;
    for (uint32_t flag : {1u << 9, 1u << 11, 1u << 12}) {
        auto spec = gradientSpec(NONE);
        spec.versionFlags = flag;
        CHECK(loader.loadFromMemory(buildExr(spec)) == nullptr);
    }
}

TEST_CASE("EXRLoader rejects a foreign or truncated file") {

    EXRLoader loader;

    CHECK(loader.loadFromMemory({}) == nullptr);
    CHECK(loader.loadFromMemory(std::vector<unsigned char>{0x89u, 'P', 'N', 'G'}) == nullptr);

    const auto full = buildExr(gradientSpec(ZIP));
    for (double fraction : {0.25, 0.5, 0.9}) {
        std::vector<unsigned char> cut(full.begin(),
                                       full.begin() + static_cast<size_t>(full.size() * fraction));
        CHECK(loader.loadFromMemory(cut) == nullptr);
    }
}

TEST_CASE("EXRLoader rejects a file with no colour channels") {

    ExrSpec spec;
    spec.width = 2;
    spec.height = 1;
    spec.compression = NONE;
    spec.channels = {ExrChannel{"Z", {toHalf(1.f), toHalf(2.f)}}};

    EXRLoader loader;
    CHECK(loader.loadFromMemory(buildExr(spec)) == nullptr);
}

// A PIZ file written by OpenEXR itself, base64'd.
//
// The codecs above are synthesized in-test because their write side is a few
// lines each. PIZ's is a wavelet plus a canonical-Huffman coder — several
// hundred lines — and a fixture produced by our own encoder would only prove the
// two halves agree with each other, not that either matches the format. So this
// one comes from outside.
//
// It is 16 x 40 RGBA half, 40 rows so it spans two 32-line PIZ blocks with the
// second one partial, and every value is exactly representable as a half:
//
//     R = x/16,  G = y/64,  B = ((3x + y) % 16)/16,  A = 1
//
// To regenerate: in Blender, make a 16x40 float image with those values (its
// pixel buffer is bottom-up), set colorspace Non-Color and view transform Raw,
// then save as OPEN_EXR with codec PIZ and colour depth 16, and base64 the file.
namespace {

    const char* PIZ_FIXTURE_BASE64 =
        "di8xAQIAAABTb2Z0d2FyZQBzdHJpbmcADQAAAEJsZW5kZXIgNS4wLjFjaGFubmVscwBjaGxpc3QASQAAAEEAAQAAAAAAAAAB"
        "AAAAAQAAAEIAAQAAAAAAAAABAAAAAQAAAEcAAQAAAAAAAAABAAAAAQAAAFIAAQAAAAAAAAABAAAAAQAAAABjb2xvckludGVy"
        "b3BJRABzdHJpbmcAEAAAAGxpbl9yZWM3MDlfc2NlbmVjb21wcmVzc2lvbgBjb21wcmVzc2lvbgABAAAABGRhdGFXaW5kb3cA"
        "Ym94MmkAEAAAAAAAAAAAAAAADwAAACcAAABkaXNwbGF5V2luZG93AGJveDJpABAAAAAAAAAAAAAAAA8AAAAnAAAAbGluZU9y"
        "ZGVyAGxpbmVPcmRlcgABAAAAAHBpeGVsQXNwZWN0UmF0aW8AZmxvYXQABAAAAAAAgD9zY3JlZW5XaW5kb3dDZW50ZXIAdjJm"
        "AAgAAAAAAAAAAAAAAHNjcmVlbldpbmRvd1dpZHRoAGZsb2F0AAQAAAAAAIA/eERlbnNpdHkAZmxvYXQABAAAAAAAkEIAvQEA"
        "AAAAAADVCAAAAAAAAAAAAAAQBwAAgASABwEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAA"
        "AAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAA"
        "AAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAA"
        "AAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAAAAAAA"
        "AAEAAAAAAAAAAQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAA"
        "AAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAA"
        "AAEHBAAAAAAAAAAAAQDZAQAAyxAAAAAAAAAEgAYgAgr4jwcAbwkr8BH/ASv/////////////////////////////////////"
        "////////////////////////////////////////////////////////////////////////////////////////////////"
        "////////////////////////////////////////////////////////////////////////////////////////////////"
        "////////////////////////////////////////////////////////////////////////////////////////////////"
        "////////////////////////////////////////////////////////////////////////////////////////////////"
        "////////////////////////////////////////////////////////////////////////////////////////////////"
        "///////////////////////////////////////////////////////////////////HG/DCQB+x+xxfBQBxxBBAygCgH/AF"
        "AP+ARApRQYGSgwRCCUOiDDndRTuIIaKEBSAiBiARQ4M7qKdzogw6EEoUgUooMDIqDBDudEGHO6igRQ4MQQ0UICkBEDERBhzu"
        "op3PQYIhBKGQKUUGBkdRTudEGHOCIGIBFDgxBDRQgKRzogw53UU4oMDIqDBEIJQpApRhzuop3OiKEBSAiBiARQ4MQQ0UU7nR"
        "BhzuAiBSigwMlBgiEEodEGHO6incQQ0UICkBEDEAihwZ3UU7nRBh0IJQpApRQYGRUGCHc6IMOd1FAihwYghooQFICIGIiDDn"
        "dRTuegwRCCUMgUooMDI6inc6IMOcEQMQCKHBiCGihAUjnRBhzuopxQYGRUGCIQShSBSjDndRTudEUICkBEDEAihwYghoop3O"
        "iDDncAP/+7u7u7r169e7u7u7n/P+7u7u7r169e7u7u7l//7u7u7uvXr17u7u7uf8/7u7u7uvXr17u7u7uAv/+7u7u7r169e7"
        "u7u7n/P+7u7u7r169e7u7u7l//7u7u7uvXr17u7u7uf8/7u7u7uvXr17u7u7uAJlMMZTAbVm1f//znOd3d///MpzKdq7V///"
        "Oc53d3//8ymGMp2rNq///nOc7u7//+ZTmU7V2r//+c5zu7v//wBMphjKYDas2r//+c5zu7v//5lOZTtXav//5znO7u///mUw"
        "xlO1ZtX//85znd3f//zKcynau1f//znOd3d//+AgAAAAAAQAAAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8AAAA"
        "MgA2gDgAOoA7ADAANQA4gDkAOwAsADQANwA5gDoAOAA4ADgAOAA4ADgAOAA4ADgAOAA4ADgAOAA4ADgAOAAAACwAMAAyADQA"
        "NQA2ADcAOIA4ADmAOQA6gDoAO4A7ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwALAA0ADcAOYA6AAAAMgA2gDgA"
        "OoA7ADAANQA4gDkAOyA4IDggOCA4IDggOCA4IDggOCA4IDggOCA4IDggOCA4AAAALAAwADIANAA1ADYANwA4gDgAOYA5ADqA"
        "OgA7gDsAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAAwADUAOIA5ADsALAA0ADcAOYA6AAAAMgA2gDgAOoA7QDhA"
        "OEA4QDhAOEA4QDhAOEA4QDhAOEA4QDhAOEA4QDgAAAAsADAAMgA0ADUANgA3ADiAOAA5gDkAOoA6ADuAOwA8ADwAPAA8ADwA"
        "PAA8ADwAPAA8ADwAPAA8ADwAPAA8ADIANoA4ADqAOwAwADUAOIA5ADsALAA0ADcAOYA6AABgOGA4YDhgOGA4YDhgOGA4YDhg"
        "OGA4YDhgOGA4YDhgOAAAACwAMAAyADQANQA2ADcAOIA4ADmAOQA6gDoAO4A7ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwA"
        "PAA8ADwANAA3ADmAOgAAADIANoA4ADqAOwAwADUAOIA5ADsALIA4gDiAOIA4gDiAOIA4gDiAOIA4gDiAOIA4gDiAOIA4AAAA"
        "LAAwADIANAA1ADYANwA4gDgAOYA5ADqAOgA7gDsAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA1ADiAOQA7ACwA"
        "NAA3ADmAOgAAADIANoA4ADqAOwAwoDigOKA4oDigOKA4oDigOKA4oDigOKA4oDigOKA4oDgAAAAsADAAMgA0ADUANgA3ADiA"
        "OAA5gDkAOoA6ADuAOwA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADaAOAA6gDsAMAA1ADiAOQA7ACwANAA3ADmA"
        "OgAAADLAOMA4wDjAOMA4wDjAOMA4wDjAOMA4wDjAOMA4wDjAOAAAACwAMAAyADQANQA2ADcAOIA4ADmAOQA6gDoAO4A7ADwA"
        "PAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8ADwANwA5gDoAAAAyADaAOAA6gDsAMAA1ADiAOQA7ACwANOA44DjgOOA44Djg"
        "OOA44DjgOOA44DjgOOA44DjgOOA4AAAALAAwADIANAA1ADYANwA4gDgAOYA5ADqAOgA7gDs=";

}// namespace

TEST_CASE("EXRLoader decodes a PIZ file written by OpenEXR") {

    const auto data = utils::base64Decode(PIZ_FIXTURE_BASE64);

    EXRLoader loader;
    auto texture = loader.loadFromMemory(data, "piz", false);
    REQUIRE(texture != nullptr);

    REQUIRE(texture->image().width() == 16u);
    REQUIRE(texture->image().height() == 40u);

    const auto& p = pixels(texture);
    REQUIRE(p.size() == static_cast<size_t>(16 * 40 * 4));

    // Every texel, not a sample: a wavelet fault shows up as a block or a
    // quadrant, and the partial second block is exactly where it would hide.
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 16; ++x) {
            const size_t i = (static_cast<size_t>(y) * 16 + x) * 4;
            CHECK(p[i + 0] == static_cast<float>(x) / 16.f);
            CHECK(p[i + 1] == static_cast<float>(y) / 64.f);
            CHECK(p[i + 2] == static_cast<float>((3 * x + y) % 16) / 16.f);
            CHECK(p[i + 3] == 1.f);
        }
    }
}

TEST_CASE("EXRLoader survives a corrupted PIZ file") {

    // The wavelet and Huffman decoders index tables and step pointers with
    // values that come straight out of the file, which is the part of this
    // loader most likely to run off the end of a buffer. Nothing here asserts
    // what a mangled file decodes to — only that the loader either refuses it
    // or returns a whole, correctly-sized image, and does not crash doing so.
    const auto full = utils::base64Decode(PIZ_FIXTURE_BASE64);

    EXRLoader loader;

    auto check = [](const std::shared_ptr<Texture>& t) {
        if (!t) return;
        CHECK(t->image().width() == 16u);
        CHECK(t->image().height() == 40u);
        CHECK(t->image().data<float>().size() == static_cast<size_t>(16 * 40 * 4));
    };

    for (size_t cut = 1; cut < full.size(); cut += 7) {
        const std::vector<unsigned char> truncated(full.begin(), full.begin() + cut);
        check(loader.loadFromMemory(truncated));
    }

    for (size_t i = 0; i < full.size(); i += 11) {
        auto data = full;
        data[i] = static_cast<unsigned char>(data[i] ^ 0xffu);
        check(loader.loadFromMemory(data));
    }
}
