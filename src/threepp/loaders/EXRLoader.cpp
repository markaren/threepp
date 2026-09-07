
#include "threepp/loaders/EXRLoader.hpp"

#include "threepp/loaders/HdrTexture.hpp"
#include "threepp/loaders/exr/PizDecode.hpp"

// stb_image.h is already compiled via ImageLoader.cpp — declared here only for
// stbi_zlib_decode_buffer. EXR's ZIP/ZIPS chunks are ordinary zlib streams, so
// the inflate that PNG decoding already brings into the build is the whole of
// the dependency; no zlib/miniz of our own.
#include "stb_image.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr uint32_t EXR_MAGIC = 0x01312f76u;

    // Bits above the low-byte version number in the version field.
    constexpr uint32_t FLAG_TILED = 1u << 9;
    constexpr uint32_t FLAG_DEEP = 1u << 11;
    constexpr uint32_t FLAG_MULTIPART = 1u << 12;

    // Guard rails on an untrusted header. 64k on a side is far past any HDRI,
    // and the pixel cap keeps width*height*4 floats inside 4 GB.
    constexpr int64_t MAX_DIM = 65536;
    constexpr int64_t MAX_PIXELS = 1ll << 28;

    // Every attribute and channel name has a documented 255-byte ceiling (31
    // without the long-name flag); the cap exists so a file missing its
    // terminator cannot grow a string until the process dies.
    constexpr size_t MAX_NAME = 255;

    enum class Compression {
        None = 0,
        Rle = 1,
        Zips = 2,
        Zip = 3,
        Piz = 4,
        Pxr24 = 5,
        B44 = 6,
        B44A = 7,
        Dwaa = 8,
        Dwab = 9
    };

    const char* compressionName(Compression c) {

        switch (c) {
            case Compression::None: return "NONE";
            case Compression::Rle: return "RLE";
            case Compression::Zips: return "ZIPS";
            case Compression::Zip: return "ZIP";
            case Compression::Piz: return "PIZ";
            case Compression::Pxr24: return "PXR24";
            case Compression::B44: return "B44";
            case Compression::B44A: return "B44A";
            case Compression::Dwaa: return "DWAA";
            case Compression::Dwab: return "DWAB";
        }
        return "unknown";
    }

    // How many scanlines one chunk covers. Fixed by the codec, not by the file.
    int scanLinesPerBlock(Compression c) {

        switch (c) {
            case Compression::None:
            case Compression::Rle:
            case Compression::Zips: return 1;
            case Compression::Zip:
            case Compression::Pxr24: return 16;
            case Compression::Piz:
            case Compression::B44:
            case Compression::B44A:
            case Compression::Dwaa: return 32;
            case Compression::Dwab: return 256;
        }
        return 0;
    }

    enum class PixelType { Uint = 0, Half = 1, Float = 2 };

    size_t bytesPerSample(PixelType t) {

        return t == PixelType::Half ? 2 : 4;
    }

    struct Channel {
        std::string name;
        PixelType type{PixelType::Half};
        int32_t xSampling{1};
        int32_t ySampling{1};
    };

    float halfToFloat(uint16_t h) {

        const uint32_t sign = static_cast<uint32_t>(h >> 15) & 1u;
        int32_t exp = (h >> 10) & 0x1f;
        uint32_t mant = h & 0x3ffu;

        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign << 31;// ±0
            } else {
                // Subnormal half, normal float: shift the mantissa left until the
                // implicit leading 1 appears and pay for each shift in the exponent.
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                ++exp;
                mant &= ~0x400u;
                bits = (sign << 31) | (static_cast<uint32_t>(exp + (127 - 15)) << 23) | (mant << 13);
            }
        } else if (exp == 0x1f) {
            bits = (sign << 31) | 0x7f800000u | (mant << 13);// Inf / NaN
        } else {
            bits = (sign << 31) | (static_cast<uint32_t>(exp + (127 - 15)) << 23) | (mant << 13);
        }

        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    }

    float sampleToFloat(const unsigned char* p, PixelType t) {

        if (t == PixelType::Half) {
            uint16_t h;
            std::memcpy(&h, p, sizeof h);
            return halfToFloat(h);
        }
        if (t == PixelType::Float) {
            float f;
            std::memcpy(&f, p, sizeof f);
            return f;
        }
        uint32_t u;
        std::memcpy(&u, p, sizeof u);
        return static_cast<float>(u);
    }

    // Bounds-checked cursor over the file bytes. Every read is guarded and sets
    // `ok` once instead of throwing, so the parse can run to a single failure
    // check at each level rather than testing every field.
    //
    // EXR is defined little-endian and so is every platform threepp builds for;
    // the memcpy reads assume it.
    struct ByteReader {

        const unsigned char* data;
        size_t size;
        size_t pos{0};
        bool ok{true};

        bool need(size_t n) {
            if (!ok || n > size - std::min(pos, size)) {
                ok = false;
                return false;
            }
            return true;
        }

        uint8_t u8() {
            if (!need(1)) return 0;
            return data[pos++];
        }

        uint32_t u32() {
            if (!need(4)) return 0;
            uint32_t v;
            std::memcpy(&v, data + pos, sizeof v);
            pos += 4;
            return v;
        }

        int32_t i32() { return static_cast<int32_t>(u32()); }

        uint64_t u64() {
            if (!need(8)) return 0;
            uint64_t v;
            std::memcpy(&v, data + pos, sizeof v);
            pos += 8;
            return v;
        }

        void skip(size_t n) {
            if (need(n)) pos += n;
        }

        // Null-terminated string. An empty result with ok==true is the
        // terminator that ends both the header and the channel list.
        std::string str() {
            std::string s;
            while (need(1)) {
                const char c = static_cast<char>(data[pos++]);
                if (c == '\0') return s;
                if (s.size() >= MAX_NAME) {
                    ok = false;
                    break;
                }
                s.push_back(c);
            }
            return {};
        }
    };

    struct ExrHeader {
        std::vector<Channel> channels;
        Compression compression{Compression::None};
        int32_t xMin{0}, yMin{0}, xMax{-1}, yMax{-1};
        bool hasChannels{false};
        bool hasCompression{false};
        bool hasDataWindow{false};
    };

    bool parseChannelList(ByteReader& r, size_t end, ExrHeader& h) {

        while (r.pos < end) {
            const std::string name = r.str();
            if (!r.ok) return false;
            if (name.empty()) return true;// channel-list terminator

            Channel c;
            c.name = name;
            const int32_t type = r.i32();
            if (type < 0 || type > 2) {
                std::cerr << "[EXRLoader] channel '" << name << "' has unknown pixel type " << type << std::endl;
                return false;
            }
            c.type = static_cast<PixelType>(type);
            r.skip(4);// pLinear byte + 3 reserved
            c.xSampling = r.i32();
            c.ySampling = r.i32();
            if (!r.ok) return false;
            h.channels.push_back(std::move(c));
        }
        return false;// ran off the end of the attribute without a terminator
    }

    bool parseHeader(ByteReader& r, ExrHeader& h) {

        for (;;) {
            const std::string name = r.str();
            if (!r.ok) return false;
            if (name.empty()) return true;// header terminator

            const std::string type = r.str();
            const int32_t size = r.i32();
            if (!r.ok || size < 0) return false;

            const size_t end = r.pos + static_cast<size_t>(size);
            if (end > r.size) return false;

            if (name == "channels" && type == "chlist") {
                if (!parseChannelList(r, end, h)) return false;
                h.hasChannels = true;
            } else if (name == "compression" && type == "compression") {
                h.compression = static_cast<Compression>(r.u8());
                h.hasCompression = true;
            } else if (name == "dataWindow" && type == "box2i") {
                h.xMin = r.i32();
                h.yMin = r.i32();
                h.xMax = r.i32();
                h.yMax = r.i32();
                h.hasDataWindow = true;
            }

            // Resync to the attribute's declared end whether or not it was one we
            // read — unknown attributes are the common case and must be skipped
            // by their own size, not by guessing their layout.
            r.pos = end;
            if (!r.ok) return false;
        }
    }

    // Reverse of ImfZipCompressor's write-side transform, in the same order:
    // delta-decode the byte stream, then re-interleave the two halves it was
    // split into. Applied to ZIP, ZIPS and RLE chunks alike.
    void reconstruct(std::vector<unsigned char>& buf, std::vector<unsigned char>& scratch) {

        const size_t n = buf.size();
        if (n == 0) return;

        for (size_t i = 1; i < n; ++i) {
            const int d = static_cast<int>(buf[i - 1]) + static_cast<int>(buf[i]) - 128;
            buf[i] = static_cast<unsigned char>(d);
        }

        scratch.resize(n);
        const unsigned char* t1 = buf.data();
        const unsigned char* t2 = buf.data() + (n + 1) / 2;
        for (size_t i = 0; i < n; ++i) {
            scratch[i] = (i % 2 == 0) ? *(t1++) : *(t2++);
        }
        buf.swap(scratch);
    }

    // ImfRle.cpp's rleUncompress. A negative count byte means "the next -count
    // bytes are literal", a non-negative one means "repeat the next byte
    // count+1 times".
    bool rleUncompress(const unsigned char* in, size_t inLength, std::vector<unsigned char>& out) {

        size_t written = 0;
        while (inLength > 0) {
            const int8_t control = static_cast<int8_t>(*in++);
            --inLength;

            if (control < 0) {
                const size_t count = static_cast<size_t>(-static_cast<int>(control));
                if (count > inLength || written + count > out.size()) return false;
                std::memcpy(out.data() + written, in, count);
                written += count;
                in += count;
                inLength -= count;
            } else {
                const size_t count = static_cast<size_t>(control) + 1;
                if (inLength < 1 || written + count > out.size()) return false;
                std::memset(out.data() + written, *in, count);
                written += count;
                ++in;
                --inLength;
            }
        }
        return written == out.size();
    }

    struct DecodedExr {
        std::vector<float> rgba;
        unsigned int width{};
        unsigned int height{};
    };

    std::optional<DecodedExr> decode(const unsigned char* bytes, size_t size, bool flipY) {

        ByteReader r{bytes, size};

        if (r.u32() != EXR_MAGIC || !r.ok) {
            std::cerr << "[EXRLoader] not an OpenEXR file (bad magic)" << std::endl;
            return std::nullopt;
        }

        const uint32_t version = r.u32();
        if (!r.ok) return std::nullopt;
        if (version & FLAG_TILED) {
            std::cerr << "[EXRLoader] tiled EXR files are not supported" << std::endl;
            return std::nullopt;
        }
        if (version & (FLAG_DEEP | FLAG_MULTIPART)) {
            std::cerr << "[EXRLoader] deep / multi-part EXR files are not supported" << std::endl;
            return std::nullopt;
        }

        ExrHeader h;
        if (!parseHeader(r, h)) {
            std::cerr << "[EXRLoader] malformed header" << std::endl;
            return std::nullopt;
        }
        if (!h.hasChannels || !h.hasCompression || !h.hasDataWindow) {
            std::cerr << "[EXRLoader] header is missing channels, compression or dataWindow" << std::endl;
            return std::nullopt;
        }

        const int linesPerBlock = scanLinesPerBlock(h.compression);
        if (linesPerBlock == 0 ||
            (h.compression != Compression::None && h.compression != Compression::Rle &&
             h.compression != Compression::Zips && h.compression != Compression::Zip &&
             h.compression != Compression::Piz)) {
            std::cerr << "[EXRLoader] unsupported compression: " << compressionName(h.compression)
                      << " (supported: NONE, RLE, ZIPS, ZIP, PIZ)" << std::endl;
            return std::nullopt;
        }

        const int64_t width = static_cast<int64_t>(h.xMax) - h.xMin + 1;
        const int64_t height = static_cast<int64_t>(h.yMax) - h.yMin + 1;
        if (width <= 0 || height <= 0 || width > MAX_DIM || height > MAX_DIM || width * height > MAX_PIXELS) {
            std::cerr << "[EXRLoader] implausible dataWindow: " << width << 'x' << height << std::endl;
            return std::nullopt;
        }

        // Which output component each channel feeds. -1 = present in the file but
        // not something a texture wants (Z, object IDs, an AOV layer); its bytes
        // still have to be stepped over.
        std::vector<int> component(h.channels.size(), -1);
        bool haveRgb = false, haveY = false;
        for (size_t i = 0; i < h.channels.size(); ++i) {
            const auto& c = h.channels[i];
            if (c.xSampling != 1 || c.ySampling != 1) {
                std::cerr << "[EXRLoader] channel '" << c.name << "' is subsampled; not supported" << std::endl;
                return std::nullopt;
            }
            if (c.name == "R") { component[i] = 0; haveRgb = true; }
            else if (c.name == "G") { component[i] = 1; haveRgb = true; }
            else if (c.name == "B") { component[i] = 2; haveRgb = true; }
            else if (c.name == "A") { component[i] = 3; }
            else if (c.name == "Y") { component[i] = 4; haveY = true; }// luminance → broadcast
        }
        if (!haveRgb && !haveY) {
            std::string found;
            for (const auto& c : h.channels) found += (found.empty() ? "" : ", ") + c.name;
            std::cerr << "[EXRLoader] no R/G/B or Y channel; file has: " << found << std::endl;
            return std::nullopt;
        }
        if (haveRgb && haveY) {
            // Both would write the same components; RGB wins and Y is skipped
            // rather than racing it to the last write.
            for (size_t i = 0; i < component.size(); ++i) {
                if (component[i] == 4) component[i] = -1;
            }
        }

        // Byte offset of each channel inside one scanline. Channels are stored
        // whole-row at a time, in chlist order, not interleaved per pixel.
        std::vector<size_t> channelOffset(h.channels.size(), 0);
        std::vector<int> shortsPerSample(h.channels.size(), 0);// PIZ codes 16-bit words
        size_t bytesPerLine = 0;
        for (size_t i = 0; i < h.channels.size(); ++i) {
            channelOffset[i] = bytesPerLine;
            shortsPerSample[i] = static_cast<int>(bytesPerSample(h.channels[i].type) / sizeof(uint16_t));
            bytesPerLine += static_cast<size_t>(width) * bytesPerSample(h.channels[i].type);
        }
        // stbi_zlib_decode_buffer takes int lengths, and a block is the unit it
        // decodes. Reachable only with an absurd channel count, but that is
        // exactly what a crafted header would carry.
        if (bytesPerLine * static_cast<size_t>(linesPerBlock) > static_cast<size_t>(INT32_MAX)) {
            std::cerr << "[EXRLoader] scanline block too large (" << h.channels.size() << " channels)" << std::endl;
            return std::nullopt;
        }

        const int64_t numBlocks = (height + linesPerBlock - 1) / linesPerBlock;
        std::vector<uint64_t> offsets(static_cast<size_t>(numBlocks));
        for (auto& o : offsets) o = r.u64();
        if (!r.ok) {
            std::cerr << "[EXRLoader] truncated chunk offset table" << std::endl;
            return std::nullopt;
        }

        DecodedExr result;
        result.width = static_cast<unsigned int>(width);
        result.height = static_cast<unsigned int>(height);
        result.rgba.assign(static_cast<size_t>(width * height) * 4, 0.f);
        // Prefilled so a file without an A channel — the usual case for a sky —
        // comes out opaque rather than fully transparent.
        for (size_t i = 3; i < result.rgba.size(); i += 4) result.rgba[i] = 1.f;

        std::vector<unsigned char> block;
        std::vector<unsigned char> scratch;

        for (int64_t b = 0; b < numBlocks; ++b) {

            const uint64_t offset = offsets[static_cast<size_t>(b)];
            if (offset >= size) {
                std::cerr << "[EXRLoader] chunk offset past end of file" << std::endl;
                return std::nullopt;
            }

            ByteReader cr{bytes, size, static_cast<size_t>(offset)};
            const int32_t y = cr.i32();
            const int32_t dataSize = cr.i32();
            if (!cr.ok || dataSize < 0) {
                std::cerr << "[EXRLoader] truncated chunk header" << std::endl;
                return std::nullopt;
            }
            if (y < h.yMin || y > h.yMax || (y - h.yMin) % linesPerBlock != 0) {
                std::cerr << "[EXRLoader] chunk claims scanline " << y << ", outside the dataWindow" << std::endl;
                return std::nullopt;
            }
            if (!cr.need(static_cast<size_t>(dataSize))) {
                std::cerr << "[EXRLoader] chunk data runs past end of file" << std::endl;
                return std::nullopt;
            }

            const int64_t firstRow = y - h.yMin;
            const int64_t linesInBlock = std::min<int64_t>(linesPerBlock, height - firstRow);
            const size_t uncompressedSize = static_cast<size_t>(linesInBlock) * bytesPerLine;

            const unsigned char* chunk = bytes + cr.pos;
            block.assign(uncompressedSize, 0);

            // A codec that failed to shrink a chunk stores it raw; the reader
            // tells the two apart by size alone, exactly as OpenEXR does. Getting
            // this wrong turns every noisy or tiny image into garbage.
            if (h.compression == Compression::None || static_cast<size_t>(dataSize) >= uncompressedSize) {
                if (static_cast<size_t>(dataSize) < uncompressedSize) {
                    std::cerr << "[EXRLoader] uncompressed chunk is short" << std::endl;
                    return std::nullopt;
                }
                std::memcpy(block.data(), chunk, uncompressedSize);
            } else if (h.compression == Compression::Rle) {
                if (!rleUncompress(chunk, static_cast<size_t>(dataSize), block)) {
                    std::cerr << "[EXRLoader] corrupt RLE chunk" << std::endl;
                    return std::nullopt;
                }
                reconstruct(block, scratch);
            } else if (h.compression == Compression::Piz) {
                // PIZ carries its own transform end to end — no predictor, no
                // byte de-interleave; those belong to ZIP/RLE only.
                if (!detail::pizDecode(chunk, static_cast<size_t>(dataSize), shortsPerSample,
                                       static_cast<int>(width), static_cast<int>(linesInBlock),
                                       block.data(), uncompressedSize)) {
                    std::cerr << "[EXRLoader] corrupt PIZ chunk" << std::endl;
                    return std::nullopt;
                }
            } else {
                const int written = stbi_zlib_decode_buffer(
                        reinterpret_cast<char*>(block.data()), static_cast<int>(uncompressedSize),
                        reinterpret_cast<const char*>(chunk), dataSize);
                if (written != static_cast<int>(uncompressedSize)) {
                    std::cerr << "[EXRLoader] corrupt ZIP chunk (inflate produced " << written
                              << " of " << uncompressedSize << " bytes)" << std::endl;
                    return std::nullopt;
                }
                reconstruct(block, scratch);
            }

            for (int64_t line = 0; line < linesInBlock; ++line) {

                const int64_t fileRow = firstRow + line;
                const int64_t outRow = flipY ? (height - 1 - fileRow) : fileRow;
                float* dst = result.rgba.data() + static_cast<size_t>(outRow * width) * 4;
                const unsigned char* linePtr = block.data() + static_cast<size_t>(line) * bytesPerLine;

                for (size_t ci = 0; ci < h.channels.size(); ++ci) {

                    const int comp = component[ci];
                    if (comp < 0) continue;

                    const PixelType type = h.channels[ci].type;
                    const size_t stride = bytesPerSample(type);
                    const unsigned char* src = linePtr + channelOffset[ci];

                    for (int64_t x = 0; x < width; ++x) {
                        const float v = sampleToFloat(src + static_cast<size_t>(x) * stride, type);
                        if (comp == 4) {
                            dst[x * 4 + 0] = v;
                            dst[x * 4 + 1] = v;
                            dst[x * 4 + 2] = v;
                        } else {
                            dst[x * 4 + comp] = v;
                        }
                    }
                }
            }
        }

        return result;
    }

}// namespace

std::shared_ptr<Texture> EXRLoader::load(const std::filesystem::path& path, bool flipY) {

    if (!std::filesystem::exists(path)) {
        std::cerr << "[EXRLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
        return nullptr;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[EXRLoader] Failed to open '" << path.string() << "'" << std::endl;
        return nullptr;
    }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());

    auto decoded = decode(bytes.data(), bytes.size(), flipY);
    if (!decoded) {
        std::cerr << "[EXRLoader] Failed to load '" << path.string() << "'" << std::endl;
        return nullptr;
    }

    return detail::makeHdrTexture(std::move(decoded->rgba), decoded->width, decoded->height,
                                  path.stem().string());
}

std::shared_ptr<Texture> EXRLoader::loadFromMemory(const std::vector<unsigned char>& data,
                                                   const std::string& name, bool flipY) {

    auto decoded = decode(data.data(), data.size(), flipY);
    if (!decoded) return nullptr;

    return detail::makeHdrTexture(std::move(decoded->rgba), decoded->width, decoded->height, name);
}
